import torch
import math
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchbenchmark.util.e2emodel import E2EBenchmarkModel
from torchbenchmark.tasks import NLP
from datasets import load_metric
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)
from typing import Optional
from torchbenchmark.util.framework.transformers.text_classification.dataset import prep_dataset, preprocess_dataset, prep_labels
from torchbenchmark.util.framework.transformers.text_classification.args import parse_args, parse_torchbench_args

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

class Model(E2EBenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE: int = 32
    DEFAULT_EVAL_BSIZE: int = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        # TODO: currently only support 1 GPU device
        self.device_num = 1
        # TODO: get number of examples
        self.examples = 1
        # Parse the extra arguments
        self.tb_args = parse_torchbench_args(extra_args)
        torch.manual_seed(1337)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Parameters
        model_name = "bert-base-cased"
        max_seq_length = "128"
        learning_rate = "2e-5"
        num_train_epochs = "3"
        # this benchmark runs on a single GPU
        cuda_visible_devices = "0"
        output_dir = os.path.join(CURRENT_DIR, ".output")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        in_arg = ["--model_name_or_path", model_name, "--task_name", self.tb_args.task_name,
                  "--max_length", max_seq_length,
                  "--per_device_train_batch_size", str(self.batch_size), 
                  "--learning_rate", learning_rate,
                  "--num_train_epochs", num_train_epochs,
                  "--output_dir", output_dir]
        hf_args = parse_args(in_arg)
        # setup other members
        self.prep(hf_args)
    
    def prep(self, hf_args):
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        accelerator = Accelerator(fp16=self.tb_args.fp16)
        accelerator.wait_for_everyone()
        raw_datasets = prep_dataset(hf_args)
        num_labels, label_list, is_regression = prep_labels(hf_args, raw_datasets)
        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(hf_args.model_name_or_path, num_labels=num_labels, finetuning_task=hf_args.task_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_args.model_name_or_path, use_fast=not hf_args.use_slow_tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_args.model_name_or_path,
            from_tf=bool(".ckpt" in hf_args.model_name_or_path),
            config=config,)
        train_dataset, eval_dataset, self.mnli_eval_dataset = preprocess_dataset(hf_args, config, model, \
            tokenizer, raw_datasets, num_labels, label_list, is_regression, accelerator)
        # DataLoaders creation:
        if hf_args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            self.data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=self.data_collator, batch_size=hf_args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=self.data_collator, batch_size=hf_args.per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": hf_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=hf_args.learning_rate)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / hf_args.gradient_accumulation_steps)
        if hf_args.max_train_steps is None:
            hf_args.max_train_steps = hf_args.num_train_epochs * num_update_steps_per_epoch
        else:
            hf_args.num_train_epochs = math.ceil(hf_args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=hf_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=hf_args.num_warmup_steps,
            num_training_steps=hf_args.max_train_steps,
        )
        # Steup metrics
        # Get the metric function
        if hf_args.task_name is not None:
            self.metric = load_metric("glue", hf_args.task_name)
        else:
            self.metric = load_metric("accuracy")
        # Setup class members
        self.hf_args = hf_args
        self.is_regression = is_regression
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

    def train(self) -> Optional[dict]:
        completed_steps = 0
        for _epoch in range(self.hf_args.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                loss = loss / self.hf_args.gradient_accumulation_steps
                self.accelerator.backward(loss)
                if step % self.hf_args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    completed_steps += 1

                if completed_steps >= self.hf_args.max_train_steps:
                    break

            self.model.eval()
            for step, batch in enumerate(self.eval_dataloader):
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not self.is_regression else outputs.logits.squeeze()
                self.metric.add_batch(
                    predictions=self.accelerator.gather(predictions),
                    references=self.accelerator.gather(batch["labels"]),
                )
            eval_metric = self.metric.compute()

        if self.hf_args.task_name == "mnli":
            # Final evaluation on mismatched validation set
            eval_dataset = self.mnli_eval_dataset
            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=self.data_collator, batch_size=self.hf_args.per_device_eval_batch_size
            )
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

            self.model.eval()
            for step, batch in enumerate(eval_dataloader):
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                self.metric.add_batch(
                    predictions=self.accelerator.gather(predictions),
                    references=self.accelerator.gather(batch["labels"]),
                )

            eval_metric = self.metric.compute()
        return eval_metric

    def eval(self) -> Optional[dict]:
        self.model.eval()
        for _step, batch in enumerate(self.eval_dataloader):
            outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not self.is_regression else outputs.logits.squeeze()
            self.metric.add_batch(
                    predictions=self.accelerator.gather(predictions),
                    references=self.accelerator.gather(batch["labels"]),
                )
        eval_metric = self.metric.compute()
        return eval_metric