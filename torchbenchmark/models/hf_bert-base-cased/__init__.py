import torch
import math
from torch.utils.data import DataLoader
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
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
import os
from pathlib import Path

from torchbenchmark.util.framework.transformers.text_classification.dataset import prep_dataset, preprocess_dataset, prep_labels
from torchbenchmark.util.framework.transformers.text_classification.args import parse_args

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
OUTPUT_DIR = os.path.join(CURRENT_DIR, ".output")

torch.manual_seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING

    def __init__(self, device=None, jit=False, train_bs=32, task_name="cola", run_style="single-batch"):
        super().__init__()
        self.device = device
        self.jit = jit
        model_name = "bert-base-cased"
        max_seq_length = "128"
        learning_rate = "2e-5"
        num_train_epochs = "3"
        output_dir = OUTPUT_DIR
        in_arg = ["--model_name_or_path", model_name, "--task_name", task_name,
                  "--do_train", "--do_eval", "--max_seq_length", max_seq_length,
                  "--per_device_train_batch_size", str(train_bs), 
                  "--learning_rate", learning_rate,
                  "--num_train_epochs", num_train_epochs,
                  "--output_dir", OUTPUT_DIR]
        model_args, data_args, training_args = parse_args(in_arg)
        # setup other members
        self.prep(model_args, data_args, training_args)
    
    def prep(self, model_args, data_args, training_args):
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        accelerator = Accelerator()
        accelerator.wait_for_everyone()
        raw_datasets = prep_dataset(data_args, training_args)
        num_labels, label_list, is_regression = prep_labels(data_args, raw_datasets)
        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        train_dataset, eval_dataset, _predict_dataset = preprocess_dataset(data_args, training_args, config, model, \
            tokenizer, raw_datasets, num_labels, label_list, is_regression)
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
        if training_args.max_steps is None or training_args.max_steps == -1:
            training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        else:
            training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )
        # Setup class members
        self.training_args = training_args
        self.is_regression = is_regression
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr_scheduler = lr_scheduler

    def get_module(self):
        raise NotImplementedError("get_module is not supported by this model")

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        if not self.device == "cuda":
            raise NotImplementedError("Only CUDA is supported by this model")
        assert self.training_args.do_train, "Must train with `do_train` arg being set"
        completed_steps = 0
        for _epoch in range(self.training_args.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                loss = loss / self.training_args.gradient_accumulation_steps
                self.accelerator.backward(loss)
                if step % self.training_args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.completed_steps += 1

                if completed_steps >= self.training_args.max_steps:
                    break

            self.model.eval()
            for step, batch in enumerate(self.eval_dataloader):
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not self.is_regression else outputs.logits.squeeze()

    def eval(self, niter=1):
        raise NotImplementedError("Eval is not supported by this model")