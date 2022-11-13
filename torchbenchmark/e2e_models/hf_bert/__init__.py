from accelerate.utils.dataclasses import DeepSpeedPlugin
import torch
import torch._dynamo
import math
import os
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torchbenchmark.util.e2emodel import E2EBenchmarkModel
from torchbenchmark.tasks import NLP
import evaluate
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

    def __init__(self, test, batch_size=None, extra_args=[]):
        super().__init__(test=test, batch_size=batch_size, extra_args=extra_args)
        # TODO: currently only support 1 GPU device
        self.device = "cuda"
        self.device_num = 1
        # Parse the extra arguments
        self.tb_args = parse_torchbench_args(self.extra_args)
        torch.manual_seed(1337)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Parameters
        model_name = "bert-base-cased"
        max_seq_length = "128"
        learning_rate = "2e-5"
        num_train_epochs = "3"
        max_train_steps = "100" # overrides num_train_epochs to run faster
        # this benchmark runs on a single GPU
        cuda_visible_devices = "0"
        output_dir = os.path.join(CURRENT_DIR, ".output")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        in_arg = ["--model_name_or_path", model_name, "--task_name", self.tb_args.task_name,
                  "--max_length", max_seq_length,
                  "--per_device_train_batch_size", str(self.batch_size), 
                  "--per_device_eval_batch_size", str(self.batch_size),
                  "--learning_rate", learning_rate,
                  "--num_train_epochs", num_train_epochs,
                  "--max_train_steps", max_train_steps,
                  "--output_dir", output_dir]
        hf_args = parse_args(in_arg)

        # ideally we don't modify the model code directly, but attaching deepspeed
        # must be done before self.prep initializes accelerator.
        if self.tb_args.distributed not in ["deepspeed", "ddp", "fsdp", "none"]:
            raise RuntimeError(f"Unsupported distributed scheme {self.tb_args.distributed} for model hf_t5")
        if self.tb_args.distributed == "deepspeed":
            zero_opt_cfg = {
                "zero_optimization": {
                    "stage": 1,
                    "reduce_bucket_size": 2e8,
                    "overlap_comm": True,
                    "contiguous_gradients": False
                }
            }
            hf_args.deepspeed_plugin = DeepSpeedPlugin()
            hf_args.deepspeed_plugin.deepspeed_config.update(zero_opt_cfg)
        hf_args.distributed = self.tb_args.distributed # pass in distributed config to prep as a hf_arg

        # setup other members
        self.prep(hf_args)
        if test == "train":
            self.num_examples = len(self.train_dataloader) * self.batch_size
        elif test == "eval":
            self.num_examples = len(self.eval_dataloader) * self.batch_size
    
    def prep(self, hf_args):
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        if hf_args.distributed == "deepspeed":
            # Note: self.tb_args.fp16 could be renamed to better clarify its meaning
            assert self.tb_args.fp16=="amp", "deepspeed is only supported with bf16/amp enabled"
            accelerator = Accelerator(deepspeed_plugin=hf_args.deepspeed_plugin, mixed_precision='bf16')
        else:
            accelerator = Accelerator(mixed_precision='fp16' if self.tb_args.fp16=='amp' else 'no')
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

        # transform model for DDP and FSDP
        if hf_args.distributed == "ddp":
            # prepare before wrap w/ DDP (or else error)
            model = accelerator.prepare(model)
            local_rank = int(os.getenv("LOCAL_RANK", -1))
            model = DDP(
                model,
                device_ids=[local_rank],
                # If buffer broadcast is necessary, specific optimizations might be
                # necessary to optimize performance. Disable it by default.
                broadcast_buffers=False,
                # Set gradient as bucket view to avoid unnecessary copies
                gradient_as_bucket_view=True,
                # TODO: tune bucket_cap_mb
                static_graph=True,
            )
        elif hf_args.distributed == "fsdp":
            # model needs to be prepared and wrapped w/ FSDP before optimizer is created, because FSDP flattens params
            model = accelerator.prepare(model)
            local_rank = int(os.getenv("LOCAL_RANK", -1))
            torch.cuda.set_device(local_rank)
            model = FSDP(
                model,
                device_id = torch.cuda.current_device()
            )

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

        # Prepare everything with our `accelerator` with deepspeed or non-distributed environment.
        if hf_args.distributed == "deepspeed" or hf_args.distributed == "none":
            # deepspeed will error unless all components prepared at the same time
            model, train_dataloader, eval_dataloader, optimizer = accelerator.prepare(model, train_dataloader, eval_dataloader, optimizer)
        else:
             # ddp and fsdp need model prepared before wrapping.
            train_dataloader, eval_dataloader, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, optimizer)
            

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
            self.metric = evaluate.load("glue", hf_args.task_name)
        else:
            self.metric = evaluate.load("accuracy")
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
        eval_metric = None
        for _epoch in range(self.hf_args.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                loss = self.run_forward(batch)
                loss = loss / self.hf_args.gradient_accumulation_steps
                self.run_backward(loss)
                if step % self.hf_args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    self.run_optimizer_step()
                    completed_steps += 1

                if completed_steps >= self.hf_args.max_train_steps:
                    break
            if self.tb_args.validate_in_train:
                self.model.eval()
                for step, batch in enumerate(self.eval_dataloader):
                    outputs = self.run_eval(batch)
                    predictions = outputs.logits.argmax(dim=-1) if not self.is_regression else outputs.logits.squeeze()
                    self.metric.add_batch(
                        predictions=self.accelerator.gather(predictions),
                        references=self.accelerator.gather(batch["labels"]),
                    )
                eval_metric = self.metric.compute()
        if self.tb_args.validate_in_train:
            if self.hf_args.task_name == "mnli":
                # Final evaluation on mismatched validation set
                eval_dataset = self.mnli_eval_dataset
                eval_dataloader = DataLoader(
                    eval_dataset, collate_fn=self.data_collator, batch_size=self.hf_args.per_device_eval_batch_size
                )
                eval_dataloader = self.accelerator.prepare(eval_dataloader)

                self.model.eval()
                for step, batch in enumerate(eval_dataloader):
                    outputs = self.run_eval(batch)
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
            with torch.no_grad():
                outputs = self.run_eval(batch)
            predictions = outputs.logits.argmax(dim=-1) if not self.is_regression else outputs.logits.squeeze()
            self.metric.add_batch(
                    predictions=self.accelerator.gather(predictions),
                    references=self.accelerator.gather(batch["labels"]),
                )
        eval_metric = self.metric.compute()
        return eval_metric

    def next_batch(self):
        return next(iter(self.train_dataloader))

    def run_forward(self, input):
        """
        compute model forward and return loss
        """
        if self.dynamo:
            backend = self.opt_args.torchdynamo
            return torch._dynamo.optimize(backend)(self._run_forward)(input)
        else:
            return self._run_forward(input)

    def _run_forward(self, input):
        return self.model(**input).loss

    def run_backward(self, loss):
        if self.dynamo:
            backend = self.opt_args.torchdynamo
            return torch._dynamo.optimize(backend)(self._run_backward)(loss)
        else:
            return self._run_backward(loss)

    def _run_backward(self, loss):
        self.accelerator.backward(loss)

    def run_optimizer_step(self):
        if self.dynamo:
            backend = self.opt_args.torchdynamo
            return torch._dynamo.optimize(backend)(self._run_optimizer_step)()
        else:
            return self._run_optimizer_step()

    def _run_optimizer_step(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def run_eval(self, input):
        if self.dynamo:
            backend = self.opt_args.torchdynamo
            return torch._dynamo.optimize(backend)(self._run_eval)(input)
        else:
            return self._run_eval(input)

    def _run_eval(self, input):
        return self.model(**input)
