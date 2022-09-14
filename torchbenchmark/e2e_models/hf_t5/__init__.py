from accelerate.utils.dataclasses import DeepSpeedPlugin
import functools
import torch
import numpy as np
import math
import os
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torchbenchmark.util.e2emodel import E2EBenchmarkModel
from torchbenchmark.tasks import NLP

import evaluate
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    get_scheduler,
    MBartTokenizer,
    MBartTokenizerFast
)
from transformers.models.t5.modeling_t5 import T5Block
from torchbenchmark.util.framework.transformers.translation.dataset import prep_dataset, preprocess_dataset
from torchbenchmark.util.framework.transformers.translation.args import parse_args, parse_torchbench_args, task_to_keys

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

class Model(E2EBenchmarkModel):
    task = NLP.TRANSLATION
    DEFAULT_TRAIN_BSIZE: int = 32
    DEFAULT_EVAL_BSIZE: int = 1

    def __init__(self, test, batch_size=None, extra_args=[]):
        super().__init__(test=test, batch_size=batch_size, extra_args=extra_args)
        self.device = "cuda"
        self.device_num = 1
        # Parse the extra arguments
        self.tb_args = parse_torchbench_args(extra_args)
        torch.manual_seed(1337)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Parameters
        model_name = "t5-base"
        max_source_length = "1024"
        max_target_length = "128"
        learning_rate = "2e-5"
        num_train_epochs = "3" # this takes a rather long time for wmt-en-ro
        max_train_steps = "100" # overrides num_train_epochs to run faster
        checkpointing_steps = None # set to a string value, like "1000"

        task_name = self.tb_args.task_name
        task_args = task_to_keys[task_name] # dataset specific hf_args
        # T5 requires source prefix to know what to translate
        if task_name == "wmt-en-ro":
            source_prefix = "translate English to Romanian: "
        elif task_name == "wmt-en-de":
            source_prefix = "translate English to German: "
        else:
            raise RuntimeError(f"Unsupported translation task {task_name} for model hf_t5")
        task_args.extend(["--source_prefix", source_prefix])

        # this benchmark runs on a single GPU
        cuda_visible_devices = "0"
        output_dir = os.path.join(CURRENT_DIR, ".output")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        in_arg = ["--model_name_or_path", model_name, 
                  "--max_source_length", max_source_length,
                  "--max_target_length", max_target_length,
                  "--per_device_train_batch_size", str(self.batch_size), 
                  "--per_device_eval_batch_size", str(self.batch_size),
                  "--learning_rate", learning_rate,
                  "--num_train_epochs", num_train_epochs,
                  "--max_train_steps", max_train_steps,
                  "--checkpointing_steps", checkpointing_steps,
                  "--output_dir", output_dir]
        in_arg.extend(task_args)
        hf_args = parse_args(in_arg)

        # ideally we don't modify the model code directly, but attaching deepspeed
        # must be done before self.prep initialiazes accelerator.
        hf_args.distributed = self.tb_args.distributed
        # supported distributed backends
        if hf_args.distributed not in ["deepspeed", "ddp", "fsdp", "none"]:
            raise RuntimeError(f"Unsupported distributed scheme {self.tb_args.distributed} for model hf_t5")
        # prep args for any distributed backend that needs it
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

        # Handle the repository creation
        if accelerator.is_main_process:
            if hf_args.output_dir is not None:
                os.makedirs(hf_args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        raw_datasets = prep_dataset(hf_args)

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        if hf_args.config_name:
            config = AutoConfig.from_pretrained(hf_args.config_name)
        elif hf_args.model_name_or_path:
            config = AutoConfig.from_pretrained(hf_args.model_name_or_path)
        else: 
            config = CONFIG_MAPPING[hf_args.model_type]()
            # logger.warning("You are instantiating a new config instance from scratch.")

        if hf_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(hf_args.tokenizer_name, use_fast=not hf_args.use_slow_tokenizer)
        elif hf_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(hf_args.model_name_or_path, use_fast=not hf_args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        if hf_args.model_name_or_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                hf_args.model_name_or_path,
                from_tf=bool(".ckpt" in hf_args.model_name_or_path),
                config=config,
            )
        else:
            # logger.info("Training new model from scratch")
            model = AutoModelForSeq2SeqLM.from_config(config)

        model.resize_token_embeddings(len(tokenizer))

        # Set decoder_start_token_id
        if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            assert (
                hf_args.target_lang is not None and hf_args.source_lang is not None
            ), "mBart requires --target_lang and --source_lang"
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[hf_args.target_lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(hf_args.target_lang)

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        # For translation we set the codes of our source and target languages (only useful for mBART, the others will
        # ignore those attributes).
        if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if hf_args.source_lang is not None:
                tokenizer.src_lang = hf_args.source_lang
            if hf_args.target_lang is not None:
                tokenizer.tgt_lang = hf_args.target_lang

        prefix = hf_args.source_prefix if hf_args.source_prefix is not None else ""

        train_dataset, eval_dataset = preprocess_dataset(hf_args, raw_datasets, tokenizer, prefix, accelerator)

        # # Log a few random samples from the training set:
        # for index in random.sample(range(len(train_dataset)), 3):
        #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            
        # DataLoaders creation:
        label_pad_token_id = -100 if hf_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        if hf_args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            self.data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if accelerator.use_fp16 else None,
            )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=self.data_collator, batch_size=hf_args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=self.data_collator, batch_size=hf_args.per_device_eval_batch_size)
        
        # set distributed strategy before creating optimizer
        if hf_args.distributed == "ddp":
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
            model = accelerator.prepare(model)
            transformer_auto_wrapper_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    T5Block,
                },
            )
            local_rank = int(os.getenv("LOCAL_RANK", -1))
            torch.cuda.set_device(local_rank)
            model = FSDP(
                model,
                # TODO: seems to make benchmark slower? and profile doesn't work? investigate
                # auto_wrap_policy=transformer_auto_wrapper_policy,
                device_id = torch.cuda.current_device()
            )

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=hf_args.learning_rate)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / hf_args.gradient_accumulation_steps)
        if hf_args.max_train_steps is None:
            hf_args.max_train_steps = hf_args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=hf_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=hf_args.num_warmup_steps,
            num_training_steps=hf_args.max_train_steps,
        )

        # Prepare everything with our `accelerator`.
        if hf_args.distributed == "deepspeed":
            # deepspeed will error unless all components prepared at the same time
            model, train_dataloader, eval_dataloader, optimizer = accelerator.prepare(model, train_dataloader, eval_dataloader, optimizer)
        else:
             # ddp and fsdp need model prepared before wrapping.
            train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(train_dataloader, eval_dataloader, optimizer, lr_scheduler)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / hf_args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            hf_args.max_train_steps = hf_args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        hf_args.num_train_epochs = math.ceil(hf_args.max_train_steps / num_update_steps_per_epoch)
        # Figure out how many steps we should save the Accelerator states
        if hasattr(hf_args.checkpointing_steps, "isdigit"):
            hf_args.checkpointing_steps = hf_args.checkpointing_steps
            if hf_args.checkpointing_steps.isdigit():
                hf_args.checkpointing_steps = int(hf_args.checkpointing_steps)
        else:
            hf_args.checkpointing_steps = None

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]

            return preds, labels

        metric = evaluate.load("sacrebleu")

        # Setup class members
        self.hf_args = hf_args
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.metric = metric
        self.config = config
        self.postprocess_text = postprocess_text

    def train(self):
        completed_steps = 0
        eval_metric = None
        for epoch in range(self.hf_args.num_train_epochs):
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

                if isinstance(self.hf_args.checkpointing_steps, int):
                    if completed_steps % self.hf_args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if self.hf_args.output_dir is not None:
                            output_dir = os.path.join(self.hf_args.output_dir, output_dir)
                        self.accelerator.save_state(output_dir)

                if completed_steps >= self.hf_args.max_train_steps:
                    break
            if self.tb_args.validate_in_train:
                eval_metric = self.eval() # run evaluation 
        return eval_metric
    
    def eval(self):
        self.model.eval()

        if self.hf_args.val_max_target_length is None:
            self.hf_args.val_max_target_length = self.hf_args.max_target_length

        gen_kwargs = {
            "max_length": self.hf_args.val_max_target_length if self.hf_args is not None else self.config.max_length,
            "num_beams": self.hf_args.num_beams,
        }
        samples_seen = 0
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not self.hf_args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = self.accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)

                generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
                labels = self.accelerator.gather(labels).cpu().numpy()

                if self.hf_args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

                # If we are in a multiprocess environment, the last batch has duplicates
                if self.accelerator.num_processes > 1:
                    if step == len(self.eval_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(self.eval_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(self.eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)

                self.metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        eval_metric = self.metric.compute()
        # logger.info({"bleu": eval_metric["score"]})
        return eval_metric

    def next_batch(self):
        return next(iter(self.train_dataloader))

    def run_forward(self, input):
        """
        compute model forward and return loss
        """
        return self.model(**input).loss

    def run_backward(self, loss):
        self.accelerator.backward(loss)

    def run_optimizer_step(self):
        self.optimizer.step()

