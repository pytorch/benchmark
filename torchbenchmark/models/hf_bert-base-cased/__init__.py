import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    default_data_collator,
)
import numpy as np
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

    def __init__(self, device=None, jit=False, train_bs=32, task_name="cola"):
        super().__init__()
        self.device = device
        self.jit = jit
        model_name = "bert-base-cased"
        dataset_name = "imdb" 
        max_seq_length = "128"
        learning_rate = "2e-5"
        num_train_epochs = "3"
        output_dir = OUTPUT_DIR
        in_arg = ["--model_name_or_path", model_name, "--task_name", task_name,
                  "--dataset_name", dataset_name,
                  "--do_train", "--do_eval", "--max_seq_length", max_seq_length,
                  "--per_device_train_batch_size", str(train_bs), 
                  "--learning_rate", learning_rate,
                  "--num_train_epochs", num_train_epochs,
                  "--output_dir", OUTPUT_DIR]
        model_args, data_args, training_args = parse_args(in_arg)
        # setup other members
        self.prep(model_args, data_args, training_args)
    
    def prep(self, model_args, data_args, training_args):
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
        train_dataset, eval_dataset, predict_dataset = preprocess_dataset(data_args, training_args, config, model, \
            tokenizer, raw_datasets, num_labels, label_list, is_regression)
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None
        # Setup class members
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        self.raw_datasets = raw_datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.predict_dataset = predict_dataset
        self.data_args = data_args
        self.training_args = training_args
        self.is_regression = is_regression

    def get_module(self):
        raise NotImplementedError("get_module is not supported by this model")

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        if not self.device == "cuda":
            raise NotImplementedError("Only CUDA is supported by this model")
        assert self.training_args.do_train, "Must train with `do_train` arg being set"
        self.trainer.train()
        if self.training_args.do_eval:
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [self.data_args.task_name]
            eval_datasets = [self.eval_dataset]
            if self.data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                eval_datasets.append(self.raw_datasets["validation_mismatched"])

            for eval_dataset, task in zip(eval_datasets, tasks):
                metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        if self.training_args.do_predict:
            # logger.info("*** Predict ***")
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [self.data_args.task_name]
            predict_datasets = [self.predict_dataset]
            if self.data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                predict_datasets.append(self.raw_datasets["test_mismatched"])

            for predict_dataset, task in zip(predict_datasets, tasks):
                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                predict_dataset = predict_dataset.remove_columns("label")
                predictions = self.trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
                predictions = np.squeeze(predictions) if self.is_regression else np.argmax(predictions, axis=1)

    def eval(self, niter=1):
        raise NotImplementedError("Eval is not supported by this model")