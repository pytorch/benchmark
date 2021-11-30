import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from torchbenchmark.util.framework.transformers.text_classification.dataset import prep_dataset, preprocess_dataset, prep_labels
from torchbenchmark.util.framework.transformers.text_classification.args import parse_args

torch.manual_seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING

    def __init__(self, device=None, jit=False, train_bs=32):
        super().__init__()
        self.device = device
        self.jit = jit
        model_name = "bert-base-cased"
        dataset_name = "imdb" 
        max_seq_length = 128
        learning_rate = 2e-5
        num_train_epochs = 3
        in_arg = ["--model_name_or_path", model_name, "--dataset_name", dataset_name,
                  "--do_train", "--do_predict", "--max_seq_length", max_seq_length,
                  "--per_device_train_batch_size", train_bs, 
                  "--learning_rate", learning_rate,
                  "--num_train_epochs", num_train_epochs]
        model_args, data_args, training_args = parse_args(in_arg)
        self.prep(model_args, data_args, training_args)
    
    def prep(self, model_args, data_args, training_args):
        raw_datasets = prep_dataset(data_args)
        num_labels, label_list, is_regression = prep_labels(raw_datasets)
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
        prosessed_dataset = preprocess_dataset(training_args, model, tokenizer, raw_datasets, num_labels, label_list, is_regression)

    def get_module(self):
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        
        return self.model, (self.eval_inputs["input_ids"], )

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        if not self.device == "cuda":
            raise NotImplementedError("Only CUDA is supported by this model")

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        if not self.device == "cuda":
            raise NotImplementedError("Only CUDA is supported by this model")