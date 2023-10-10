import math
import random
import os
import torch
from contextlib import nullcontext
from torch import optim
import torch.nn as nn
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import transformers
from transformers import AutoConfig, ReformerConfig, BertConfig, GenerationConfig, WhisperConfig, LlamaConfig
from typing import Tuple

class_models = {
    # 'name': (train_max_length, eval_max_length, config, model)
    'hf_GPT2': (512, 1024, 'AutoConfig.from_pretrained("gpt2")', 'AutoModelForCausalLM'),
    'hf_GPT2_large': (512, 1024, 'AutoConfig.from_pretrained("gpt2-large")', 'AutoModelForCausalLM'),
    'hf_T5': (1024, 2048, 'AutoConfig.from_pretrained("t5-small")', 'AutoModelForSeq2SeqLM'),
    'hf_T5_base': (1024, 2048, 'AutoConfig.from_pretrained("t5-base")', 'AutoModelForSeq2SeqLM'),
    'hf_T5_large': (512, 512, 'AutoConfig.from_pretrained("t5-large")', 'AutoModelForSeq2SeqLM'),
    'hf_Bart': (512, 512, 'AutoConfig.from_pretrained("facebook/bart-base")', 'AutoModelForSeq2SeqLM'),
    'hf_Reformer': (4096, 4096, 'ReformerConfig()', 'AutoModelForMaskedLM'),
    'hf_BigBird': (1024, 4096, 'BigBirdConfig(attention_type="block_sparse",)', 'AutoModelForMaskedLM'),
    'hf_Albert': (512, 512, 'AutoConfig.from_pretrained("albert-base-v2")', 'AutoModelForMaskedLM'),
    'hf_DistilBert': (512, 512, 'AutoConfig.from_pretrained("distilbert-base-uncased")', 'AutoModelForMaskedLM'),
    'hf_Longformer': (1024, 4096, 'AutoConfig.from_pretrained("allenai/longformer-base-4096")', 'AutoModelForMaskedLM'),
    'hf_Bert': (512, 512, 'BertConfig()', 'AutoModelForMaskedLM'),
    # see https://huggingface.co/bert-large-cased
    'hf_Bert_large': (512, 512, 'BertConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16)', 'AutoModelForMaskedLM'),
    'hf_Whisper': (1024, 1024, 'WhisperConfig()', 'AutoModelForAudioClassification'),
    # default num_hidden_layers=32 but that OOMs, feel free to change this config to something more real
    'llama_v2_7b_16h' : (128,512, 'LlamaConfig(num_hidden_layers=16)', 'AutoModelForCausalLM'),
    'hf_MPT_7b_instruct': (512, 512, 'AutoConfig.from_pretrained("mosaicml/mpt-7b-instruct", trust_remote_code=True)', 'AutoModelForCausalLM'),
    'llama_v2_7b' : (512,512, 'AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")', 'AutoModelForCausalLM'),
    'llama_v2_13b' : (512,512, 'AutoConfig.from_pretrained("meta-llama/Llama-2-13b-hf")', 'AutoModelForCausalLM'),
    'llama_v2_70b' : (512, 512, 'AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")', 'AutoModelForMaskedLM'),
}

cpu_input_slice = {
    'hf_BigBird': 5,
    'hf_Longformer': 8,
    'hf_T5': 4,
    'hf_GPT2': 4,
    'hf_Reformer': 2,
}

class ArgsToKwargsWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ArgsToKwargsWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

class HuggingFaceModel(BenchmarkModel):
    HF_MODEL = True
    # Default eval precision on CUDA device is fp16(half mode)
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    # If you suffix a model with '_generate', we will instead wrap the
    # unsuffixed model with GenerationWrapper which will make it do
    # autoregressive text generation instead of a probability prediction
    # NB: name is used as kwarg, cannot rename it here
    def __init__(self, name, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)

        self.name = name
        if name.endswith('_generate'):
            self.is_generate = True
            self.unqual_name = name[:-len('_generate')]
        else:
            self.is_generate = False
            self.unqual_name = name
        name = self.unqual_name  # we don't want to refer to the qualified name anymore
        if test == "train":
            self.max_length = class_models[name][0]
        elif test == "eval":
            self.max_length = class_models[name][1]
        # workaround the bigbird config import
        if name == "hf_BigBird":
            from transformers import BigBirdConfig
        config = eval(class_models[name][2])
        if class_models[name][2] == "ReformerConfig()" and not config.num_buckets:
            # silence "config.num_buckets is not set. Setting config.num_buckets to 128"
            config.num_buckets = 128
        class_ctor = getattr(transformers, class_models[name][3])
        kwargs = {}
        if name == "hf_Falcon_7b" or name == "hf_MPT_7b_instruct":
            kwargs["trust_remote_code"] = True
        self.model = class_ctor.from_config(config, **kwargs).to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            # TODO resolve https://github.com/pytorch/torchdynamo/issues/1083
            capturable=bool(int(os.getenv("ADAM_CAPTURABLE", 0)
        )))

        if name in ["llama_v2_7b_16h"]:
            self.optimizer = optim.SGD(self.model.parameters(), lr= 0.001)

        # populate these on-demand to avoid wasting memory when not used
        self.vocab_size = config.vocab_size

        if test == "train":
            input_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            decoder_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            self.example_inputs = {'input_ids': input_ids, 'labels': decoder_ids}
            self.model.train()
        elif test == "eval":
            # Cut the length of sentence when running on CPU, to reduce test time
            if self.device == "cpu" and name in cpu_input_slice:
                self.max_length = int(self.max_length / cpu_input_slice[name])
            eval_context = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            self.example_inputs = {'input_ids': eval_context, }
            if class_models[name][3] == 'AutoModelForSeq2SeqLM':
                self.example_inputs['decoder_input_ids'] = eval_context
            self.model.eval()
        self.amp_context = nullcontext

    def get_module(self, wrap_model=True):
        if not self.is_generate and class_models[self.unqual_name][3] == 'AutoModelForSeq2SeqLM':
            k = 'labels' if self.test == 'train' else 'decoder_input_ids'
            if not wrap_model:
                return self.model, (
                    self.example_inputs['input_ids'], self.example_inputs[k])
            return ArgsToKwargsWrapper(self.model), (
                    self.example_inputs['input_ids'], self.example_inputs[k])
        return self.model, (self.example_inputs["input_ids"], )

    def get_input_iter(self):
        """Yield randomized bucket length of inputs."""
        nbuckets = 8
        n = int(math.log2(self.max_length))
        buckets = [2**n for n in range(n - nbuckets, n)]
        if class_models[self.unqual_name][3] == 'AutoModelForSeq2SeqLM':
            raise NotImplementedError("AutoModelForSeq2SeqLM is not yet supported")
        while True:
            # randomize bucket_len
            bucket_len = random.choice(buckets)
            dict_input = {
                'input_ids': torch.randint(0, self.vocab_size, (self.batch_size, bucket_len)).to(self.device),
                'labels': torch.randint(0, self.vocab_size, (self.batch_size, bucket_len)).to(self.device),
            }
            yield dict_input

    def forward(self):
        with self.amp_context():
            outputs = self.model(**self.example_inputs)
        return outputs.loss

    def backward(self, losses):
        losses.backward()

    def optimizer_step(self):
        self.optimizer.step()

    def eval(self) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            with self.amp_context():
                out = self.model(**self.example_inputs)
        # logits: prediction scores of language modeling head
        # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/modeling_outputs.py#L455
        # transformations such as fx2trt will cast the original output type to dict
        if isinstance(out, tuple):
            return out
        elif hasattr(out, 'logits'):
            return (out.logits, )
        else:
            return (out["logits"], )

class HuggingFaceAuthMixin:
    def __init__(self):
        if not 'HUGGING_FACE_HUB_TOKEN' in os.environ:
            raise NotImplementedError("Make sure to set `HUGGING_FACE_HUB_TOKEN` so you can download weights")


class HuggingFaceGenerationModel(HuggingFaceModel):
    task = NLP.GENERATION
    DEFAULT_EVAL_BSIZE = 1

    """
    Instead of just running __call__ on the model, use generate to generate
    text.
    """
    def __init__(self, name, test, device, batch_size=None, extra_args=[]):
        super().__init__(name=name, test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        # Make this configurable with extra_args
        # NB: this is *fixed* generation size as eos_token_id is None
        # These params were cribbed off of
        # https://github.com/younesbelkada/hf-torch-compile-benchmark
        generation_config = GenerationConfig(
            max_new_tokens=256,
            pad_token_id=0,
            eos_token_id=None,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
        self.model = GenerationWrapper(self.model, generation_config)

    def train(self):
        raise NotImplementedError("_generate variant doesn't train")

    def eval(self) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            with self.amp_context():
                out = self.model(self.example_inputs['input_ids'])
        return (out,)


class GenerationWrapper(nn.Module):
    def __init__(self, model, generation_config):
        super().__init__()
        self.model = model
        self.generation_config = generation_config

    def forward(self, inputs):
        return self.model.generate(inputs, self.generation_config)
