import math
import os
import random
from contextlib import nullcontext
from typing import Tuple

import torch

import torch.nn as nn
from torchbenchmark.tasks import NLP
from torchbenchmark.util.model import BenchmarkModel
from transformers import GenerationConfig

from .basic_configs import is_basic_huggingface_models
from .extended_configs import (
    BATCH_SIZE_DIVISORS,
    BATCH_SIZE_KNOWN_MODELS,
    is_extended_huggingface_models,
)


class HuggingFaceModel(BenchmarkModel):
    HF_MODEL = True
    # Default eval precision on CUDA device is fp16(half mode)
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    # If you suffix a model with '_generate', we will instead wrap the
    # unsuffixed model with GenerationWrapper which will make it do
    # autoregressive text generation instead of a probability prediction
    # NB: name is used as kwarg, cannot rename it here
    def __init__(self, name, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        self.name = name
        if name.endswith("_generate"):
            self.is_generate = True
            self.unqual_name = name[: -len("_generate")]
        else:
            self.is_generate = False
            self.unqual_name = name
        name = self.unqual_name  # we don't want to refer to the qualified name anymore
        is_training = self.test == "train"
        if is_basic_huggingface_models(name):
            from .basic_configs import (
                download_model,
                generate_inputs_for_model,
                generate_optimizer_for_model,
            )

            self.model_cls, self.model = download_model(name)
            self.model = self.model.to(self.device)
            self.example_inputs = generate_inputs_for_model(
                self.model_cls,
                self.model,
                name,
                self.batch_size,
                self.device,
                is_training,
            )
            if is_training:
                self.optimizer = generate_optimizer_for_model(self.model, name)
        elif is_extended_huggingface_models(name):
            from .extended_configs import (
                download_model,
                generate_inputs_for_model,
                generate_optimizer_for_model,
            )

            self.model_cls, self.model = download_model(name)
            self.model = self.model.to(self.device)
            self.example_inputs = generate_inputs_for_model(
                self.model_cls,
                self.model,
                name,
                self.batch_size,
                self.device,
                include_loss_args=True,
            )
            if is_training:
                self.optimizer = generate_optimizer_for_model(self.model, name)
        else:
            assert False, f"Huggingface model {name} is not supported yet."

        if is_training:
            self.model.train()
        else:
            self.model.eval()
        self.amp_context = nullcontext

    def get_module(self):
        return self.model, self.example_inputs

    def get_input_iter(self):
        """Yield randomized bucket length of inputs."""
        from .basic_configs import generate_input_iter_for_model

        generator = generate_input_iter_for_model(
            self.model_cls,
            self.model,
            self.unqual_name,
            self.batch_size,
            self.device,
            self.test == "train",
        )
        yield next(generator)

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
        elif hasattr(out, "logits"):
            return (out.logits,)
        else:
            return (out["logits"],)


class HuggingFaceAuthMixin:
    def __init__(self):
        if not "HUGGING_FACE_HUB_TOKEN" in os.environ:
            raise NotImplementedError(
                "Make sure to set `HUGGING_FACE_HUB_TOKEN` so you can download weights"
            )


class HuggingFaceGenerationModel(HuggingFaceModel):
    task = NLP.GENERATION
    DEFAULT_EVAL_BSIZE = 1

    """
    Instead of just running __call__ on the model, use generate to generate
    text.
    """

    def __init__(self, name, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            name=name,
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )
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
        self.example_inputs = (self.example_inputs["input_ids"],)

    def train(self):
        raise NotImplementedError("_generate variant doesn't train")

    def eval(self) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            with self.amp_context():
                out = self.model(*self.example_inputs)
        return (out,)


class GenerationWrapper(nn.Module):
    def __init__(self, model, generation_config):
        super().__init__()
        self.model = model
        self.generation_config = generation_config

    def forward(self, inputs):
        return self.model.generate(inputs, self.generation_config)


class ExtendedHuggingFaceModel(HuggingFaceModel):
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        recorded_batch_size = BATCH_SIZE_KNOWN_MODELS[self.name]
        if self.name in BATCH_SIZE_DIVISORS:
            recorded_batch_size = max(
                int(recorded_batch_size / BATCH_SIZE_DIVISORS[self.name]), 1
            )
        self.DEFAULT_TRAIN_BSIZE = recorded_batch_size
        self.DEFAULT_EVAL_BSIZE = recorded_batch_size
        super().__init__(
            name=self.name,
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )
