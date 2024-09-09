import importlib.util
import os.path
import sys

import torch.nn as nn
from lit_llama import Tokenizer

from .. import lit_llama as lit_llama
from ..lit_llama import LIT_LLAMA_PATH


def import_from_file_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


lit_llama_generate = import_from_file_path(
    "lit_llama_generate", os.path.join(LIT_LLAMA_PATH, "generate.py")
)


class GenerationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx, max_new_tokens):
        return lit_llama_generate.generate(self.model, idx, max_new_tokens)


class Model(lit_llama.Model):
    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )
        self.model = GenerationWrapper(self.model)
        tokenizer = Tokenizer(
            os.path.join(LIT_LLAMA_PATH, "checkpoints/lit-llama/tokenizer.model")
        )
        # max_new_tokens matches lit-llama/generate.py
        self.example_inputs = (
            tokenizer.encode(
                "The meaning of life is", bos=True, eos=False, device=device
            ),
            50,
        )

    def train(self):
        return NotImplementedError("cannot train on autoregressive generation")

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            y = self.model(*self.example_inputs)
        return (y,)
