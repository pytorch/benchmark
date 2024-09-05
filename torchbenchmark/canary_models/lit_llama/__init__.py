import os

import torch
from torchbenchmark import add_path, REPO_PATH
from torchbenchmark.tasks import NLP

from ...util.model import BenchmarkModel

LIT_LLAMA_PATH = os.path.join(REPO_PATH, "submodules", "lit-llama")

with add_path(LIT_LLAMA_PATH):
    from lit_llama import LLaMA
    from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup


class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        checkpoint_path = os.path.join(
            LIT_LLAMA_PATH, "checkpoints/lit-llama/7B/lit-llama.pth"
        )
        if not os.path.exists(checkpoint_path):
            raise NotImplementedError("checkpoint doesn't exist")
        with lazy_load(checkpoint_path) as checkpoint:
            name = llama_model_lookup(checkpoint)

            with EmptyInitOnDevice(device=device):
                model = LLaMA.from_name(name)
            model.load_state_dict(checkpoint)

        self.model = model
        self.seq_len = 32
        self.max_seq_len = 64
        self.example_inputs = (
            torch.ones(
                [self.batch_size, self.seq_len], dtype=torch.int32, device=self.device
            ),
            self.max_seq_len,
            torch.arange(
                self.seq_len, dtype=torch.int64, device=self.device
            ),  # positions
        )

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        return NotImplementedError("you will OOM trying to train directly")

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(*self.example_inputs)
        return (logits,)
