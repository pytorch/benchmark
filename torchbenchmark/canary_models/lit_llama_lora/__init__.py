from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from ..lit_llama import LIT_LLAMA_PATH
import os.path
import sys
from lit_llama.lora import mark_only_lora_as_trainable, lora
from torchbenchmark import REPO_PATH

LIT_LLAMA_PATH = os.path.join(REPO_PATH, "submodules", "lit-llama")

sys.path.insert(0, LIT_LLAMA_PATH)

from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from lit_llama import LLaMA


class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 1
    DEFAULT_TRAIN_BSIZE = 4  # micro_batch_size in lora.py

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        # From finetune/lora.py hyperparameters
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.05

        checkpoint_path = os.path.join(
            LIT_LLAMA_PATH, "checkpoints/lit-llama/7B/lit-llama.pth"
        )
        if not os.path.exists(checkpoint_path):
            raise NotImplementedError("checkpoint doesn't exist")
        with lazy_load(checkpoint_path) as checkpoint, lora(
            r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True
        ):
            name = llama_model_lookup(checkpoint)

            with EmptyInitOnDevice(device=device):
                model = LLaMA.from_name(name)
            # LoRA weights won't be in base checkpoint
            model.load_state_dict(checkpoint, strict=False)

        mark_only_lora_as_trainable(model)

        self.model = model
        self.seq_len = 32
        self.max_seq_len = 64
        self.example_inputs = (
            torch.ones(
                [self.batch_size, self.seq_len], dtype=torch.int32, device=self.device
            ),
            self.max_seq_len,
        )

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        logits = self.model(*self.example_inputs)
        logits.sum().backward()
        # meh this sucks

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(*self.example_inputs)
        return (logits,)
