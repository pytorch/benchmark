from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import GPT, GPTConfig
from typing import Optional
from dataclasses import dataclass


@dataclass
class GPTTrainingConfig:
    weight_decay: float = 1e-1
    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95

@dataclass
class GPTGenerationConfig:
    max_new_tokens: int = 512  # max number of new tokens to generate
    temperature: float = 1.0  # temperature for sampling. > 1.0: more exploring, < 1.0: more conservative.
    top_k: Optional[int] = None  # top_k > 0: keep only top k tokens with highest probability (top-k filtering).

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)

        prompt_size = 64

        # train params
        self.train_config = GPTTrainingConfig()

        # eval params
        self.generate_config = GPTGenerationConfig(max_new_tokens=32, temperature=0.8, top_k=200)

        # Use the default configs
        self.gpt_config = GPTConfig()
        self.model = GPT(self.gpt_config).to(device)
        self.optimizer = self.model.configure_optimizers(self.train_config.weight_decay, self.train_config.learning_rate, (self.train_config.beta1, self.train_config.beta2), device)
        self.example_inputs = (
            torch.randint(1, self.gpt_config.vocab_size, (self.batch_size, prompt_size)).to(self.device),
        )

        if self.test == "train":
            self.model.train()
        else:
            self.model.eval()

    def get_module(self):
        return self.model, self.example_inputs

    def forward(self):
        logits = self.model(*self.example_inputs)
        loss = logits.sum() / logits.numel()
        return loss

    def backward(self, loss):
        loss.backward()

    def optimizer_step(self):
        self.optimizer.step()

    def eval(self):
        self.model.eval()
        out = self.model.generate(*self.example_inputs, self.generate_config.max_new_tokens, self.generate_config.temperature, self.generate_config.top_k)
        return (out,)
