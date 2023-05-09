from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import GPT, GPTConfig


class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        # Use the default config
        self.config = GPTConfig()
        self.model = GPT(self.config).to(device)
        self.temperature = 0.8
        self.max_new_tokens = 500
        self.top_k = 200
        self.prompt_size = 64
        self.example_inputs = (
            torch.randint(1, self.config.vocab_size, (self.batch_size, self.prompt_size)).to(self.device),
        )

    def get_module(self):
        return self.model, self.example_inputs

    # The code included here is specialized for eval
    def train(self):
        return NotImplementedError("training script not published")

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            y = self.model.generate(*self.example_inputs, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
        return (y,)

    def get_optimizer():
        return super().get_optimizer()

    def set_optimizer(optimizer):
        return super().get_optimizer(optimizer)
