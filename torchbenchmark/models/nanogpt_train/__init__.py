from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import GPT, GPTConfig


class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        weight_decay = 1e-1
        learning_rate = 6e-4
        beta1 = 0.9
        beta2 = 0.95
        prompt_size = 64

        gptconf = GPTConfig()
        self.model = GPT(gptconf).to(device)
        self.optimizer = self.model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
        self.example_inputs = (
            torch.randint(1, gptconf.vocab_size, (self.batch_size, prompt_size)).to(self.device),
        )

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        logits = self.model(*self.example_inputs)
        loss = logits.sum() / logits.numel()
        loss.backward()
        self.optimizer.step()

    def eval(self):
        return NotImplementedError("Inference not supported for this model")
