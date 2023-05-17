from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import GPT, SequenceGeneratorNanoGPT, GPTConfig, GPTGenerationConfig


class Model(BenchmarkModel):
    task = NLP.GENERATION
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        # Use the default configs
        self.gpt_config = GPTConfig()
        self.generator_config = GPTGenerationConfig(32, 0.8, 200)
        self.model = SequenceGeneratorNanoGPT(GPT(self.gpt_config), self.generator_config).eval().to(self.device)
        self.prompt_size = 64
        self.example_inputs = (
            torch.randint(1, self.gpt_config.vocab_size, (self.batch_size, self.prompt_size)).to(self.device),
        )

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        return NotImplementedError("Training not supported for this model")

    def eval(self):
        with torch.no_grad():
            out = self.model(*self.example_inputs)
        return (out,)
