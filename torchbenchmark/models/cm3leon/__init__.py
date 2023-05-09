from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import SequenceGenerator, create_model
import torch

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        embed_dim = 1536
        # These are picked to match _generate config on HuggingFace
        beam_size = 1
        generate_size = 256
        self.model = SequenceGenerator(
            create_model(embed_dim),
            beam_size,
            generate_size,
        ).eval().to(self.device)
        prompt_size = 256 # made smaller
        vocab_size = 128  # cribbed from original script
        self.example_inputs = (
            torch.randint(1, vocab_size, (self.batch_size, prompt_size)).to(self.device),
        )

    def get_module(self):
        return self.model, self.example_inputs

    # The code included here is specialized for eval
    def train(self):
        return NotImplementedError("training script not published")

    def eval(self):
        out = self.model(*self.example_inputs)
        return (out,)
