from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import SequenceGenerator, create_model
import torch

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        embed_dim = 1536
        beam_size = 1
        # This is quite a bit smaller than, e.g., T5, because this model is
        # quite a bit slower to run
        generate_size = 64
        self.model = SequenceGenerator(
            create_model(embed_dim),
            beam_size,
            generate_size,
        ).eval().to(self.device)
        prompt_size = 64
        vocab_size = 128  # cribbed from original script
        self.example_inputs = (
            torch.randint(1, vocab_size, (self.batch_size, prompt_size)).to(self.device),
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        self.model.train() 

        outputs = self.model(*self.example_inputs)
        
        # This doesn't make semantic sense but works for demonstration.
        loss = self.criterion(outputs, self.example_inputs[0])
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def eval(self):
        out = self.model(*self.example_inputs)
        return (out,)
