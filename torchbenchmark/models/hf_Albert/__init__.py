import torch
import torch.optim as optim
from typing import Tuple
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from transformers import *

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 8
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        torch.manual_seed(42)
        config = AutoConfig.from_pretrained("albert-base-v2")
        self.model = AutoModelForMaskedLM.from_config(config).to(device)

        if test =="train":
            input_ids = torch.randint(0, config.vocab_size, (self.batch_size, 512)).to(device)
            decoder_ids = torch.randint(0, config.vocab_size, (self.batch_size, 512)).to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.example_inputs = {'input_ids': input_ids, 'labels': decoder_ids}
            self.model.train()
        elif test == "eval":
            eval_context = torch.randint(0, config.vocab_size, (self.batch_size, 512)).to(device)
            self.example_inputs = {'input_ids': eval_context, }
            self.model.eval()

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.model, (self.example_inputs["input_ids"], )

    def train(self, niter=3):
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            outputs = self.model(**self.example_inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1) -> Tuple[torch.Tensor]:
        if self.jit:
            raise NotImplementedError()
        with torch.no_grad():
            for _ in range(niter):
                out = self.model(**self.example_inputs)
        # logits: prediction scores of language modeling head
        # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/modeling_outputs.py#L455
        return (out.logits, )
