import torch
import torch.optim as optim
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from typing import Tuple
from transformers import *

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    HF_MODEL = True
    DEFAULT_TRAIN_BSIZE = 4
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        config = BertConfig()
        self.model = AutoModelForMaskedLM.from_config(config).to(device)
        self.max_length = 512

        if test == "train":
            self.model.train()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            input_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            decoder_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            self.example_inputs = {'input_ids': input_ids, 'labels': decoder_ids}
        elif test == "eval":
            self.model.eval()
            eval_context = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            self.example_inputs = {'input_ids': eval_context, }

    def get_module(self):
        return self.model, (self.example_inputs["input_ids"], )

    def enable_fp16_half(self):
        self.model = self.model.half()

    def train(self, niter=3):
        self.model.train()
        for _ in range(niter):
            outputs = self.model(**self.example_inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1) -> Tuple[torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                out = self.model(**self.example_inputs)
        if hasattr(out, 'logits'):
            return (out.logits, )
        else:
            return (out["logits"], )

    