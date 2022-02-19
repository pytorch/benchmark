import torch
import torch.optim as optim
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from transformers import *
from typing import Tuple

class ArgsToKwargsWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ArgsToKwargsWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 4
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        torch.manual_seed(42)
        config = AutoConfig.from_pretrained("facebook/bart-base")
        self.model = AutoModelForSeq2SeqLM.from_config(config).to(device)

        if test =="train":
            input_ids = torch.randint(0, config.vocab_size, (self.batch_size, 512)).to(device)
            decoder_ids = torch.randint(0, config.vocab_size, (self.batch_size, 512)).to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.example_inputs = {'input_ids': input_ids, 'labels': decoder_ids}
            self.model.train()
        elif test == "eval":
            eval_context = torch.randint(0, config.vocab_size, (self.batch_size, 512)).to(device)
            self.model = AutoModelForSeq2SeqLM.from_config(config).to(device)
            self.example_inputs = {'input_ids': eval_context, 'decoder_input_ids': eval_context }
            self.model.eval()

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        k = 'labels' if self.test == 'train' else 'decoder_input_ids'
        return ArgsToKwargsWrapper(self.model), (
                self.example_inputs['input_ids'], self.example_inputs[k])

    def train(self, niter=3):
        if self.jit:
            raise NotImplementedError()
        self.model.train()
        for _ in range(niter):
            outputs = self.model(**self.example_inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1) -> Tuple[torch.Tensor]:
        if self.jit:
            raise NotImplementedError()
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                out = self.model(**self.example_inputs)
        return (out, )
