import torch
import torch.optim as optim
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from transformers import *
from typing import Tuple

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class ArgsToKwargsWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ArgsToKwargsWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    HF_MODEL = True
    # Original train batch size per device: 8
    # Source: https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_t5_mlm_flax.py#L83
    DEFAULT_TRAIN_BSIZE = 8
    # Original eval batch size per device: 8
    # Downscale to 1 to fit in Nvidia T4 of the infra
    DEFAULT_EVAL_BSIZE = 1
    
    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        config = AutoConfig.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_config(config).to(device)
        if test == "train":
            self.max_length = 1024
            self.model.train()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            input_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            decoder_ids = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            self.example_inputs = {'input_ids': input_ids, 'labels': decoder_ids}
        elif test == "eval":
            self.model.eval()
            self.max_length = 2048
            eval_context = torch.randint(0, config.vocab_size, (self.batch_size, self.max_length)).to(device)
            self.example_inputs = {'input_ids': eval_context, 'decoder_input_ids': eval_context }

    def get_module(self):
        k = 'labels' if self.test == 'train' else 'decoder_input_ids'
        return ArgsToKwargsWrapper(self.model), (
                self.example_inputs['input_ids'], self.example_inputs[k])

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