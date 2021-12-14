import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from transformers import *
from datasets import load_dataset

torch.manual_seed(1337)
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

    # Original train batch size per device: 8
    # Source: https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_t5_mlm_flax.py#L83
    # Original eval batch size per device: 8
    # Downscale to 2 to fit in Nvidia T4 of the infra
    def __init__(self, device=None, jit=False, train_bs=8, eval_bs=2):
        super().__init__()
        self.device = device
        self.jit = jit

        config = AutoConfig.from_pretrained("t5-base")
        self.model = AutoModelForSeq2SeqLM.from_config(config).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # input_ids = torch.randint(0, config.vocab_size, (train_bs, 1024)).to(device)
        # decoder_ids = torch.randint(0, config.vocab_size, (train_bs, 1024)).to(device)

        eval_context = torch.randint(0, config.vocab_size, (eval_bs, 2048)).to(device)

        # self.train_inputs = {'input_ids': input_ids, 'labels': decoder_ids}
        self.eval_inputs = {'input_ids': eval_context, 'decoder_input_ids': eval_context}

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return ArgsToKwargsWrapper(self.model), (
            self.eval_inputs["input_ids"], self.eval_inputs["decoder_input_ids"])

    # TODO: re-enable train test when infra has capacity
    def train(self, niter=3):
        if self.jit:
            raise NotImplementedError()
        if self.device == "cpu":
            raise NotImplementedError("Disable CPU train test because it is too slow")
        if self.device == "cuda":
            raise NotImplementedError("Disable CUDA train test because limited infra capacity")
        self.model.train()
        for _ in range(niter):
            outputs = self.model(**self.train_inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                out = self.model(**self.eval_inputs)
