import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING

    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit

        batch_size = 8
        seq_length = 512

        input_ids = torch.randint(0, 5000, (batch_size, seq_length)).to(device)
        decoder_ids = torch.randint(0, 5000, (batch_size, seq_length)).to(device)

        self.model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
        self.optimizer = optim.Adam(self.model.parameters())

        self.train_inputs = {'input_ids': input_ids, 'labels': decoder_ids}
        self.eval_inputs = {'input_ids': input_ids[0:1]}

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.model, self.eval_inputs

    def train(self, niter=3):
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            outputs = self.model(**self.train_inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        self.model(eval_inputs)


if __name__ == "__main__":
    import time
    m = Model(device="cuda")
    module, example_inputs = m.get_module()
    begin = time.time()
    m.train(niter=1)
    print(time.time()-begin)

    begin = time.time()
    m.eval(niter=1)
    print(time.time()-begin)
