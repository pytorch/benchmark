from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import NLP

from .models import Transformer

class Model(BenchmarkModel):
    task = NLP.TRANSLATION
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
    

    def get_module(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass