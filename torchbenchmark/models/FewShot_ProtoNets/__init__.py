import torch
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from .experiments.proto_nets import ProtoNets

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 256
    ALLOW_CUSTOMIZE_BSIZE = False
    # run only 1 batch
    NUM_BATCHES = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        if device == "cuda":
          torch.backends.cudnn.benchmark = True
        self.proto_nets = ProtoNets(test=self.test, bs=self.batch_size, num_of_batches=self.NUM_BATCHES, device=self.device, precision=self.dargs.precision,
                                    dataset='omniglot', k_train=20, k_test=20, n_test=5, n_train=5)

    def get_module(self):
        return self.proto_nets.get_module()

    def eval(self):
      self.proto_nets.Eval()

    def train(self):
      self.proto_nets.Train()
