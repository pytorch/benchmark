
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from ..FewShot_ProtoNets.experiments.maml import MAML

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 256
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.maml = MAML(train_bs=self.DEFAULT_TRAIN_BSIZE, eval_bs=self.DEFAULT_EVAL_BSIZE, dataset='omniglot',
          order=1, n=5, k=5, eval_batches=10, epoch_len=batch_size)

    def get_module(self):
        return self.maml.get_module()

    def eval(self, niter=1):
      self.maml.Eval(niter)

    def train(self, niter=1):
      self.maml.Train()
