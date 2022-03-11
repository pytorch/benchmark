
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from ..FewShot_ProtoNets.experiments.matching_nets import MatchingNets

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 256
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.matching_nets = MatchingNets(train_bs=self.DEFAULT_TRAIN_BSIZE, eval_bs=self.DEFAULT_EVAL_BSIZE, dataset='omniglot', k_test=20, n_test=5, n_train=5, distance='cosine', fce=True)

    def get_module(self):
        return self.matching_nets.get_module()

    def eval(self, niter=1):
      self.matching_nets.Eval(niter)

    def train(self, niter=1):
      self.matching_nets.Train()
