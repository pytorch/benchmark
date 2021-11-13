import torchtext
import torch, random
import numpy as np
import pytest
from .torch_struct import SentCFG
from .torch_struct.networks import NeuralCFG
from .torch_struct import data

from torchbenchmark.util.torchtext_legacy.field import Field
from torchbenchmark.util.torchtext_legacy.datasets import UDPOS
from torchbenchmark.util.torchtext_legacy.iterator import BucketIterator

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class Model(BenchmarkModel):
  task = OTHER.OTHER_TASKS
  def __init__(self, device=None, jit=False):
    super().__init__()
    self.device = device
    self.jit = jit

    # Download and the load default data.
    WORD = Field(include_lengths=True)
    UD_TAG = Field(
        init_token="<bos>", eos_token="<eos>", include_lengths=True
    )

    # Download and the load default data.
    train, val, test = UDPOS.splits(
        fields=(("word", WORD), ("udtag", UD_TAG), (None, None)),
        filter_pred=lambda ex: 5 < len(ex.word) < 30,
    )

    WORD.build_vocab(train.word, min_freq=3)
    UD_TAG.build_vocab(train.udtag)
    self.train_iter = data.TokenBucket(train, batch_size=100, device=device)

    H = 256
    T = 30
    NT = 30
    self.model = NeuralCFG(len(WORD.vocab), T, NT, H)
    if jit:
        self.model = torch.jit.script(self.model)
    self.model.to(device=device)
    self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=[0.75, 0.999])
    for i, ex in enumerate(self.train_iter):
      words, lengths = ex.word
      self.words = words.long().to(device).transpose(0, 1)
      self.lengths = lengths.to(device)
      break

  def get_module(self):
    for ex in self.train_iter:
      words, _ = ex.word
      words = words.long()
      return self.model, (words.to(device=self.device).transpose(0, 1),)

  def train(self, niter=1):
    for _ in range(niter):
      losses = []
      self.opt.zero_grad()
      params = self.model(self.words)
      dist = SentCFG(params, lengths=self.lengths)
      loss = dist.partition.mean()
      (-loss).backward()
      losses.append(loss.detach())
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
      self.opt.step()

  def eval(self, niter=1):
    for _ in range(niter):
      params = self.model(self.words)

def cuda_sync(func, sync=False):
    func()
    if sync:
      torch.cuda.synchronize()

@pytest.mark.parametrize('jit',  [True, False], ids=['jit', 'no-jit'])
@pytest.mark.parametrize('device',  ['cpu', 'cuda'])
class TestBench():
  def test_train(self, benchmark, device, jit):
    m = Model(device=device, jit=jit)
    benchmark(cuda_sync, m.train, device=='cuda')

  def test_eval(self, benchmark, device, jit):
    m = Model(device=device, jit=jit)
    benchmark(cuda_sync, m.eval, device=='cuda')

if __name__ == '__main__':
  for device in ['cpu', 'cuda']:
    for jit in [True, False]:
      print("Testing device {}, JIT {}".format(device, jit))
      m = Model(device=device, jit=jit)
      model, example_inputs = m.get_module()
      model(*example_inputs)
      m.train()
      m.eval()
