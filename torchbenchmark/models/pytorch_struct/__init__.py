"""
pytorch_struct model, Unsupervised CFG task
https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/Unsupervised_CFG.ipynb
"""
import os
import pytest
import torchtext
import numpy as np
import torch, random
import torch_struct
from torch_struct import SentCFG
from .networks.NeuralCFG import NeuralCFG

from .legacy.field import Field
from .legacy.datasets import UDPOS
from .legacy.iterator import BucketIterator

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def TokenBucket(
    train, batch_size, device="cuda:0", key=lambda x: max(len(x.word[0]), 5)
):
    def batch_size_fn(x, _, size):
        return size + key(x)

    return BucketIterator(
        train,
        train=True,
        sort=False,
        sort_within_batch=True,
        shuffle=True,
        batch_size=batch_size,
        sort_key=lambda x: key(x),
        repeat=True,
        batch_size_fn=batch_size_fn,
        device=device,
    )

class Model(BenchmarkModel):
  task = OTHER.OTHER_TASKS

  def __init__(self, device=None, jit=False, train_bs=8):
    super().__init__()
    self.device = device
    self.jit = jit

    WORD = Field(include_lengths=True)
    UD_TAG = Field(init_token="<bos>", eos_token="<eos>", include_lengths=True)

    train, val, test = UDPOS.splits(
      fields=(('word', WORD), ('udtag', UD_TAG), (None, None)), 
      filter_pred=lambda ex: 5 < len(ex.word) < 30
    )

    WORD.build_vocab(train.word, min_freq=3)
    UD_TAG.build_vocab(train.udtag)
    self.train_iter = TokenBucket(train, 
                                  batch_size=200,
                                  device=self.device)

    # Build model
    H = 256
    T = 30
    NT = 30
    self.model = NeuralCFG(len(WORD.vocab), T, NT, H)
    if jit:
        self.model = torch.jit.script(self.model)
    self.model.to(device=device)
    self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=[0.75, 0.999])

  def get_module(self):
    for ex in self.train_iter:
      words, _ = ex.word
      words = words.long()
      return self.model, (words.to(device=self.device).transpose(0, 1),)

  def train(self, niter=1):
    for _, ex in zip(range(niter), self.train_iter):
      losses = []
      self.opt.zero_grad()
      words, lengths = ex.word
      N, batch = words.shape
      words = words.long()
      params = self.model(words.to(self.device).transpose(0, 1))
      dist = SentCFG(params, lengths=lengths)
      loss = dist.partition.mean()
      (-loss).backward()
      losses.append(loss.detach())
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
      self.opt.step()

  def eval(self, niter=1):
    raise NotImplementedError("Eval is not supported by this model")

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

if __name__ == '__main__':
  for device in ['cpu', 'cuda']:
    for jit in [True, False]:
      print("Testing device {}, JIT {}".format(device, jit))
      m = Model(device=device, jit=jit)
      model, example_inputs = m.get_module()
      model(*example_inputs)
      m.train()
