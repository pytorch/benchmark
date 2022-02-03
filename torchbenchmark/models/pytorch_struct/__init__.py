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

from torchbenchmark.util.torchtext_legacy.field import Field
from torchbenchmark.util.torchtext_legacy.datasets import UDPOS
from torchbenchmark.util.torchtext_legacy.iterator import BucketIterator

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def _prefetch(loader, device, limit=10):
    data = []
    for _, ex in zip(range(limit), loader):
        words, lengths = ex.word
        words = words.long()
        words = words.to(device).transpose(0, 1)
        data.append((words, lengths))
    return data

def TokenBucket(
    train, batch_size, device, key=lambda x: max(len(x.word[0]), 5)
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

  # Original train batch size: 200
  # Source: https://github.com/harvardnlp/pytorch-struct/blob/f4e374e894b94a9411fb3d2dfb44201a18e37b26/notebooks/Unsupervised_CFG.ipynb
  def __init__(self, test="eval", device=None, jit=False, train_bs=200):
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
                                  batch_size=train_bs,
                                  device=self.device)

    # Build model
    H = 256
    T = 30
    NT = 30
    self.model = NeuralCFG(len(WORD.vocab), T, NT, H)
    self.model.to(device=device)
    self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=[0.75, 0.999])
    self.train_data = _prefetch(self.train_iter, self.device)

  def get_module(self):
    for words, _ in self.train_data:
      return self.model, (words, )

  def train(self, niter=1):
    if self.jit:
        raise NotImplementedError("JIT is not supported by this model")
    for _, (words, lengths) in zip(range(niter), self.train_data):
      losses = []
      self.opt.zero_grad()
      params = self.model(words)
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
