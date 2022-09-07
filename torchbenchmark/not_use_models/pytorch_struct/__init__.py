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
  DEFAULT_TRAIN_BSIZE = 200
  NUM_OF_BATCHES = 1

  def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
    super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

    WORD = Field(include_lengths=True)
    UD_TAG = Field(init_token="<bos>", eos_token="<eos>", include_lengths=True)

    train, val, test = UDPOS.splits(
      fields=(('word', WORD), ('udtag', UD_TAG), (None, None)), 
      filter_pred=lambda ex: 5 < len(ex.word) < 30
    )

    WORD.build_vocab(train.word, min_freq=3)
    UD_TAG.build_vocab(train.udtag)
    self.iter = TokenBucket(train, batch_size=self.batch_size,
                            device=self.device)

    # Build model
    H = 256
    T = 30
    NT = 30
    self.model = NeuralCFG(len(WORD.vocab), T, NT, H)
    self.model.to(device=device)
    self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=[0.75, 0.999])
    self.example_inputs = _prefetch(self.iter, self.device)

  def get_module(self):
    for words, _ in self.example_inputs:
      return self.model, (words, )

  def train(self):
    for _, (words, lengths) in zip(range(self.NUM_OF_BATCHES), self.example_inputs):
      losses = []
      self.opt.zero_grad()
      params = self.model(words)
      dist = SentCFG(params, lengths=lengths)
      loss = dist.partition.mean()
      (-loss).backward()
      losses.append(loss.detach())
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
      self.opt.step()

  def eval(self):
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
