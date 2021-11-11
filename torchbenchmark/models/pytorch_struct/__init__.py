"""
pytorch_struct model, Unsupervised CFG task
https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/Unsupervised_CFG.ipynb
"""
import os
import pytest
import torchtext
import numpy as np
import torch, random
from torch_struct import SentCFG
from .networks.NeuralCFG import NeuralCFG

from collections import Counter
from torchtext.vocab import vocab_factory
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

# setup environment variable
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, ".data", "udpos")
torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

_text_transform = lambda vocab, x: [vocab['<bos>']] + [vocab[token] for token in x] + [vocab['<eos>']]

class Model(BenchmarkModel):
  task = OTHER.OTHER_TASKS

  def _collate_batch(batch):
    label_list, text_list = [], []
    for line in batch:
      for sentence in line:
        processed_text = torch.tensor(_text_transform(self.train_vocab, sentence))
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=3.0)

  def __init__(self, device=None, jit=False, train_bs=256):
    super().__init__()
    self.device = device
    self.jit = jit

    # Download and the load default data.
    train = torchtext.datasets.UDPOS(root=DATA_DIR, split=('train'))
    # Build vocab 
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in train:
      for sentence in line:
        for word in sentence:
          counter.update(tokenizer(word))
    self.train_vocab = vocab_factory.vocab(counter, min_freq=3, specials=('<bos>', '<eos>'))

    # Build model
    H = 256
    T = 30
    NT = 30
    self.model = NeuralCFG(len(self.train_vocab), T, NT, H)
    if jit:
        self.model = torch.jit.script(self.model)
    self.model.to(device=device)
    self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=[0.75, 0.999])

    self.train_loader = DataLoader(train, batch_size=train_bs, collate_fn=self._collate_batch)

  def get_module(self):
    for ex in self.train_loader:
      words, _ = ex.word
      words = words.long()
      return self.model, (words.to(device=self.device).transpose(0, 1),)

  def train(self, niter=1):
    for _, words in zip(range(niter), self.train_loader):
      losses = []
      self.opt.zero_grad()
      params = self.model(words)
      dist = SentCFG(params, lengths=self.lengths)
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
