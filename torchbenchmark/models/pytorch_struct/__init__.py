"""
pytorch_struct model, Unsupervised CFG task
https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/Unsupervised_CFG.ipynb
"""
import pytest
import numpy as np
import torch, random


from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch_struct import SentCFG
from .networks.NeuralCFG import NeuralCFG

from torchtext.datasets import UDPOS
from torchtext.vocab import build_vocab_from_iterator

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def yield_tokens(data_iter):
    for text in data_iter:
        yield from text


def collate_batch(batch, vocab, device, padding_value, filter_pred):
    label_list, text_list = [], []

    for example in batch:
        mask = [i for i, e in enumerate(example[0]) if filter_pred(e)]
        selected_text = [example[0][i] for i in mask]
        selected_labels = [example[1][i] for i in mask]
        processed_text = torch.tensor(vocab(selected_text), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(torch.tensor(vocab(selected_labels), dtype=torch.int64))
    label_list = pad_sequence(label_list, padding_value=padding_value, batch_first=True)
    text_list = pad_sequence(text_list, padding_value=padding_value, batch_first=True)
    return label_list.to(device), text_list.to(device)

class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    # Original train batch size: 200
    # Source: https://github.com/harvardnlp/pytorch-struct/blob/f4e374e894b94a9411fb3d2dfb44201a18e37b26/notebooks/Unsupervised_CFG.ipynb
    def __init__(self, device=None, jit=False, train_bs=200):
        super().__init__()
        self.device = device
        self.jit = jit

        train_iter = UDPOS(split='train')
        vocab = build_vocab_from_iterator(
            yield_tokens(train_iter), specials=["<unk>", "<bos>", "<eos>"], min_freq=3
        )
        vocab.set_default_index(vocab["<unk>"])

        self.train_data = DataLoader(
          UDPOS(split='train'),
          batch_size=train_bs,
          collate_fn=lambda batch: collate_batch(
              batch, vocab, device, padding_value=vocab(["<pad>"])[0], filter_pred=lambda word: 5 < len(word) < 30
          )
        )

        # Build model
        H = 256
        T = 30
        NT = 30
        self.model = NeuralCFG(len(vocab), T, NT, H)
        self.model.to(device=device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.75, 0.999))

    def get_module(self):
        for words, _ in self.train_data:
            return self.model, (words,)

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


@pytest.mark.parametrize('jit', [True, False], ids=['jit', 'no-jit'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
class TestBench():
    def test_train(self, benchmark, device, jit):
        m = Model(device=device, jit=jit)
        benchmark(cuda_sync, m.train, device == 'cuda')


if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        for jit in [True, False]:
            print("Testing device {}, JIT {}".format(device, jit))
            m = Model(device=device, jit=jit)
            model, example_inputs = m.get_module()
            model(*example_inputs)
            m.train()
