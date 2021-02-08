import os
import subprocess
import sys
import torchtext
import pickle
from torch.utils.data import DataLoader
import torch

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def setup_install():
    subprocess.check_call([sys.executable, 'setup.py', 'develop'])

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label - 1

def preprocess_data():
    batch_size = 32
    train_dataset, _ = torchtext.experimental.datasets.AG_NEWS(ngrams=1)
    vocab_size = len(train_dataset.vocab)
    loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=generate_batch)
    text, offsets, cls = next(iter(loader))
    with open("example_batch.pkl", "wb") as f:
        pickle.dump(
            (batch_size, vocab_size, text, offsets, cls),
            f, pickle.DEFAULT_PROTOCOL)

if __name__ == '__main__':
    pip_install_requirements()
    setup_install()
    # FIXME: AG_NEWS isn't working for some reason
    # preprocess_data()
