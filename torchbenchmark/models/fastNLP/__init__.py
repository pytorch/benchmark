import torch
import torchtext
from fastNLP.models import BertForSequenceClassification
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
import argparse
import random
import numpy as np
import pickle
import os

try:
    from torchtext.datasets import AG_NEWS
except ImportError:
    # In older version of torchtext AG_NEWS is inside experimental module
    from torchtext.experimental.datasets import AG_NEWS

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from pathlib import Path

class Model:
    domain = "natural language processing"
    task = "other nlp"
    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit

        embed_dim = 64
        epochs = 5
        num_labels = 4
        root = str(Path(__file__).parent)
        with open(f"{root}/example_batch.pkl", "rb") as f:
            batch_size, vocab_size, text, offsets, cls = pickle.load(f)
        self.text, self.offsets, self.cls = [t.to(self.device) for t in (text, offsets, cls)]

        bert_embed = torch.nn.EmbeddingBag(vocab_size, embed_dim)
        self.model = BertForSequenceClassification(bert_embed, num_labels=num_labels).to(self.device)

        if self.jit:
            self.model = torch.jit.script(self.model)

        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

    def get_module(self):
        return self.model, (self.text, self.offsets)

    def eval(self, niter=1):
        with torch.no_grad():
            for _ in range(niter):
                output = self.model(self.text, self.offsets)
                loss = self.criterion(output, self.cls)

    def train(self, niter=1):
        for _ in range(niter):
            self.optimizer.zero_grad()
            output = self.model(self.text, self.offsets)
            loss = self.criterion(output, self.cls)
            loss.backward()
            self.optimizer.step()

        # Adjust the learning rate
        # Should we benchmark this?  It's run once per epoch
        # self.scheduler.step()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = Model(device=device, jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
