import torchtext
import torch

from torchbenchmark.util.torchtext_legacy.field import Field
from torchbenchmark.util.torchtext_legacy.datasets import UDPOS
from torch_struct import SentCFG
from torch_struct.networks import NeuralCFG
import torch_struct.data
import sys

import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', metavar='fn', default="", help="Dump outputs into file")
parser.add_argument('--script', default=False, help="Script the model")
args = parser.parse_args()

random.seed(1337)
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
train_iter = torch_struct.data.TokenBucket(train, batch_size=100, device="cuda:0")

H = 256
T = 30
NT = 30
model = NeuralCFG(len(WORD.vocab), T, NT, H)
if args.script:
    print("scripting...")
    model = torch.jit.script(model)
model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.75, 0.999])

def train():
    # model.train()
    losses = []
    for epoch in range(2):
        for i, ex in enumerate(train_iter):
            opt.zero_grad()
            words, lengths = ex.word
            N, batch = words.shape
            words = words.long()
            params = model(words.cuda().transpose(0, 1))
            dist = SentCFG(params, lengths=lengths)
            loss = dist.partition.mean()
            (-loss).backward()
            losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

            if i > 100:
                break
            if i % 100 == 1:
                print(-torch.tensor(losses).mean(), words.shape)
                losses = []
    if args.debug:
        print(f"saving to {args.debug}...")
        torch.save(params[0], args.debug)

train()
