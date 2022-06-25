import torch
import sys
import os
from torchbenchmark import REPO_PATH
from typing import Tuple
import torch.nn as nn

# Import FAMBench model path
class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

WLM_PATH = os.path.join(REPO_PATH, "submodules", "examples", "word_language_model")
DATA_PATH = os.path.join(WLM_PATH, "data", "wikitext-2")

with add_path(WLM_PATH):
    import data
    import model as wlm_model

from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
from .config import parse_args

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(args, source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    # default batch size from:
    # https://github.com/pytorch/examples/blob/main/word_language_model/main.py#L30
    DEFAULT_NUM_BATCHES = 1
    DEFAULT_TRAIN_BSIZE = 20
    DEFAULT_EVAL_BSIZE = 20

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        model_args = ["--data", DATA_PATH, "--batch_size", str(self.batch_size), "--model", "Transformer", "--epochs", "1"]
        args = parse_args(model_args)
        corpus = data.Corpus(args.data)
        self.ntokens = len(corpus.dictionary)
        self.model = wlm_model.TransformerModel(self.ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(self.device)
        self.args = args
        if self.test == "train":
            self.lr = args.lr
            self.criterion = nn.NLLLoss()
            self.example_inputs = batchify(corpus.train, self.batch_size).to(self.device)
            self.model.train()
        else:
            self.example_inputs = batchify(corpus.valid, self.batch_size).to(self.device)
            self.model.eval()

    def get_module(self):
        for batch, i in enumerate(range(0, self.example_inputs.size(0)-1, self.args.bptt)):
            data, _targets = get_batch(self.args, self.example_inputs, i)
            return self.model, (data, )

    def train(self):
        for batch, i in enumerate(range(0, self.example_inputs.size(0)-1, self.args.bptt)):
            data, targets = get_batch(self.args, self.example_inputs, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.model.zero_grad()
            output = self.model(data)
            output = output.view(-1, self.ntokens)
            loss = self.criterion(output, targets)
            loss.backward()

             # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            for p in self.model.parameters():
                p.data.add_(p.grad, alpha=-self.lr)
            total_loss += loss.item()

    def eval(self) -> Tuple[torch.Tensor]:
        ntokens = self.ntokens
        with torch.no_grad():
            for i in range(0, self.example_inputs.size(0) - 1, self.args.bptt):
                data, _targets = get_batch(self.args, self.example_inputs, i)
                output = self.model(data)
                output = output.view(-1, ntokens)
        return output