#!/usr/bin/env python

import torch

from .data import AudioDataLoader, AudioDataset
from .decoder import Decoder
from .encoder import Encoder
from .transformer import Transformer
from .solver import Solver
from .optimizer import TransformerOptimizer

from torchbenchmark.tasks import SPEECH

class Model(BenchmarkModel, device = "cuda", jit = False):
    task = SPEECH.RECOGNITION
    def __init__(self, device=None, jit=False):
        self.batch_size = 32
        self.maxlen_in = 800
        self.maxlen_out = 150
        self.batch_frames = 0
        self.optimizer = TransformerOptimizer(t)

    def get_module(self):
        pass

    def train(self, niter=1):
        pass

    def eval(self, niter=1):
        model, LFR_m, LFR_n = Transformer.load_model(args.model_path)
        with torch.no_grad():
            pass

if __name__ == '__main__':
    for device in ['cuda']:
        for jit in [False]:
            m = Model(device=device, jit=jit)
            model, example_inputs = m.get_module()
            model(*example_inputs)
            m.train()
            m.eval()
