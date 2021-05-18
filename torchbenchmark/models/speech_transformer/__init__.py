#!/usr/bin/env python
import torch

from .decoder import Decoder
from .encoder import Encoder
from .transformer import Transformer
from .solver import Solver
from .optimizer import TransformerOptimizer
from .config import SpeechTransformerConfig, SpeechTransformerEvalConfig

from torchbenchmark.tasks import SPEECH

class Model(BenchmarkModel, device = "cuda", jit = False):
    task = SPEECH.RECOGNITION
    def __init__(self, device=None, jit=False):
        if jit:
            raise NotImplementedError()
        if not device == "cuda":
            raise NotImplementedError()
        self.jit = jit
        self.device = device
        self.traincfg = SpeechTransformerTrainConfig()
        self.evalcfg = SpeechTransformerEvalConfig()
        self.traincfg.model.cuda()
        self.evalcfg.model.cuda()

    def get_module(self):
        if not self.device == "cuda":
            raise NotImplementedError()
        if self.jit:
            raise NotImplementedError()
        return self.traincfg.model, self.traincfg.tr_loader

    def train(self, niter=1):
        if not self.device == "cuda":
            raise NotImplementedError()
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            self.traincfg.train()

    def eval(self, niter=1):
        if not self.device == "cuda":
            raise NotImplementedError()
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            self.evalcfg.eval()

if __name__ == '__main__':
    for device in ['cuda']:
        for jit in [False]:
            m = Model(device=device, jit=jit)
            model, example_inputs = m.get_module()
            model(*example_inputs)
            m.train()
            m.eval()
