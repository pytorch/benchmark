#!/usr/bin/env python
#
# The SpeechTransformer model copied from https://github.com/kaituoxu/Speech-Transformer, commit e684777.
# The model only supports CUDA and eager mode.
# The input data files in the input_data/ directory are generated with a minimized aishell data
# containing the following files in the original dataset:
# S0002.tar.gz, S0757.tar.gz, S0915.tar.gz
# 
import os
import sys
import torch

# Add the current path to sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
sys.path.append(os.path.join(dir_path, "transformer"))
sys.path.append(os.path.join(dir_path, "utils"))

from ...util.model import BenchmarkModel
from config import SpeechTransformerTrainConfig, SpeechTransformerEvalConfig
from torchbenchmark.tasks import SPEECH

class Model(BenchmarkModel):
    task = SPEECH.RECOGNITION
    def __init__(self, device=None, jit=False):
        self.jit = jit
        self.device = device
        if jit:
            return
        if not device == "cuda":
            return
        self.traincfg = SpeechTransformerTrainConfig()
        self.evalcfg = SpeechTransformerEvalConfig(self.traincfg)
        self.traincfg.model.cuda()
        self.evalcfg.model.cuda()

    def get_module(self):
        if not self.device == "cuda":
            raise NotImplementedError("CPU is not supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        for data in self.traincfg.tr_loader:
            padded_input, input_lengths, padded_target = data
            return self.traincfg.model, (padded_input.cuda(), input_lengths.cuda(), padded_target.cuda())

    def train(self, niter=1):
        if not self.device == "cuda":
            raise NotImplementedError("CPU is not supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        for i in range(niter):
            self.traincfg.train(epoch = i)

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
