#!/usr/bin/env python
#
# The SpeechTransformer model copied from https://github.com/kaituoxu/Speech-Transformer, commit e684777.
# The model only supports CUDA and eager mode.
# The input data files in the input_data/ directory are generated with a minimized aishell data
# containing the following files in the original dataset:
# S0002.tar.gz, S0757.tar.gz, S0915.tar.gz
#
import os
import torch

from .config import SpeechTransformerTrainConfig, SpeechTransformerEvalConfig
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH

NUM_TRAIN_BATCH = 1
NUM_EVAL_BATCH = 1

class Model(BenchmarkModel):
    task = SPEECH.RECOGNITION
    # Original batch size: 32
    # Source: https://github.com/kaituoxu/Speech-Transformer/blob/e6847772d6a786336e117a03c48c62ecbf3016f6/src/bin/train.py#L68
    # This model does not support adjusting eval bs
    def __init__(self, device=None, jit=False, train_bs=32):
        self.jit = jit
        self.device = device
        if jit:
            return
        if device == "cpu":
            return
        self.traincfg = SpeechTransformerTrainConfig(prefetch=True, train_bs=train_bs, num_train_batch=NUM_TRAIN_BATCH)
        self.evalcfg = SpeechTransformerEvalConfig(self.traincfg, num_eval_batch=NUM_EVAL_BATCH)
        self.traincfg.model.to(self.device)
        self.evalcfg.model.to(self.device)

    def get_module(self):
        if self.device == "cpu":
            raise NotImplementedError("CPU is not supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        for data in self.traincfg.tr_loader:
            padded_input, input_lengths, padded_target = data
            return self.traincfg.model, (padded_input.to(self.device), input_lengths.to(self.device), padded_target.to(self.device))

    def train(self, niter=1):
        if self.device == "cpu":
            raise NotImplementedError("CPU is not supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        for i in range(niter):
            self.traincfg.train(epoch = i)

    def eval(self, niter=1):
        if self.device == "cpu":
            raise NotImplementedError("CPU is not supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
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
