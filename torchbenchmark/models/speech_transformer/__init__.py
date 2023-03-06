#!/usr/bin/env python
#
# The SpeechTransformer model copied from https://github.com/kaituoxu/Speech-Transformer, commit e684777.
# The model only supports CUDA and eager mode.
# The input data files in the input_data/ directory are generated with a minimized aishell data
# containing the following files in the original dataset:
# S0002.tar.gz, S0757.tar.gz, S0915.tar.gz
#
import os
import itertools
import torch

from .config import SpeechTransformerTrainConfig, SpeechTransformerEvalConfig
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH
from typing import Tuple

NUM_TRAIN_BATCH = 1
NUM_EVAL_BATCH = 1

class Model(BenchmarkModel):
    task = SPEECH.RECOGNITION
    # Original batch size: 32
    # Source: https://github.com/kaituoxu/Speech-Transformer/blob/e6847772d6a786336e117a03c48c62ecbf3016f6/src/bin/train.py#L68
    # This model does not support adjusting eval bs
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.traincfg = SpeechTransformerTrainConfig(prefetch=True, train_bs=self.batch_size, num_train_batch=NUM_TRAIN_BATCH, device=self.device)
        if test == "train":
            self.traincfg.model.to(self.device)
            self.traincfg.model.train()
        elif test == "eval":
            self.evalcfg = SpeechTransformerEvalConfig(self.traincfg, num_eval_batch=NUM_EVAL_BATCH, device=self.device)
            self.evalcfg.model.to(self.device)
            self.evalcfg.model.eval()

    def get_module(self):
        for data in self.traincfg.tr_loader:
            padded_input, input_lengths, padded_target = data
            if self.test == "train":
                return self.traincfg.model, (padded_input.to(self.device), input_lengths.to(self.device), padded_target.to(self.device))
            elif self.test == "eval":
                return self.evalcfg.model, (padded_input.to(self.device), input_lengths.to(self.device), padded_target.to(self.device))

    def set_module(self, new_model):
        if self.test == "train":
            self.traincfg.model = new_model
        elif self.test == "eval":
            self.evalcfg.model = new_model

    def train(self):
        self.traincfg.train(epoch=1)

    def eval(self) -> Tuple[torch.Tensor]:
        out = self.evalcfg.eval()
        # only the first element of model output is a tensor
        out = tuple(itertools.chain(*list(map(lambda x: x.values(), out))))
        return (out[0], )

    def get_optimizer(self):
        return self.traincfg.get_optimizer()

    def set_optimizer(self, optimizer) -> None:
        return self.traincfg.set_optimizer(optimizer)

    def set_raw_optimizer(self, optimizer) -> None:
        return self.traincfg.set_raw_optimizer(optimizer)
