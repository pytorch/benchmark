# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.



from ...util.model import BenchmarkModel
from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model(BenchmarkModel):
    def __init__(self, temperature: float = 0.8, top_p: float = 0.95):
        self.model_args = ModelArgs()
        self.generator = Transformer(self.model_args)
        self.temperature = temperature
        self.top_p = top_p
    
    def inference(self, prompts : str):
        prompts = ["The capital of Germany is the city of", "Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]
        results = self.generator.generate(prompts, max_gen_len=256, temperature=self.temperature, top_p=self.top_p)

        for result in results:
            print(result)
            print("\n==================================\n")





