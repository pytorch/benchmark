# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.



from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import ModelArgs, Transformer
import torch

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        self.model_args = ModelArgs(vocab_size=32000,device=self.device)
        torch.set_default_device(self.device)
        self.model = Transformer(self.model_args).to(self.device)
        self.seq_len = 32
        self.example_inputs = (torch.ones([self.batch_size, self.seq_len], dtype=torch.int).to(self.device), 1)

        
    def get_module(self):
        return self.model, self.example_inputs
    
    def train(self):
        error_msg = """
            As of March 6, 2023
            The weights for this model are not publicly available and require a valid research reason to use
            The publicly available github repo is inference only
            https://github.com/facebookresearch/llama
        """
        return NotImplementedError(error_msg)

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            out=self.model(*self.example_inputs)
        return (out,)
