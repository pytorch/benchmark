# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.



from ...util.model import BenchmarkModel
import torch
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model(BenchmarkModel):
    def __init__(self):
        self.model_args = ModelArgs()
        self.model = Transformer(self.model_args)
        self.example_inputs = torch.tensor([[1, 1], [1,1]], dtype=torch.int)

        
    def get_module(self):
        return self.transformer, self.example_inputs
    
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
            out=self.model(self.example_inputs, 1)
        return (out,)





