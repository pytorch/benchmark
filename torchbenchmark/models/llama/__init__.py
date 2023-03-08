# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.



from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import ModelArgs, Transformer
import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model(BenchmarkModel):
    DEFAULT_EVAL_BSIZE = 32
    task = NLP.LANGUAGE_MODELING

    def __init__(self, test, device, jit=False, batch_size=DEFAULT_EVAL_BSIZE, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.model_args = ModelArgs(vocab_size=32) # TODO: Configuring arguments is breaking stuff: max_batch_size=batch_size, max_seq_len=1032 is breaking stuff
        self.model = Transformer(self.model_args)

        # TODO: Implement batching

        if device == "cuda":
            torch.set_default_device("cuda")
            self.model.to(torch.device("cuda"))
        self.example_inputs = [torch.tensor([[1, 1], [1,1]], dtype=torch.int)]

        
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
            for example_input in self.example_inputs:
                out=self.model(example_input, 1)
        return (out,)





