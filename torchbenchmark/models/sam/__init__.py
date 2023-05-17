# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from ...util.model import BenchmarkModel
from .sam import Sam
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

from torchbenchmark.tasks import ComputerVision
import torch

    
class Model(BenchmarkModel):
    task = NLP.SEGMENTATION
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.model_args = ModelArgs(vocab_size=32000,device=self.device)
        torch.set_default_device(self.device)
        self.model = Transformer(self.model_args).to(self.device)
        self.seq_len = 32
        self.example_inputs = (torch.ones([self.batch_size, self.seq_len], dtype=torch.int).to(self.device), 1)

        
    def get_module(self):
        return self.model, self.example_inputs
    
    def train(self):
        error_msg = """
            As of May 17, 2023
            Some base VIT checkpoints are available for SAM but getting the dataset
            requires a research license. It's easy to make up a training loop on random
            data and if that's interesting please let @msaroufim know
            https://github.com/facebookresearch/segment-anything
        """
        return NotImplementedError(error_msg)

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            out=self.model(*self.example_inputs)
        return (out,)
