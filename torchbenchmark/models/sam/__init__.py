# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from ...util.model import BenchmarkModel
from sam import Sam
from image_encoder import ImageEncoderViT
from mask_decoder import MaskDecoder
from prompt_encoder import PromptEncoder
from transformer import TwoWayTransformer
from predictor import SamPredictor


from torchbenchmark.tasks import ComputerVision
import torch

    
class Model(BenchmarkModel):
    task = NLP.SEGMENTATION
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, jit=False, batch_size=1, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        
        # Checkpoint options are here https://github.com/facebookresearch/segment-anything#model-checkpoints
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(device=device)

        self.example_inputs = torch.randn(batch_size, 3, 224, 224).to(device=device)
        
        # We will just run an inference over a random tensor
        # predictor = SamPredictor(sam)
        
    def get_module(self):
        return self.model, self.example_inputs
    
    def train(self):
        error_msg = """
            As of May 17, 2023
            Some base VIT checkpoints are available for SAM but getting the dataset
            requires a research license. It's easy to make up a training loop on random
            data and if that's interesting please let @msaroufim know
            https://github.com/facebookresearch/segment-anything#dataset
        """
        return NotImplementedError(error_msg)

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            out=self.model(*self.example_inputs)
        return (out,)
