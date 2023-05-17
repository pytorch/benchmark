# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from ...util.model import BenchmarkModel
from .build_sam import sam_model_registry
from .predictor import SamPredictor
from PIL import Image
import numpy as np

from torchbenchmark.tasks import COMPUTER_VISION
import torch


    
class Model(BenchmarkModel):
    task = COMPUTER_VISION.SEGMENTATION
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, jit=False, batch_size=1, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        
        # Checkpoint options are here https://github.com/facebookresearch/segment-anything#model-checkpoints
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        # TODO Before merge: Add the real checkpoint when done testing
        self.model = sam_model_registry[model_type](checkpoint=None)
        self.model.to(device=device)

        # TODO Before merge: Make the batch size configurable
        # We don't actually pass in a tensor but pass in an image
        # self.example_inputs = [{0 : torch.randn(3, 224, 224).to(device=device)}], 
        
   
    def get_module(self):
        return self.model # self.example_inputs
    
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
        predictor = SamPredictor(self.model)
        random_image_path = generate_random_image(128, 128, 3)
        masks, _, _ = predictor.predict(random_image_path)
        self.model.eval()
        with torch.no_grad():
            out=self.model(self.example_inputs, multimask_output=True)
        return (masks,)


# TODO: I'm open to wgetting a real image but this seems useful for now
# Generate a random image with specified width, height, and color channels
def generate_random_image(width, height, channels):
    # Create a random numpy array representing the image pixels
    image_data = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    
    # Create a Pillow image object from the numpy array
    image = Image.fromarray(image_data)
    
    # Save the image to a file
    image_path = "random_image.jpg"
    image.save(image_path)
    
    return image_path