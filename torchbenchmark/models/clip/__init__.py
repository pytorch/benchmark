# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from PIL import Image
import numpy as np
import cv2
import torch
import os
from ...util.model import BenchmarkModel
from torchmultimodal.transforms.clip_transform import CLIPTextTransform, CLIPImageTransform
from torchmultimodal.models.clip.model import clip_vit_b32
from PIL import Image


    
class Model(BenchmarkModel):
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, jit=False, batch_size=1, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        
        self.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
  

   
    def get_module(self):
       return NotImplementedError("Will get to this")
            
    def train(self):
        error_msg = """
            train() is not implemented for CLIP but could be
            let us know if you're interested
        """
        return NotImplementedError(error_msg)

    def eval(self):
        image = Image.open(os.path.join(self.data_folder, "pizza.jpg"))
        text = ["pizza", "dog", "sun"]
        img_transform = CLIPImageTransform(is_train=False)
        text_transform = CLIPTextTransform()

        image_tensor = img_transform(image)
        text_tensor = text_transform(text)
        model = clip_vit_b32()
        model.eval()
        with torch.no_grad():
            image_embedding, text_embedding = model(image_tensor, text_tensor)
            score = image_embedding @ text_embedding.t() 
        
        text[torch.argmax(score)]