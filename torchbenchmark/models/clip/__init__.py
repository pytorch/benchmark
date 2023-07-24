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
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
from PIL import Image
import math



class Model(BenchmarkModel):
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, jit=False, batch_size=1, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        
        self.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
        self.image_name = "Pizza-3007395.jpg"
        self.image = Image.open(os.path.join(self.data_folder, self.image_name))
        self.text = ["pizza", "dog", "sun"]
        self.img_transform = CLIPImageTransform(is_train=False)
        self.text_transform = CLIPTextTransform()

        self.image_tensor = self.img_transform(self.image)
        self.text_tensor = self.text_transform(self.text)
        self.model = clip_vit_b32()
        self.model.to(self.device)

   
    def get_module(self):
        return self.model, (self.image_tensor, self.text_tensor)

            
    def train(self):
        self.model.train()

        # Create optimizer
        loss_fn = ContrastiveLossWithTemperature()
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(loss_fn.parameters()),
            lr=5.0e-4,
            weight_decay=1.0e-4,
            eps=1.0e-6,
            foreach=True,
        )

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        image_embedding, text_embedding = self.model(self.image_tensor, self.text_tensor)
        score = image_embedding @ text_embedding.t()

        # Compute loss
        loss = loss_fn(score)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        return loss.item()
    
    def eval(self):
        self.model.eval()
    
        with torch.no_grad():
            image_embedding, text_embedding = self.model(self.image_tensor, self.text_tensor)
            score = image_embedding @ text_embedding.t() 
        
        return self.text[torch.argmax(score)]