# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from ...util.model import BenchmarkModel
from .build_sam import sam_model_registry
from .predictor import SamPredictor
from PIL import Image
import numpy as np
import cv2
from torchbenchmark.tasks import COMPUTER_VISION
import torch
import os

    
class Model(BenchmarkModel):
    task = COMPUTER_VISION.SEGMENTATION
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, jit=False, batch_size=1, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        
        # Checkpoint options are here https://github.com/facebookresearch/segment-anything#model-checkpoints
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
        sam_checkpoint = os.path.join(data_folder, 'sam_vit_h_4b8939.pth')
        model_type = "vit_h"

        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(device=device)   
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')

        image_path = os.path.join(data_folder, 'truck.jpg')
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)     
        self.sample_image  = torch.randn((3, 256, 256)).to(device)

   
    def get_module(self):
        example_input = [
            {
                'image': self.sample_image,
                'original_size': (256, 256),
            }
        ]

        multimask_output = False

        return self.model, (example_input, multimask_output)
            
    def train(self):
        """
        This really means finetune since it's too expensive to train this from scratch
        """
        optimizer = torch.optim.Adam(self.model.mask_decoder.parameters()) 
        loss_fn = torch.nn.MSELoss()
        images = []

        for input_image, input_mask in images:

            with torch.no_grad():
                image_embedding = self.model.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder()
            
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Get input_size and original_image_size here
            upscaled_masks = self.model.postprocess_masks(low_res_masks, input_size, original_image_size).to(self.device)

            from torch.nn.functional import threshold, normalize

            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(self.device)

            loss = loss_fn(binary_mask, input_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def eval(self):
        predictor = SamPredictor(self.model)

        predictor.set_image(self.image)

        input_point = np.array([[500, 375]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True)
        return (masks,)