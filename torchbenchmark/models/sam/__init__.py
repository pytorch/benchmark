# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os

import cv2
import numpy as np
import torch
from torchbenchmark import DATA_PATH
from torchbenchmark.tasks import COMPUTER_VISION

from ...util.model import BenchmarkModel
from .build_sam import sam_model_registry
from .predictor import SamPredictor


class Model(BenchmarkModel):
    task = COMPUTER_VISION.SEGMENTATION
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, batch_size=1, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        # Checkpoint options are here https://github.com/facebookresearch/segment-anything#model-checkpoints
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
        sam_checkpoint = os.path.join(data_folder, "sam_vit_h_4b8939.pth")
        if not os.path.exists(sam_checkpoint):
            from torchbenchmark.util.framework.fb.installer import install_model_weights

            sam_checkpoint = install_model_weights(self.name)
        model_type = "vit_h"

        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(device=device)
        image_path = os.path.join(DATA_PATH, "sam_inputs", "truck.jpg")
        if not os.path.exists(image_path):
            from torchbenchmark.util.framework.fb.installer import install_data

            image_path = os.path.join(install_data("truck"), "truck.jpg")
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.sample_image = torch.randn((3, 256, 256)).to(device)

    def get_module(self):
        example_input = [
            {
                "image": self.sample_image,
                "original_size": (256, 256),
            }
        ]

        multimask_output = False
        return self.model, (example_input, multimask_output)

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
        # To test for bfloat16 uncomment the below line
        # predictor = SamPredictor(self.model.to(dtype=torch.bfloat16))

        predictor = SamPredictor(self.model)

        predictor.set_image(self.image)

        input_point = np.array([[500, 375]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=True
        )
        return (masks,)
