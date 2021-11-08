"""
Maskrcnn model from torchvision
"""

import torch
import os
import yaml
import random
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

# Model specific imports
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

MASTER_SEED = 1337
torch.manual_seed(MASTER_SEED)
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Input tensors:
# Tensor [N, C, H, W]
# N: Number of pictures
# C: Channels of color
# H: Picture Height
# W: Picture Weight
# Targets:
# Boxes: FloatTensor[N, 4]
# Labels: Int64Tensor[N]
# Masks: UInt8Tensor[N, H, W]

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    def __init__(self, device=None, jit=False, train_bs=1, eval_bs=1):
        self.device = device
        self.jit = jit
        self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        self.backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                                aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=7,
                                                         sampling_ratio=2)
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                             output_size=14,
                                                             sampling_ratio=2)
        self.model = MaskRCNN(backbone, num_classes=2,
                              rpn_anchor_generator=anchor_generator,
                              box_roi_pool=roi_pooler,
                              mask_roi_poll=mask_roi_pooler).to(self.device)
        # Generate inputs
        with open(coco2017_config, "r") as cc2017:
            self.cfg = yaml.safe_load(cc2017)
        self.example_inputs = self._gen_inputs(train_bs)
        self.example_targets = self._gen_targets(train_bs)
        self.infer_example_inputs = self._gen_input(eval_bs)

    def _gen_inputs(self, batch_size):
        return torch.rand(batch_size, self.cfg.C, self.cfg.W, self.cfg.H)

    def _gen_targets(self, batch_size):
        targets = {}
        targets["boxes"] = torch.rand(batch_size, 4)
        targets["labels"] = torch.randint(batch_size)
        targets["scores"] = torch.rand(batch_size)
        targets["masks"] = torch.randint(batch_size, 1, self.cfg.H, self.cfg.W)
        return targets
        
    def train(self, niter=1):
        self.model.train()
        self.optimizer.zero_grad()
        for iter in range(niter):
            for images, targets in zip(self.example_inputs, self.example_targets):
                # Images: input images represented as tensors
                # Targets: the ground truth of boxes
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                self.optimizer.backward(losses)
                self.optimizer.step()  # This will sync
                # post-processing
                self.optimizer.zero_grad()
                self.scheduler.step()

    def eval(self, niter=1):
        for iter in range(niter):
            for image in self.infer_example_inputs:
                model(image)

if __name__ == "__main__":
    pass
