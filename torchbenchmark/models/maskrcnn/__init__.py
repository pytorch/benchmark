"""
Maskrcnn model from torchvision
"""

import torch
import os
import yaml
import random
import numpy as np
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

    def __init__(self, device=None, jit=False, train_bs=1, eval_bs=1, config="coco2017_config.yaml"):
        self.device = device
        self.jit = jit
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
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
                              mask_roi_pool=mask_roi_pooler).to(self.device)
        # Generate inputs
        current_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_dir, config), "r") as cc2017:
            self.cfg = yaml.safe_load(cc2017)
        self.example_inputs = self._gen_inputs(train_bs)
        self.example_targets = self._gen_targets(train_bs)
        self.infer_example_inputs = self._gen_inputs(eval_bs)

    def _gen_inputs(self, batch_size):
        inputs = torch.rand(batch_size, self.cfg['C'], self.cfg['H'], self.cfg['W'], device=self.device)
        return inputs

    # Generate three target boxes
    def _gen_boxes(self):
        return torch.tensor([
            # Big box: 200x100
            [0, 0, 200, 100],
            # Medium box: 100x50
            [230, 50, 330, 100],
            # Small box: 50x25
            [220, 120, 270, 145] ], dtype=torch.float64, device=self.device)

    def _gen_labels(self, box_cnt):
        return torch.tensor([1]*box_cnt, dtype=torch.int64, device=self.device)

    def _gen_scores(self, box_cnt):
        return torch.rand(box_cnt, dtype=torch.float64, device=self.device)

    def _gen_masks(self, box_cnt):
        return torch.rand(box_cnt, 1, self.cfg['H'], self.cfg['W'], dtype=torch.float64, device=self.device)

    def _gen_targets(self, batch_size, box_cnt=3):
        targets = []
        for _ in range(batch_size):
            target = {}
            target["boxes"] = self._gen_boxes()
            target["labels"] = self._gen_labels(box_cnt)
            target["scores"] = self._gen_scores(box_cnt)
            target["masks"] = self._gen_masks(box_cnt)
            targets.append(target)
        return targets
        
    def train(self, niter=1):
        if self.jit:
            return NotImplementedError("JIT is not supported by this model")
        self.model.train()
        for iter in range(niter):
            self.model(self.example_inputs, self.example_targets)

    def eval(self, niter=1):
        if self.jit:
            return NotImplementedError("JIT is not supported by this model")
        self.model.eval()
        with torch.no_grad():
            for iter in range(niter):
                out = self.model(self.infer_example_inputs)
                print(out)

if __name__ == "__main__":
    pass
