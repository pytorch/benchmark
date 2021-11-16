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
from torch.utils.data import DataLoader
from .transforms import Compose
from .coco_utils import ConvertCocoPolysToMask
from torchvision.datasets.coco import CocoDetection
from pycocotools.coco import COCO

MASTER_SEED = 1337
torch.manual_seed(MASTER_SEED)
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, ".data", "coco2017-minimal")
COCO_DATA_KEY = "coco_2017_val_100"
COCO_DATA = {
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json")
}

def _collate_fn(batch):
    return tuple(zip(*batch))

def _prefetch(loader, device):
    items = []
    for images, targets in loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        items.append((images, targets))
    return items

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    def __init__(self, device=None, jit=False, train_bs=4, eval_bs=4):
        self.device = device
        self.jit = jit
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device)

        # setup optimizer
        # optimizer parameters copied from
        # https://github.com/pytorch/vision/blob/30f4d108319b0cd28ae5662947e300aad98c32e9/references/detection/train.py#L77
        lr = 0.02
        momentum = 0.9
        weight_decay = 1e-4
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.train_bs = train_bs
        self.eval_bs = eval_bs
        t = [ConvertCocoPolysToMask()]
        transforms = Compose(t)
        dataset = CocoDetection(root=os.path.join(DATA_DIR, COCO_DATA[COCO_DATA_KEY][0]),
                                annFile=os.path.join(DATA_DIR, COCO_DATA[COCO_DATA_KEY][1]),
                                transforms=transforms)
        sampler = torch.utils.data.SequentialSampler(dataset)

        self.eval_data_loader = _prefetch(torch.utils.data.DataLoader(dataset, batch_size=eval_bs,
                                                                      sampler=sampler,
                                                                      collate_fn=_collate_fn), self.device)
        self.train_data_loader = _prefetch(torch.utils.data.DataLoader(dataset, batch_size=train_bs,
                                                                       sampler=sampler,
                                                                       collate_fn=_collate_fn), self.device)


    def get_module(self):
        self.model.eval()
        for (example_inputs, _example_targets) in self.eval_data_loader:
            return self.model, (example_inputs, )

    def train(self, niter=1):
        if self.jit:
            return NotImplementedError("JIT is not supported by this model")
        if not self.device == "cuda":
            return NotImplementedError("CPU is not supported by this model")
        self.model.train()
        for _, (images, targets) in zip(range(niter), self.train_data_loader):
            # images = list(image.to(self.device) for image in images)
            # targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        if self.jit:
            return NotImplementedError("JIT is not supported by this model")
        if not self.device == "cuda":
            return NotImplementedError("CPU is not supported by this model")
        self.model.eval()
        with torch.no_grad():
            for _, (images, _targets) in zip(range(niter), self.eval_data_loader):
                # images = list(image.to(self.device) for image in images)
                self.model(images)

if __name__ == "__main__":
    pass
