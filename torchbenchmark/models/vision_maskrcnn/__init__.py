"""
Maskrcnn model from torchvision
"""

import torch
import os
import itertools
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from pathlib import Path
from typing import Tuple

# Model specific imports
import torchvision
from .coco_utils import ConvertCocoPolysToMask
from torchvision.datasets.coco import CocoDetection

# silence some spam
from pycocotools import coco

coco.print = lambda *args: None

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR.parent.parent, "data", ".data", "coco2017-minimal")
assert os.path.exists(
    DATA_DIR
), "Couldn't find coco2017 minimal data dir, please run install.py again."
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
    # MaskRCNN doesn't actually take the inputs in batches; it takes a list
    # of tensors which individually are CHW
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    NUM_OF_BATCHES = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, batch_size=None, extra_args=[], model_kwargs={}):
        # reduce the eval batch size when running on CPU
        # see: https://github.com/pytorch/benchmark/issues/895
        if device == "cpu":
            self.DEFAULT_EVAL_BSIZE = 1
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
            **model_kwargs,
        ).to(self.device)
        # setup optimizer
        # optimizer parameters copied from
        # https://github.com/pytorch/vision/blob/30f4d108319b0cd28ae5662947e300aad98c32e9/references/detection/train.py#L77
        lr = 0.02
        momentum = 0.9
        weight_decay = 1e-4
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        transforms = ConvertCocoPolysToMask()
        dataset = CocoDetection(
            root=os.path.join(DATA_DIR, COCO_DATA[COCO_DATA_KEY][0]),
            annFile=os.path.join(DATA_DIR, COCO_DATA[COCO_DATA_KEY][1]),
            transforms=transforms,
        )
        sampler = torch.utils.data.SequentialSampler(dataset)
        self.data_loader = _prefetch(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=_collate_fn,
            ),
            self.device,
        )

    def get_module(self):
        self.model.eval()
        for (example_inputs, _example_targets) in self.data_loader:
            return self.model, (example_inputs,)

    def train(self):
        self.model.train()
        for _batch_id, (images, targets) in zip(
            range(self.NUM_OF_BATCHES), self.data_loader
        ):
            # images = list(image.to(self.device) for image in images)
            # targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

    def eval(self) -> Tuple[torch.Tensor]:
        self.model.eval()
        for _batch_id, (images, _targets) in zip(
            range(self.NUM_OF_BATCHES), self.data_loader
        ):
            out = self.model(images)
        out = list(map(lambda x: x.values(), out))
        return tuple(itertools.chain(*out))
