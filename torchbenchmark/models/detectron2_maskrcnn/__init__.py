import itertools
import os
from pathlib import Path
from typing import Tuple

import torch

from detectron2.checkpoint import DetectionCheckpointer
from torchbenchmark.tasks import COMPUTER_VISION

# TorchBench imports
from torchbenchmark.util.model import BenchmarkModel

MODEL_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR.parent.parent, "data", ".data", "coco2017-minimal")
if not os.path.exists(DATA_DIR):
    try:
        from torchbenchmark.util.framework.fb.installer import install_data

        DATA_DIR = install_data("coco2017-minimal")
    except Exception:
        pass
assert os.path.exists(
    DATA_DIR
), "Couldn't find coco2017 minimal data dir, please run install.py again."
if not "DETECTRON2_DATASETS" in os.environ:
    os.environ["DETECTRON2_DATASETS"] = DATA_DIR

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.utils.events import EventStorage
from torch.utils._pytree import tree_map

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def prefetch(dataloader, device, precision="fp32"):
    r = []
    dtype = torch.float16 if precision == "fp16" else torch.float32
    for batch in dataloader:
        r.append(
            tree_map(
                lambda x: (
                    x.to(device, dtype=dtype) if isinstance(x, torch.Tensor) else x
                ),
                batch,
            )
        )
    return r


class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION
    model_file = os.path.join(MODEL_DIR, ".data", f"{MODEL_NAME}.pkl")
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    DISABLE_DETERMINISM = True

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        model_cfg = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
        data_cfg = model_zoo.get_config("common/data/coco.py").dataloader

        if test == "train":
            # use a mini dataset
            data_cfg.train.dataset.names = "coco_2017_val_100"
            data_cfg.train.total_batch_size = self.batch_size
            self.model = instantiate(model_cfg).to(self.device)
            train_loader = instantiate(data_cfg.train)
            self.example_inputs = prefetch(
                itertools.islice(train_loader, 100), self.device
            )
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.0)
        elif test == "eval":
            data_cfg.test.dataset.names = "coco_2017_val_100"
            data_cfg.test.batch_size = self.batch_size
            self.model = instantiate(model_cfg).to(self.device)
            # load model from checkpoint
            if not os.path.exists(self.model_file):
                try:
                    from torchbenchmark.util.framework.fb.installer import (
                        install_model_weights,
                    )

                    self.model_file = install_model_weights(self.name)
                except Exception:
                    pass
            DetectionCheckpointer(self.model).load(self.model_file)
            self.model.eval()
            test_loader = instantiate(data_cfg.test)
            self.example_inputs = prefetch(
                itertools.islice(test_loader, 100), self.device
            )

    def get_module(self):
        return self.model, (self.example_inputs[0],)

    def train(self):
        self.model.train()
        idx = 0
        with EventStorage():
            losses = self.model(self.example_inputs[idx])
            loss = sum(losses.values())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def eval(self) -> Tuple[torch.Tensor]:
        self.model.eval()
        idx = 0
        out = self.model(self.example_inputs[idx])
        # retrieve output tensors
        outputs = []
        for item in out:
            fields = list(map(lambda x: list(x.get_fields().values()), item.values()))
            for boxes in fields:
                tensor_box = list(filter(lambda x: isinstance(x, torch.Tensor), boxes))
                outputs.extend(tensor_box)
        return tuple(outputs)
