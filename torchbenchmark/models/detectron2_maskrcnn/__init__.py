import torch
import os
import itertools
import random
import itertools
from pathlib import Path
from typing import Tuple

# TorchBench imports
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR.parent.parent, "data", ".data", "coco2017-minimal")
assert os.path.exists(DATA_DIR), "Couldn't find coco2017 minimal data dir, please run install.py again."
if not 'DETECTRON2_DATASETS' in os.environ:
    os.environ['DETECTRON2_DATASETS'] = DATA_DIR

from detectron2.config import instantiate
from detectron2 import model_zoo
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 2

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        model_cfg = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
        data_cfg = model_zoo.get_config("common/data/coco.py").dataloader

        if test == "train":
            # use a mini dataset
            data_cfg.train.dataset.names = "coco_2017_val_100"
            data_cfg.train.total_batch_size = self.batch_size
            self.model = instantiate(model_cfg).to(self.device)
            train_loader = instantiate(data_cfg.train)
            self.example_inputs = itertools.cycle(itertools.islice(train_loader, 100))
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.)
        elif test == "eval":
            data_cfg.test.dataset.names = "coco_2017_val_100"
            data_cfg.test.batch_size = self.batch_size
            self.model = instantiate(model_cfg).to(self.device)
            self.model.eval()
            test_loader = instantiate(data_cfg.test)
            self.example_inputs = itertools.cycle(itertools.islice(test_loader, 100))

    def get_module(self):
        for data in self.example_inputs:
            return self.model, (data, )

    def train(self, niter=1):
        self.model.train()
        with EventStorage():
            for idx, data in zip(range(niter), self.example_inputs):
                losses = self.model(data)
                loss = sum(losses.values())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def eval(self, niter=2) -> Tuple[torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            for idx, data in zip(range(niter), self.example_inputs):
                out = self.model(data)
        # retrieve output tensors
        outputs = []
        for item in out:
            fields = list(map(lambda x: list(x.get_fields().values()), item.values()))
            for boxes in fields:
                tensor_box = list(filter(lambda x: isinstance(x, torch.Tensor), boxes))
                outputs.extend(tensor_box)
        return tuple(outputs)
