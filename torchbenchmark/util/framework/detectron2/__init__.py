from torchbenchmark.util.model import BenchmarkModel
import itertools
import os
from pathlib import Path

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.engine import default_argument_parser
from detectron2.solver import build_optimizer
from detectron2.config import LazyConfig, get_cfg, instantiate
from detectron2 import model_zoo

import torch

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR.parent.parent.parent, "data", ".data", "coco2017-minimal")
assert os.path.exists(DATA_DIR), "Couldn't find coco2017 minimal data dir, please run install.py again."
if not 'DETECTRON2_DATASETS' in os.environ:
    os.environ['DETECTRON2_DATASETS'] = DATA_DIR


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.SOLVER.BASE_LR = 0.001  # Avoid NaNs. Not useful in this script anyway.
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg

def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    return model

def prefetch(dataloader, device):
    r = []
    for batch in dataloader:
        r.append(batch.to(device))

class Detectron2Model(BenchmarkModel):
    def __init__(self, variant, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        parser = default_argument_parser()
        args = parser.parse_args(["--config-file", variant])
        data_cfg = model_zoo.get_config("common/data/coco.py").dataloader
        cfg = setup(args)
        self.model = build_model(cfg).to(self.device)
        if self.test == "train":
            self.model.train()
            self.optimizer = build_optimizer(cfg, self.model)
            # setup train dataset
            data_cfg.train.dataset.names = "coco_2017_val_100"
            data_cfg.train.total_batch_size = self.batch_size
            train_loader = instantiate(data_cfg.train)
            self.example_inputs = prefetch(itertools.islice(train_loader, 100), self.device)
        elif self.test == "eval":
            self.model.eval()
            # setup eval dataset
            data_cfg.test.dataset.names = "coco_2017_val_100"
            data_cfg.test.batch_size = self.batch_size
            test_loader = instantiate(data_cfg.test)
            self.example_inputs = prefetch(itertools.islice(test_loader, 100), self.device)
        cfg.defrost()

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        for _idx, d in enumerate(self.example_inputs):
            loss_dict = self.model(d)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

    def eval(self):
        with torch.no_grad():
            for _idx, d in enumerate(self.example_inputs):
                self.model(d)