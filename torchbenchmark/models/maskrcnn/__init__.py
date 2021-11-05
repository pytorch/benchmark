"""
Maskrcnn model from torchvision
"""

import torch
import os
import yaml
import random
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from .prefetcher import Prefetcher
from .maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.modeling.detector import build_detection_model

MASTER_SEED = 1337
torch.manual_seed(MASTER_SEED)
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    def __init__(self, device=None, jit=False, train_bs=1, eval_bs=1, cfg="e2e_mask_rcnn_R_50_FPN_1x.yaml"):
        self.device = device
        self.jit = jit
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", cfg)
        with open(cfg_path, "r") as cf:
            self.cfg = yaml.safe_load(cf)
        self.model = build_detection_model(self.cfg).to(self.device)
        random_number_generator = random.Random(MASTER_SEED)
        is_fp16 = (cfg.DTYPE == "float16")
        if is_fp16:
            # convert model to fp16
            self.model.half()
        # skip cuda graph support for now
        self.optimizer = make_optimizer(cfg, self.model)
        self.scheduler = make_lr_scheduler(cfg, self.optimizer)
        # train data loader
        self.train_data_loader, self.iters_per_epoch = make_data_loader(cfg, is_train=True,
                                                        is_distributed=False,
                                                        seed=seed, shapes=shapes)
        self.per_iter_callback_fn = None
        self.train_prefetcher = Prefetcher(self.train_data_loader, self.device)

    def train(self, niter=1):
        self.model.train()
        self.optimizer.zero_grad()
        for iter in niter:
            for images, targets in self.train_prefetcher:
                # Images: input images represented as tensors
                # Targets: the ground truth of boxes
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                self.optimizer.backward(losses)
                # At this point we are waiting for kernels launched to finish
                # Take advantage of this by loading next input batch before calling step
                self.train_prefetcher.prefetch()
                self.optimizer.step()  # This will sync
                # post-processing
                self.optimizer.zero_grad()
                self.scheduler.step()

    def eval(self, niter=1):
        return NotImplemented("Inference is not implemented for maskrcnn")

if __name__ == "__main__":
    pass
