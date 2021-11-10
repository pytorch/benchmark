import torch
import os
import itertools
import random
import numpy as np

# TorchBench imports
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from detectron2.config import instantiate
from detectron2 import model_zoo
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    def __init__(self, device=None, jit=False, train_bs=10, eval_bs=10):
       super().__init__()
       self.device = device
       self.jit = jit
       model_cfg = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
       self.model = instantiate(model_cfg).to(self.device)
       self.train_bs = train_bs
       self.eval_bs = eval_bs

       # setup environment variable
       current_dir = os.path.dirname(os.path.realpath(__file__))
       data_dir = os.path.join(current_dir, ".data", "detectron2_maskrcnn_benchmark_data")
       os.environ['DETECTRON2_DATASETS'] = data_dir
       data_cfg = model_zoo.get_config("common/data/coco.py").dataloader

       # use a mini dataset
       data_cfg.train.dataset.names = "coco_2017_val_100"
       data_cfg.train.total_batch_size = 4
       data_cfg.test.dataset.names = "coco_2017_val_100"

       train_loader = instantiate(data_cfg.train)
       self.train_iterator = itertools.cycle(itertools.islice(train_loader, 100))
       test_loader = instantiate(data_cfg.test)
       self.test_iterator = itertools.cycle(itertools.islice(test_loader, 100))

       self.optimizer = torch.optim.SGD(self.model.parameters(), 0.)

    def get_module(self):
        return self.module, (self.example_inputs, )

    def train(self, niter=1):
        if not self.device == "cuda":
            raise NotImplementedError("Only CUDA is supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        self.model.train()
        with EventStorage():
            for _ in range(niter):
                for idx, data in zip(range(self.train_bs), self.train_iterator):
                    losses = self.model(data)
                    loss = sum(losses.values())
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def eval(self, niter=1):
        if not self.device == "cuda":
            raise NotImplementedError("Only CUDA is supported by this model")
        if self.jit:
            raise NotImplementedError("JIT is not supported by this model")
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                for idx, data in zip(range(self.eval_bs), self.test_iterator):
                    self.model(data)
        
