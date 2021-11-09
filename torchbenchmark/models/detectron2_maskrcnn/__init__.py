import torch
import os
import itertools

from detectron2.config import instantiate
from detectron2 import model_zoo
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

    def __init__(self, device=None, jit=False):
       super().__init__()
       self.device = device
       self.jit = jit
       model_cfg = model_zoo.get_config("commons/models/mask_rcnn_fpn.py").model
       model = instantiate(model_cfg).to(self.device)

       # setup environment variable
       current_dir = os.path.dirname(os.path.realpath(__file__))
       data_dir = os.path.join(current_dir, ".data", "detectron2_maskrcnn_benchmark_data")
       os.environ['DETECTRON2_DATASETS'] = data_dir
       data_cfg = model_zoo.get_config("commons/data/coco.py").dataloader

       # use a mini dataset
       data_cfg.train.dataset.names = "coco_2017_val_100"
       data_cfg.train.total_batch_size = 4
       data_cfg.test.dataset.names = "coco_2017_val_100"

       train_loader = instantiate(data_cfg.train)
       self.train_iterator = itertools.cycle(itertools.islice(train_loader, 100))
       test_loader = instantiate(data_cfg.test)
       self.test_iterator = itertools.cycle(itertools.islice(test_loader, 100))

       self.optimizer = torch.optim.SGD(model.parameters(), 0.)

    def get_module(self):
        return self.module, (self.example_inputs, )

    def train(self, niter=1):
        self.model.train()
        for _ in range(niter):
            for idx, data in enumerate(self.train_iterator):
                if idx >= 100:
                    break
                losses = model(data)
                loss = sum(losses.values())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def eval(self, niter=1):
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                for idx, data in enumerate(self.test_iterator):
                    if idx >= 100:
                        break
                    model(data)
        
