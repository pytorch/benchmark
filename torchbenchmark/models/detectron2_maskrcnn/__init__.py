import torch
import itertools

from detectron2.config import instantiate
from detectron2 import model_zoo
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION

