import torch
import sys
import os
from torchbenchmark import REPO_PATH
from typing import Tuple

# Import FAMBench model path
class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass
RNNT_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "rnnt", "ootb", "train")

with add_path(RNNT_PATH):
    from common.data.dali.data_loader import DaliDataLoader

from .config import FambenchRNNTConfig
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH

class Model(BenchmarkModel):
    RNNT_TRAIN_CONFIG = FambenchRNNTConfig()
    # Default train batch size: 1024. Source:
    # https://github.com/facebookresearch/FAMBench/blob/main/benchmarks/rnnt/ootb/train/scripts/train.sh#L28
    DEFAULT_TRAIN_BATCH_SIZE = RNNT_TRAIN_CONFIG.train_batch_size
    DEFAULT_EVAL_BATCH_SIZE
    DEFAULT_EVAL_BATCH_SIZE = 64
    # run only 1 batch
    DEFAULT_NUM_BATCHES = 1

    def __init__(self, ):
        self.config = FambenchRNNTConfig()