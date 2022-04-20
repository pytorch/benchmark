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
DLRM_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "dlrm", "ootb")
with add_path(DLRM_PATH):
    from dlrm_s_pytorch import DLRM_Net

from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import RECOMMENDATION
from .config import FAMBenchTrainConfig, FAMBenchEvalConfig

class Model(BenchmarkModel):
    task = RECOMMENDATION.RECOMMENDATION
    FAMBENCH_MODEL = True
    # config
    DEFAULT_EVAL_ARGS = FAMBenchEvalConfig()
    DEFAULT_TRAIN_ARGS = FAMBenchTrainConfig()
    DEFAULT_EVAL_BATCH_SIZE = DEFAULT_EVAL_ARGS.mini_batch_size
    DEFAULT_TRAIN_BATCH_SIZE = DEFAULT_TRAIN_ARGS.mini_batch_size
    # run only 1 batch
    DEFAULT_NUM_BATCHES = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(self, test, device, batch_size, jit, extra_args)
        self.dlrm = DLRM_Net()