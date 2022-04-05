import torch
import sys
import os
from torchbenchmark import REPO_PATH
from typing import Tuple
from torchbenchmark.models

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
DLRM_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "rnnt", "ootb")
with add_path(DLRM_PATH):
    pass

from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH

class Model(BenchmarkModel):
    DEFAULT_EVAL_BATCH_SIZE = 64
    # run only 1 batch
    DEFAULT_NUM_BATCHES = 1