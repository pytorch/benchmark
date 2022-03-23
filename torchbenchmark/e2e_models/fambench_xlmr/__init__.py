import os
from pathlib import Path

from torchbenchmark.util.e2emodel import E2EBenchmarkModel

from typing import Optional, List

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

class Model(E2EBenchmarkModel):
    def __init__(self, test: str, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        super().__init__(test=test, batch_size=batch_size, extra_args=extra_args)

    def train(self):
        pass

    def eval(self):
        pass