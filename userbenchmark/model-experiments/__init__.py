"""
Experiments that can run by sweeping through a list of models
"""
import dataclasses
from .args import parse_args
from .experiments.base import TBExperimentBase
from typing import List, Optional

@dataclasses.dataclass
class TorchBenchModelConfig:
    device: str
    test: str
    batch_size: Optional[int]
    jit: bool
    extra_args: List[str]

def loop(e: TBExperimentBase):
    pass


def run(args: List[str]):
    args, unknown_args = parse_args(args)
