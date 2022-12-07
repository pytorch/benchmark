"""
Utilities to instantiate TorchBench models.
Functions in this file don't handle exceptions.
They expect callers handle all exceptions.
"""
import os
import dataclasses
from typing import Optional, List, Dict, Union
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark import _list_model_paths, ModelTask

WORKER_TIMEOUT = 600 # seconds
BS_FIELD_NAME = "batch_size"

@dataclasses.dataclass
class TorchBenchModelConfig:
    name: str
    device: str
    test: str
    batch_size: Optional[int]
    jit: bool
    extra_args: List[str]
    extra_env: Optional[Dict[str, str]] = None

def load_model_isolated(config: TorchBenchModelConfig, timeout: float=WORKER_TIMEOUT, extra_env: Optional[Dict[str, str]]=None) -> ModelTask:
    pass

def load_model(config: TorchBenchModelConfig) -> BenchmarkModel:
    """Load and return a model instance within the same process. """
    pass

def list_models() -> List[str]:
    """Return a list of names of all TorchBench models"""
    model_paths = _list_model_paths()
    model_names = list(map(lambda x: os.path.basename(x), model_paths))
    return model_names
