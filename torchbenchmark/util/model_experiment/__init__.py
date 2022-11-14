"""
Utilities to run experiments across models.
"""
import os
import dataclasses
from typing import Optional, List, Dict
from torchbenchmark import ModelTask

WORKER_TIMEOUT = 600 # seconds

@dataclasses.dataclass
class TorchBenchModelConfig:
    device: str
    test: str
    batch_size: Optional[int] = None
    jit: bool = False
    extra_args: List[str] = []
    extra_env: Optional[Dict[str, str]] = None

@dataclasses.dataclass
class TorchBenchModelMetrics:
    config: TorchBenchModelConfig
    precision: str
    latencies: List[float]
    extra_metrics: Dict[str, float]

def get_model_task(name: str):
    task = ModelTask(name, timeout=WORKER_TIMEOUT)
    if not task.model_details.exists:
        return None
    return task

def create_model_instance(task: ModelTask, device: str, test: str, batch_size: str,
                          jit: bool, extra_args: List[str]):
    config = TorchBenchModelConfig(device=device, test=test, batch_size=batch_size, jit=jit,
                                   extra_args=extra_args)
    task.make_model_instance(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)


def get_model_metrics(task: ModelTask) -> TorchBenchModelMetrics:
    pass
