"""
Utilities to run experiments across models.
Utility functions in this file don't handle exceptions.
They expect their callers to handle exceptions themselves.
"""
import os
import torch
import time
import dataclasses
from typing import Optional, List, Dict
from torchbenchmark import ModelTask

WARMUP_ROUNDS = 3
WORKER_TIMEOUT = 600 # seconds
BS_FIELD_NAME = "batch_size"
NANOSECONDS_PER_MILLISECONDS = 1_000_000.0

@dataclasses.dataclass
class TorchBenchModelConfig:
    name: str
    device: str
    test: str
    batch_size: Optional[int] = None
    jit: bool = False
    extra_args: List[str] = []
    extra_env: Optional[Dict[str, str]] = None

@dataclasses.dataclass
class TorchBenchModelMetrics:
    precision: str
    latencies: List[float] = []
    extra_metrics: Dict[str, float] = {}

def get_latencies(func, device: str, nwarmup=WARMUP_ROUNDS, num_iter=10) -> List[float]:
    "Run one step of the model, and return the latency in milliseconds."
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        func()
    result_summary = []
    for _i in range(num_iter):
        if device == "cuda":
            torch.cuda.synchronize()
            # Collect time_ns() instead of time() which does not provide better precision than 1
            # second according to https://docs.python.org/3/library/time.html#time.time.
            t0 = time.time_ns()
            func()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            t1 = time.time_ns()
        else:
            t0 = time.time_ns()
            func()
            t1 = time.time_ns()
        result_summary.append((t1 - t0) / NANOSECONDS_PER_MILLISECONDS)
    return result_summary

def get_model_task(name: str, timeout: float=WORKER_TIMEOUT, extra_env: Optional[Dict[str, str]]=None) -> ModelTask:
    task = ModelTask(name, timeout=timeout, extra_env=extra_env)
    if not task.model_details.exists:
        raise ValueError("Failed to import model task: {name}. Please run the model manually to make sure it succeeds, or report a bug.")
    return task

def create_model_instance(task: ModelTask, device: str, test: str, batch_size: str,
                          jit: bool, extra_args: List[str]):
    config = TorchBenchModelConfig(name=task._details.name, device=device, test=test, batch_size=batch_size, jit=jit,
                                   extra_args=extra_args)
    task.make_model_instance(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
    batch_size = task.get_model_attribute(BS_FIELD_NAME)
    if config.batch_size and (not batch_size == config.batch_size):
        raise ValueError(f"User specify batch size {config.batch_size}," +
                         f"but model {config.name} runs with batch size {batch_size}. Please report a bug.")

def get_model_metrics(task: ModelTask) -> TorchBenchModelMetrics:
    pass
