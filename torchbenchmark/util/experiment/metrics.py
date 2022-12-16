"""
Utilities to measure metrics of a model.
"""
import torch
import time
import dataclasses
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark import ModelTask
from typing import List, Union

WARMUP_ROUNDS = 10
BENCHMARK_ITERS = 15
NANOSECONDS_PER_MILLISECONDS = 1_000_000.0

@dataclasses.dataclass
class TorchBenchModelMetrics:
    latencies: List[float]

def get_latencies(func, device: str, nwarmup=WARMUP_ROUNDS, num_iter=BENCHMARK_ITERS) -> List[float]:
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

def _get_model_test_metrics(model: BenchmarkModel) -> TorchBenchModelMetrics:
    latencies = get_latencies(model.invoke, model.device)
    return TorchBenchModelMetrics(latencies=latencies)

def _get_model_test_metrics_isolated(model: ModelTask) -> TorchBenchModelMetrics:
    device = model.get_model_attribute("device")
    latencies = get_latencies(model.invoke, device)
    return TorchBenchModelMetrics(latencies=latencies)

def get_model_test_metrics(model: Union[BenchmarkModel, ModelTask]) -> TorchBenchModelMetrics:
    if isinstance(model, BenchmarkModel):
        return _get_model_test_metrics(model)
    elif isinstance(model, ModelTask):
        return _get_model_test_metrics_isolated(model)
    else:
        raise ValueError(f"Expected BenchmarkModel or ModelTask, get type: {type(model)}")