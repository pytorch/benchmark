"""
Measure the performance stableness of a model
"""

import dataclasses
from typing import List

from torchbenchmark import ModelTask
from typing import Tuple 
WARMUP_ROUNDS = 3

@dataclasses.dataclass
class Metric:
    p0: float
    p50: float
    p100: float
    stdev: float

class TBExperiment():
    pass

def _run_one_step(model_task: ModelTask, nwarmup=WARMUP_ROUNDS, num_iter=10) -> Tuple[float, Optional[Tuple[torch.Tensor]]]:
    "Run one step of the model, and return the latency in milliseconds."
    NANOSECONDS_PER_MILLISECONDS = 1_000_000.0
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
    wall_latency = numpy.median(result_summary)
    return wall_latency