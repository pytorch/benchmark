from abc import ABC, abstractmethod
from typing import Any, Callable
from torch.utils.benchmark import Timer
from torch.utils._pytree import tree_flatten


class BenchmarkCase(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self) -> Callable:
        pass


def time(fn: Callable, test_runs: int) -> float:
    t = Timer(stmt="fn()", globals={"fn": fn})
    times = t.blocked_autorange()
    return times.median * 1000  # time in ms


def benchmark(case: BenchmarkCase, warmup_runs: int = 10, test_runs: int = 20) -> float:
    for _ in range(warmup_runs):
        case.run()

    return time(case.run, test_runs)
