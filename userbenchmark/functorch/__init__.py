import torch
from ..utils import dump_output
from .cases import benchmark_cases
from .util import benchmark
import pprint
from typing import List


BM_NAME = 'functorch'


def run_benchmarks():
    metrics = {}

    for case_ctor in benchmark_cases:
        case = case_ctor()
        runtime_ms = benchmark(case)
        metrics[case.name()] = runtime_ms
    return metrics


def run(args: List[str]):
    metrics = run_benchmarks()
    result = {
        'name': BM_NAME,
        'environ': {
            'git_version': torch.version.git_version,
        },
        'metrics': metrics,
    }
    pprint.pprint(result)
    dump_output(BM_NAME, result)
