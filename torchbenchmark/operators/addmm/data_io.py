import argparse
import csv
import os
from typing import List

from torchbenchmark import REPO_PATH


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchBench Addmm operator Benchmark")
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--input", type=str)
    parser.add_argument("--col-major", type=bool, default=False)
    parser.add_argument("--large-k-shapes", type=bool, default=False)
    args = parser.parse_args(args)
    return args
