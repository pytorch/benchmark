import argparse
import os
import csv
from typing import List

from torchbenchmark import REPO_PATH


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchBench Addmm operator Benchmark")
    parser.add_argument("--m", default=8, type=int)
    parser.add_argument("--k", default=8, type=int)
    parser.add_argument("--n", default=8, type=int)
    parser.add_argument("--input", default=None, type=str)
    args = parser.parse_args(args)
    return args
