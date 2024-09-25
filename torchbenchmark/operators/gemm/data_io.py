import argparse
import csv
import os
from typing import List

from torchbenchmark import REPO_PATH


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchBench Gemm operator Benchmark")
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--bias", type=int)
    parser.add_argument("--input", type=str)
    parser.add_argument("--splitk", action="store_true", default=False)
    parser.add_argument("--llama", action="store_true", default=False)
    args = parser.parse_args(args)
    return args


def read_shapes_from_csv(csv_path: str) -> List[List[int]]:
    input_file_path = os.path.join(
        REPO_PATH, "torchbenchmark", "operators", "gemm", csv_path
    )
    shapes = []
    with open(input_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = [
                int(row.get(f)) if row.get(f) else None for f in ("M", "N", "K", "Bias")
            ]
            shapes.append(shape)
    return shapes
