import argparse
import importlib
from typing import List
from torchbenchmark.microbenchmarks.nvfuser import run_nvfuser_microbenchmarks

def list_microbenchmarks() -> List[str]:
    pass

def run():
    parser = argparse.ArgumentParser(description="Run TorchBench microbenchmarks")
    parser.add_argument("bm_name", help='name of the microbenchmark')
    parser.add_argument("--filter", nargs="*", default=[], help='List of benchmarks to test')
    args, extra_args = parser.parse_known_args()
    args = parser.parse_args()

    microbenchmark = importlib.import_module(f"torchbenchmark.microbenchmarks.{args}")
    run_nvfuser_microbenchmarks(args.filter, extra_args)


if __name__ == "__main__":
    run()
