import os
import argparse
import importlib
from typing import List

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
BENCHMARKS_DIR = os.path.join(CURRENT_DIR, "benchmarks")

def list_benchmarks() -> List[str]:
    return os.listdir(BENCHMARKS_DIR)

def run():
    parser = argparse.ArgumentParser(description="Run TorchBench benchmarks")
    parser.add_argument("bm_name", choices=list_benchmarks(), help='name of the benchmark')
    args, bm_args = parser.parse_known_args()

    try:
        benchmark = importlib.import_module(f"benchmarks.{args.bm_name}")
        benchmark.run(bm_args)
    except ImportError as e:
        print(f"Failed to import benchmark module {args.bm_name}, error: {str(e)}")

if __name__ == "__main__":
    run()
