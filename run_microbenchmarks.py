import os
import argparse
import importlib
from typing import List

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MICROBENCHMARKS_DIR = os.path.join(CURRENT_DIR, "torchbenchmark", "microbenchmarks")

def list_microbenchmarks() -> List[str]:
    return os.listdir(MICROBENCHMARKS_DIR)

def run():
    parser = argparse.ArgumentParser(description="Run TorchBench microbenchmarks")
    parser.add_argument("bm_name", choices=list_microbenchmarks(), help='name of the microbenchmark')
    args, bm_args = parser.parse_known_args()

    try:
        microbenchmark = importlib.import_module(f"torchbenchmark.microbenchmarks.{args}")
    except ImportError as e:
        print(f"Failed to import microbenchmark module {args.bm_name}, error: {str(e)}")
    microbenchmark.run(bm_args)

if __name__ == "__main__":
    run()
