import os
import traceback
import argparse
import importlib
from pathlib import Path
from typing import Dict

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def list_benchmarks() -> Dict[str, str]:
    benchmarks = {}
    import userbenchmark
    bdir = Path(userbenchmark.__file__).parent.resolve()
    fb_bdir = bdir.joinpath("fb")
    if fb_bdir.exists():
        for fb_bm in filter(lambda x: x.is_dir(), fb_bdir.iterdir()):
            benchmarks[fb_bm.name] = f"fb.{fb_bm.name}"
    for bm in filter(lambda x: x.is_dir() and not x.name == "fb", bdir.iterdir()):
        benchmarks[bm.name] = bm.name
    return benchmarks

def run():
    available_benchmarks = list_benchmarks()
    parser = argparse.ArgumentParser(description="Run a TorchBench user benchmark")
    parser.add_argument("bm_name", choices=available_benchmarks.keys(), help='name of the user benchmark')
    args, bm_args = parser.parse_known_args()

    try:
        benchmark = importlib.import_module(f"userbenchmark.{available_benchmarks[args.bm_name]}")
        benchmark.run(bm_args)
    except ImportError as e:
        print(f"Failed to import user benchmark module {args.bm_name}, error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run()
