import argparse
from torchbenchmark.microbenchmarks.nvfuser import run_nvfuser_microbenchmarks


def run():
    parser = argparse.ArgumentParser(description="Run nvfuser microbenchmarks")
    parser.add_argument("--filter", nargs="*", default=[], help='List of benchmarks to test')
    args, extra_args = parser.parse_known_args()
    args = parser.parse_args()

    run_nvfuser_microbenchmarks(args.filter, extra_args)


if __name__ == "__main__":
    run()
