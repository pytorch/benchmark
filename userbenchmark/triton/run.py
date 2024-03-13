import argparse
import sys
import triton

from typing import List

from torchbenchmark.util.operator import (
    DEFAULT_RUN_ITERS,
    DEFAULT_WARMUP,
    BenchmarkOperator,
)
from torchbenchmark.operators import load_opbench_by_name

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser(description="Run TorchBench Triton operator benchmark")
    # This option is used to filter test cases to run.
    parser.add_argument(
        "--op",
        help="Name of the operator to test",
        required=True,
        default=None,
    )

    parser.add_argument(
        "--device",
        "-d",
        help="Name of the device to run the benchmark",
        default="cuda",
        type=str,
    )

    parser.add_argument(
        "--iter",
        help="Repeat each operator for the number of iterations",
        default=DEFAULT_RUN_ITERS,
        type=int,
    )

    parser.add_argument(
        "--warmup",
        help="Number of iterations to ignore before measuring performance",
        default=DEFAULT_WARMUP,
        type=int,
    )

    parser.add_argument(
        "--training",
        action="store_true",
        help="Run both forward and backward passes",
    )
    return parser.parse_known_args(args)


def run(args: List[str]):
    args, extra_args = parse_args(args)
    Opbench = load_opbench_by_name(args.op)
    test = "train" if args.training else "eval"
    opbench = Opbench(test=test, device=args.device, extra_args=extra_args)
    metrics = opbench.run(args.warmup, args.iter)
    print(metrics)
    print(metrics.csv)
    try:
        opbench.plot()
    except NotImplementedError:
        print(f"Plotting is not implemented for {args.op}")
