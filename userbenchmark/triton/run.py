import triton
import argparse
import sys
from typing import Tuple, List

from torchbenchmark.util.triton_op import (
    DEFAULT_RUN_ITERS,
    DEFAULT_WARMUP,
    BenchmarkOperator,
)
from torchbenchmark.operators import load_opbench_by_name

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default=None, help="Operator to benchmark.")
    parser.add_argument("--training", action="store_true", help="Run both forward and backward pass.")
    parser.add_argument("--device", default="cuda", help="Device to benchmark.")
    parser.add_argument("--warmup", default=DEFAULT_WARMUP, help="Device to benchmark.")
    parser.add_argument("--iter", default=DEFAULT_RUN_ITERS, help="Device to benchmark.")
    parser.add_argument("--csv", action="store_true", help="Dump result as csv.")
    parser.add_argument("--plot", action="store_true", help="Plot the result.")
    return parser.parse_known_args(args)


def run(args: List[str]=[]):
    if args == []:
        args = sys.argv[1:]
    args, extra_args = parse_args(args)
    Opbench = load_opbench_by_name(args.op)
    test = "train" if args.training else "eval"
    opbench = Opbench(
        test=test,
        device=args.device,
        extra_args=extra_args,
    )
    metrics = opbench.run(args.warmup, args.iter)
    if args.csv:
        print(metrics.csv)
    else:
        print(metrics)
    if args.plot:
        try:
            opbench.plot()
        except NotImplementedError:
            print(f"Plotting is not implemented for {args.op}")
