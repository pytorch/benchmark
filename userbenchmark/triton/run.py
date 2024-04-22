import argparse
import os
import sys
from typing import List, Tuple

import triton

from torch import version as torch_version
from torchbenchmark.operators import load_opbench_by_name

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    DEFAULT_RUN_ITERS,
    DEFAULT_WARMUP,
)

TRITON_BENCH_CSV_DUMP_PATH = "/tmp/triton_bench/"


def parse_args(args):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--op", type=str, default=None, help="Operator to benchmark.")
    parser.add_argument(
        "--mode",
        choices=["fwd", "bwd", "fwd_bwd"],
        default="fwd",
        help="Test mode (fwd, bwd, or fwd_bwd).",
    )
    parser.add_argument("--bwd", action="store_true", help="Run backward pass.")
    parser.add_argument(
        "--fwd_bwd", action="store_true", help="Run both forward and backward pass."
    )
    parser.add_argument("--device", default="cuda", help="Device to benchmark.")
    parser.add_argument(
        "--warmup",
        default=DEFAULT_WARMUP,
        help="Num of warmup runs for reach benchmark run.",
    )
    parser.add_argument(
        "--iter", default=DEFAULT_RUN_ITERS, help="Num of reps for each benchmark run."
    )
    parser.add_argument("--csv", action="store_true", help="Print result as csv.")
    parser.add_argument("--dump-csv", action="store_true", help="Dump result as csv.")
    parser.add_argument("--skip-print", action="store_true", help="Skip printing result.")
    parser.add_argument("--plot", action="store_true", help="Plot the result.")
    if not hasattr(torch_version, "git_version"):
        parser.add_argument("--log-scuba", action="store_true", help="Log to scuba.")
    return parser.parse_known_args(args)


def run(args: List[str] = []):
    if args == []:
        args = sys.argv[1:]
    args, extra_args = parse_args(args)
    Opbench = load_opbench_by_name(args.op)
    if args.fwd_bwd:
        args.mode = "fwd_bwd"
    if args.bwd:
        args.mode = "bwd"
    opbench = Opbench(
        mode=args.mode,
        device=args.device,
        extra_args=extra_args,
    )
    metrics = opbench.run(args.warmup, args.iter)
    if not args.skip_print:
        if args.csv:
            print(metrics.csv)
        else:
            print(metrics)
    if not hasattr(torch_version, "git_version") and args.log_scuba:
        from userbenchmark.triton.fb import log_benchmark

        log_benchmark(metrics)
    if args.plot:
        try:
            opbench.plot()
        except NotImplementedError:
            print(f"Plotting is not implemented for {args.op}")

    if args.dump_csv:
        if not os.path.exists(TRITON_BENCH_CSV_DUMP_PATH):
            os.mkdir(TRITON_BENCH_CSV_DUMP_PATH)

        csv_str = metrics.csv
        csv_str_hash = abs(hash(csv_str)) % (10**8)
        file_name = f"op_{args.op}_{csv_str_hash}.csv"
        file_path = os.path.join(TRITON_BENCH_CSV_DUMP_PATH, file_name)
        with open(file_path, "w") as f:
            f.write(csv_str)
        print(f"[TritonBench] Dumped csv to {file_path}")
