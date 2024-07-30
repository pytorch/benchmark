import argparse
import os
import sys
import tempfile
from typing import List

from torch import version as torch_version
from torchbenchmark.operators import load_opbench_by_name

from torchbenchmark.util.triton_op import (
    BenchmarkOperatorResult,
    DEFAULT_RUN_ITERS,
    DEFAULT_WARMUP,
)

try:
    import torch
    if not hasattr(torch.version, "git_version"):
        from pytorch.benchmark.fb.run_utils import usage_report_logger
    else:
        usage_report_logger = lambda *args, **kwargs: None
except ImportError:
    usage_report_logger = lambda *args, **kwargs: None
from .gpu import gpu_lockdown

TRITON_BENCH_CSV_DUMP_PATH = tempfile.gettempdir() + "/tritonbench/"

def get_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--op",
        type=str,
        required=True,
        help="Operator to benchmark."
    )
    parser.add_argument(
        "--mode",
        choices=["fwd", "bwd", "fwd_bwd"],
        default="fwd",
        help="Test mode (fwd, bwd, or fwd_bwd).",
    )
    parser.add_argument(
        "--bwd",
        action="store_true",
        help="Run backward pass."
    )
    parser.add_argument(
        "--fwd_bwd",
        action="store_true",
        help="Run both forward and backward pass.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        default=DEFAULT_WARMUP,
        help="Num of warmup runs for reach benchmark run.",
    )
    parser.add_argument(
        "--iter",
        default=DEFAULT_RUN_ITERS,
        help="Num of reps for each benchmark run.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print result as csv.",
    )
    parser.add_argument(
        "--dump-csv",
        action="store_true",
        help="Dump result as csv.",
    )
    parser.add_argument(
        "--skip-print",
        action="store_true",
        help="Skip printing result.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the result.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run in the CI mode.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Metrics to collect, split with comma. E.g., --metrics latency,tflops,speedup.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Specify one or multiple operator implementations to run."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Override default baseline."
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        help="Number of example inputs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
    )
    parser.add_argument(
        "--input-id",
        type=int,
        default=0,
        help="Specify the start input id to run. " \
            "For example, --input-id 0 runs only the first available input sample." \
            "When used together like --input-id <X> --num-inputs <Y>, start from the input id <X> " \
            "and run <Y> different inputs."
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run this under test mode, potentially skipping expensive steps like autotuning."
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Dump Triton IR",
    )
    parser.add_argument(
        "--gpu-lockdown",
        action="store_true",
        help="Lock down GPU frequency and clocks to avoid throttling."
    )
    if not hasattr(torch_version, "git_version"):
        parser.add_argument("--log-scuba", action="store_true", help="Log to scuba.")
    return parser

def _run(args: argparse.Namespace, extra_args: List[str]) -> BenchmarkOperatorResult:
    Opbench = load_opbench_by_name(args.op)
    if args.fwd_bwd:
        args.mode = "fwd_bwd"
    if args.bwd:
        args.mode = "bwd"
    opbench = Opbench(
        tb_args=args,
        extra_args=extra_args,
    )
    try:
        opbench.run(args.warmup, args.iter)
    finally:
        metrics = opbench.output
        if not args.skip_print:
            if args.csv:
                metrics.write_csv_to_file(sys.stdout)
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
            os.makedirs(TRITON_BENCH_CSV_DUMP_PATH, exist_ok=True)
            path = metrics.write_csv(TRITON_BENCH_CSV_DUMP_PATH)
            print(f"[TritonBench] Dumped csv to {path}")
        return metrics

def run(args: List[str] = []):
    if args == []:
        args = sys.argv[1:]
    # Log the tool usage
    usage_report_logger(benchmark_name="tritonbench")
    parser = get_parser()
    args, extra_args = parser.parse_known_args(args)
    if args.ci:
        from .ci import run_ci
        run_ci()
        return
    with gpu_lockdown(args.gpu_lockdown):
        _run(args, extra_args)
