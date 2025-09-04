"""
Generate model traces that are runnable in Tritonbench.
Example trace:
"""

import argparse
from typing import List


class ModelTrace:
    pass


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None, help="Model to trace, split by comma."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train,eval",
        help="Mode to trace, default is train,eval.",
    )
    parser.add_argument("--output", type=str, help="Output directory.")
    return parser


def trace_model():
    pass


def run(args: List[str]):
    parser = get_parser()
    args = parser.parse_args(args)
    run_traces(extra_args=args)
