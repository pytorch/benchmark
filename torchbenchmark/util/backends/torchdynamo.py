"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
from typing import List

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    args = parser.parse_args(extra_args)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--coverage", action="store_true", help="(default) " + help(coverage_experiment)
    )
    group.add_argument(
        "--online-autotune", action="store_true", help=help(speedup_experiment)
    )
    group.add_argument(
        "--offline-autotune", action="store_true", help=help(speedup_experiment)
    )
    group.add_argument(
        "--speedup-fixed1",
        action="store_true",
        help="speedup using experimental fixed_strategy backend",
    )
    group.add_argument(
        "--speedup-fixed2",
        action="store_true",
        help="speedup using experimental fixed_strategy backend",
    )
    group.add_argument(
        "--speedup-ltc",
        action="store_true",
        help="speedup using the ltc backend",
    )
    group.add_argument(
        "--speedup-ltc-trivial",
        action="store_true",
        help="speedup using the ltc backend without reusing compiled graph",
    )
    group.add_argument(
        "--overhead", action="store_true", help=help(overhead_experiment)
    )
    group.add_argument(
        "--speedup-ts", action="store_true", help=help(speedup_experiment_ts)
    )
    group.add_argument(
        "--speedup-sr", action="store_true", help=help(speedup_experiment_sr)
    )
    group.add_argument(
        "--speedup-onnx", action="store_true", help=help(speedup_experiment_onnx)
    )
    group.add_argument(
        "--speedup-trt", action="store_true", help=help(speedup_experiment_trt)
    )
    group.add_argument(
        "--accuracy-aot-nop",
        action="store_true",
        help="Accuracy testing for AOT vs Eager",
    )
    group.add_argument(
        "--speedup-aot-efficient-fusion",
        action="store_true",
        help="speedup using experimental fixed_strategy backend",
    )
    group.add_argument("--nothing", action="store_true", help=help(null_experiment))
    group.add_argument(
        "--nops",
        action="store_true",
        help="Test that bytecode rewriting works properly.",
    )
    args = parser.parse_args()
    return args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]):
    pass