"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import functools
from typing import List
import torchdynamo

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', dyamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends()
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    args, extra_args = parser.parse_known_args(dyamo_args)
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    if args.torchdynamo == "fx2trt" and precision == "fp16":
        model.add_context(functools.partial(torchdynamo.optimize, torchdynamo.optimizations.backends.fx2trt_compiler_fp16))
    else:
        model.add_context(functools.partial(torchdynamo.optimize, args.torchdynamo))
    torchdynamo.reset()
