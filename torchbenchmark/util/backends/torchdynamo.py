"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
from typing import List
import torchdynamo

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', dynamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends()
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    parser.add_argument(
        "--extra-py-args", type=str, help="Extra Python args to evaluate."
    )
    args, extra_args = parser.parse_known_args(dynamo_args)
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    if args.torchdynamo == "fx2trt" and precision == "fp16":
        dynamo_optimizer = torchdynamo.optimize(torchdynamo.optimizations.backends.fx2trt_compiler_fp16)
    else:
        dynamo_optimizer = torchdynamo.optimize(args.torchdynamo)
    if model.test == "train":
        model.train = dynamo_optimizer(model.train)
    else:
        model.eval = dynamo_optimizer(model.eval)
    # evaluate extra python code passed by the user
    if args.extra_py_args:
        eval(args.extra_py_args)
    torchdynamo.reset()
