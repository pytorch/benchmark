"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import functools
from typing import List
import torchdynamo
from .blade import blade_optimize_dynamo

TORCHDYNAMO_ROUNDS = 3
def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', dynamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends()
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    parser.add_argument(
        "--trt", action='store_true', help="use blade trt backend"
    )
    args, extra_args = parser.parse_known_args(dynamo_args)
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    torchdynamo.config.raise_on_backend_error = True
    torchdynamo.reset()
    torchdynamo.utils.counters.clear()

    if args.torchdynamo == "fx2trt" and precision == "fp16":
        dynamo_optimizer = torchdynamo.optimize(torchdynamo.optimizations.backends.fx2trt_compiler_fp16)
    elif "blade" in args.torchdynamo:
        dynamo_optimizer = torchdynamo.optimize(functools.partial(blade_optimize_dynamo, enable_fp16=precision=="fp16", use_trt=args.trt))
    elif "ipex" in args.torchdynamo:
        if precision == "fp32":
            dynamo_optimizer = torchdynamo.optimize(torchdynamo.optimizations.backends.ipex_fp32)
        else:
            dynamo_optimizer = torchdynamo.optimize(torchdynamo.optimizations.backends.ipex_bf16)
    else:
        dynamo_optimizer = torchdynamo.optimize(args.torchdynamo)
    if model.test == "train":
        model.train = dynamo_optimizer(model.train)
    else:
        model.eval = dynamo_optimizer(model.eval)
    torchdynamo.reset()
    
    for _ in range(TORCHDYNAMO_ROUNDS):
        model.invoke()
