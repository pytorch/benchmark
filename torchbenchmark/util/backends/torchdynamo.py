"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import functools
from typing import List

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', dyamo_args: List[str]) -> argparse.Namespace:
    import torchdynamo
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torchdynamo", choices=torchdynamo.list_backends(), help="Specify torchdynamo backends"
    )
    args = parser.parse_args(dyamo_args)
    return args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    import torchdynamo
    if args.torchdynamo == "fx2trt" and precision == "fp16":
        optimize_cxt = torchdynamo.optimize(torchdynamo.optimizations.backends.fx2trt_compiler_fp16)
    else:
        optimize_cxt = torchdynamo.optimize(torchdynamo.optimize(args.torchdynamo))
    with optimize_cxt:
        pass
    model.add_context(functools.partial(torchdynamo.run))
    torchdynamo.reset()
