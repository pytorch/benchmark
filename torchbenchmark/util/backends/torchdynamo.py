"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import os
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


def patch_torchdynamo():
    """Patch TorchDynamo to workaround a performance issue:
       https://github.com/facebookresearch/torchdynamo/issues/159"""
    import patch
    import torchdynamo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(current_dir, "torchdynamo.patch")
    torchdynamo_dir = os.path.dirname(torchdynamo.__file__)
    p = patch.fromfile(patch_file)
    if not p.apply(strip=1, root=torchdynamo_dir):
        print("Failed to patch torchdynamo. Exit.")
        exit(1)


def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    patch_torchdynamo()
    import torchdynamo
    if args.torchdynamo == "fx2trt" and precision == "fp16":
        model.add_context(functools.partial(torchdynamo.optimize, torchdynamo.optimizations.backends.fx2trt_compiler_fp16))
    else:
        model.add_context(functools.partial(torchdynamo.optimize, args.torchdynamo))
    torchdynamo.reset()
