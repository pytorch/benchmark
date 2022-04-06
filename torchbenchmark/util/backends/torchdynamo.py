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
    model.add_context(functools.partial(torchdynamo.optimize, args.torchdynamo))
    torchdynamo.reset()
