"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
from typing import List

class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def help(fn):
    return fn.__doc__

def null_experiment(args, model_iter_fn, model, example_inputs):
    """
    A no-op experiment useful for making sure TorchBenchmark alone works properly.
    """

    return []

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    args = parser.parse_args(extra_args)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nothing", action="store_true", help=help(null_experiment))
    group.add_argument(
        "--nops",
        action="store_true",
        help="Test that bytecode rewriting works properly.",
    )
    parser.add_argument(
        "--nopython", action="store_true", help="Turn graph breaks into errors"
    )
    args = parser.parse_args(extra_args)
    return args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    import torchdynamo
    optimize_ctx = NullContext()
    if args.nothing:
        pass
    elif args.nops:
        optimize_ctx = torchdynamo.eval_frame._optimize_catch_errors(
            torchdynamo.testing.debug_insert_nops, nopython=args.nopython
        )
    model.add_context(lambda: optimize_ctx)
    torchdynamo.reset()
