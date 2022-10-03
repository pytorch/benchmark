"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import contextlib
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
    parser.add_argument(
        "--optimize_dynamo_ddp",
        action='store_true',
        help="enable extra optimizations for DDP + dynamo"
    )
    args, extra_args = parser.parse_known_args(dynamo_args)
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    optimize_ddp_context = contextlib.nullcontext

    if args.optimize_dynamo_ddp:
        import torchdynamo
        @contextlib.contextmanager
        def optimize_ddp_ctx(val: bool):
            old_value = torchdynamo.config.optimize_ddp
            try:
                torchdynamo.config.optimize_ddp = val
                yield
            finally:
                torchdynamo.config.optimize_ddp = old_value
        optimize_ddp_context = lambda: optimize_ddp_ctx(True)

    with optimize_ddp_context():
        if args.torchdynamo == "fx2trt" and precision == "fp16":
            dynamo_optimizer = torchdynamo.optimize(torchdynamo.optimizations.backends.fx2trt_compiler_fp16)
        else:
            dynamo_optimizer = torchdynamo.optimize(args.torchdynamo)
        # evaluate extra python code passed by the user
        if args.extra_py_args:
            exec(args.extra_py_args)
        if model.test == "train":
            model.train = dynamo_optimizer(model.train)
        else:
            model.eval = dynamo_optimizer(model.eval)

    model.add_context(optimize_ddp_context)

    torchdynamo.reset()
