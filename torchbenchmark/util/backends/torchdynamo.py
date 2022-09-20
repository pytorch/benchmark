"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
from typing import List
import torchdynamo
import logging

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', dynamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends()
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    args, extra_args = parser.parse_known_args(dynamo_args)
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    # torchdynamo.config.log_level = logging.DEBUG
    # torchdynamo.config.verbose = True
    from torchdynamo.utils import count_calls

    graph_id = 0
    if args.torchdynamo == "fx2trt" and precision == "fp16":
        compiler_fn = "fx2trt_compiler_fp16"
    else:
        compiler_fn = args.torchdynamo

    def ddp_compiler(gm, inputs):
        nonlocal graph_id
        nonlocal compiler_fn
        print(f"TORCHDYNAMO LOG - Graph id {graph_id} has {count_calls(gm.graph)} calls")
        graph_id += 1

        if compiler_fn == "inductor":
            from torchinductor.compile_fx import compile_fx

            compiler_fn = compile_fx
        elif isinstance(compiler_fn, str):
            from torchdynamo.optimizations import BACKENDS

            compiler_fn = BACKENDS[compiler_fn]
        return compiler_fn(gm, inputs)

    model.model = torchdynamo.optimize(ddp_compiler, nopython=True)(model.model)
    torchdynamo.reset()
