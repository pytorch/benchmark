"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import contextlib
import distutils.util
from typing import List, Optional
import torch._dynamo as torchdynamo
from torchbenchmark.util.model import is_staged_train_test

def parse_torchdynamo_args(
    model: 'torchbenchmark.util.model.BenchmarkModel',
    dynamo_args: List[str],
    namespace: Optional[argparse.Namespace] = None
) -> argparse.Namespace:
    # pass in `namespace` arg to add to an existing argparse.Namespace object

    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends()
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    parser.add_argument(
        "--tritonmm", type=str, help="torchinductor.config.triton.mm configuration"
    )
    parser.add_argument(
        "--optimize_dynamo_ddp",
        action='store_true',
        help="enable extra optimizations for DDP + dynamo"
    )
    parser.add_argument(
        "--torchinductor_cudagraph",
        type=distutils.util.strtobool,
        default="true",
    )
    args, extra_args = parser.parse_known_args(dynamo_args, namespace=namespace)
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    if args.torchdynamo == "fx2trt" and precision == "fp16":
        dynamo_optimizer = torchdynamo.optimize(torchdynamo.optimizations.backends.fx2trt_compiler_fp16)
    else:
        dynamo_optimizer = torchdynamo.optimize(args.torchdynamo)
    # Setup torchinductor.config.triton.mm
    if args.tritonmm == "triton":
        import torch._inductor as torchinductor
        torchinductor.config.triton.mm = "triton"
        # currently can't pass correctness with use_bmm = True
        # torchinductor.config.triton.use_bmm = True
    import torch._inductor as torchinductor
    torchinductor.config.triton.cudagraphs = bool(args.torchinductor_cudagraph)

    if model.test == "train":
        if is_staged_train_test(model):
            model.forward = dynamo_optimizer(model.forward)
        else:
            model.train = dynamo_optimizer(model.train)
    else:
        model.eval = dynamo_optimizer(model.eval)

    if args.optimize_dynamo_ddp:
        @contextlib.contextmanager
        def optimize_ddp_ctx(val: bool):
            old_value = torchdynamo.config.optimize_ddp
            try:
                torchdynamo.config.optimize_ddp = val
                yield
            finally:
                torchdynamo.config.optimize_ddp = old_value
        model.add_context(lambda: optimize_ddp_ctx(True))

    torchdynamo.reset()
