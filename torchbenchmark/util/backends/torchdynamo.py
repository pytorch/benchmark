"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import contextlib
import distutils.util
import os
import warnings
from typing import List

import torch
import torch._dynamo as torchdynamo
import torchbenchmark
from torchbenchmark.util.model import is_staged_train_test

def parse_torchdynamo_args(dynamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends(exclude_tags=None)
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    parser.add_argument(
        "--tritonmm", type=str, help="torchinductor.config.triton.mm configuration"
    )
    parser.add_argument(
        "--dynamic_shapes",
        action='store_true',
        help="dynamic shape and symbolic tracing",
    )
    parser.add_argument(
        "--pt2_debug_log",
        action='store_true',
        help="enable debug log for PT2 (dynamo, inductor, AOTAutograd)",
    )
    parser.add_argument(
        "--full_graph",
        action='store_true',
        help="capture full graph and no python",
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
    parser.add_argument(
        "--torchinductor_fallback_random",
        type=distutils.util.strtobool,
        default="false",
    )
    parser.add_argument(
        "--torchinductor_enable_group_fusion",
        action='store_true',
        help="enable group fusion in Inductor"
    )
    parser.add_argument(
        "--torchinductor_compile_threads",
        type=int,
        help="""
            Here are the precedence to decide compile_threads
            1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.  One may want to disable async compiling by
            setting this to 1 to make pdb happy.
            2. Set to 1 if it's win32 platform or it's a fbcode build
            3. decide by the number of CPU cores
            """
    )
    parser.add_argument(
        "--torchinductor_enable_batch_fusion",
        action='store_true',
        help="enable batch fusion in Inductor"
    )
    parser.add_argument(
        "--torchinductor_enable_max_autotune_gemm",
        action='store_true',
        help="Enable max autotune gemm"
    )
    parser.add_argument(
        "--torchinductor_enable_split_cat_fx_pass",
        action='store_true',
        help="enable split_cat_fx_pass in Inductor"
    )
    parser.add_argument(
        "--torchinductor_triton_unique_kernel_names",
        action='store_true',
        help="set to generate unique triton kernel names in Inductor"
    )
    parser.add_argument(
        "--torchinductor_post_grad_batch_fusion",
        type=distutils.util.strtobool,
        help="Enable BMM Linear Fusion."
    )
    parser.add_argument(
        "--dynamo_disable_optimizer_step",
        type=distutils.util.strtobool,
        default="false",
    )
    parser.add_argument(
        "--dump_triton",
        type=distutils.util.strtobool,
        default="false",
        help="Enable triton code dump by setting torch._inductor.config.debug",
    )
    args, extra_args = parser.parse_known_args(dynamo_args)
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    if args.torchdynamo == "fx2trt" and precision == "fp16":
        dynamo_optimizer = torchdynamo.optimize(torchdynamo.optimizations.backends.fx2trt_compiler_fp16)
    else:
        dynamo_kwargs = {}
        if args.dynamic_shapes:
            dynamo_kwargs["dynamic"] = True
        if args.full_graph:
            dynamo_kwargs["nopython"] = True
        dynamo_optimizer = torchdynamo.optimize(args.torchdynamo, **dynamo_kwargs)
        if args.pt2_debug_log:
            import logging
            torch._logging.set_logs(dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG)

    if args.torchdynamo == "inductor":
        import torch._inductor as torchinductor
        torchinductor.config.triton.cudagraphs = bool(args.torchinductor_cudagraph)
        if bool(args.torchinductor_post_grad_batch_fusion):
            torchinductor.config.post_grad_fusion_options["batch_linear"] = {}
        torch._inductor.config.debug = bool(args.dump_triton)

        # Setup torchinductor.config.triton.mm
        if args.tritonmm == "triton":
            torchinductor.config.triton.mm = "triton"
            # currently can't pass correctness with use_bmm = True
            # torchinductor.config.triton.use_bmm = True
        if args.torchinductor_enable_group_fusion:
            torchinductor.config.group_fusion = True
        if compile_threads := args.torchinductor_compile_threads:
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
        if args.torchinductor_enable_batch_fusion:
            torchinductor.config.pattern_matcher = True
            torchinductor.config.batch_fusion = True
        if args.torchinductor_enable_split_cat_fx_pass:
            torchinductor.config.split_cat_fx_passes = True
        if args.torchinductor_triton_unique_kernel_names:
            torchinductor.config.triton.unique_kernel_names = True
        if args.torchinductor_enable_max_autotune_gemm:
            torchinductor.config.max_autotune_gemm = True

        # used for correctness checks, to avoid triton rand() behaving differently from torch rand().
        torchinductor.config.fallback_random = bool(args.torchinductor_fallback_random)

    if bool(args.dynamo_disable_optimizer_step):
        found_optimizer_step = False
        try:
            model.cfg.optimizer.step = torch._dynamo.disable(model.cfg.optimizer.step)
            found_optimizer_step = True
        except AttributeError:
            pass

        try:
            model.optimizer.step = torch._dynamo.disable(model.optimizer.step)
            found_optimizer_step = True
        except AttributeError:
            pass

        if not found_optimizer_step:
            warnings.warn("--dynamo_disable_optimizer_step is set to True, but the optimizer could not be found on this model")

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
