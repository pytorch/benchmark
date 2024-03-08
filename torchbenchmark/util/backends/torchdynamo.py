"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import contextlib
import distutils.util
import os
import functools
import warnings
from typing import List

import torch
import torchbenchmark
from torchbenchmark.util.model import is_staged_train_test

INDUCTOR_CONFIG_KEYS = [
    "triton.cudagraphs",
    "triton.unique_kernel_names",
    "fallback_random",
    "max_autotune_gemm",
    "split_cat_fx_passes",
    "group_fusion",
    "batch_fusion",
    "debug",
]

def parse_torchdynamo_args(dynamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torchdynamo",
        choices=["inductor"],
        default=None,
        help="Measure metrics with TorchInductor",
    )
    parser.add_argument(
        "--inductor",
        action="store_true",
        help="Measure metrics with TorchInductor",
    )
    parser.add_argument(
        "--inductor-compile-mode",
        default=None,
        choices=['max-autotune'],
        help="torch.compile mode argument for inductor runs.",
    )
    parser.add_argument(
        "--nopython",
        action="store_true",
        help="Turn graph breaks into errors"
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Runs a dynamic shapes version of the benchmark, if available.",
    )
    parser.add_argument(
        "--dynamic-batch-only",
        action="store_true",
        help="Only assume batch dimension is dynamic.  Implies --dynamic-shapes",
    )
    parser.add_argument(
        "--dynamo_disable_optimizer_step",
        type=distutils.util.strtobool,
        default="false",
    )
    parser.add_argument(
        "--pt2_debug_log",
        action='store_true',
        help="enable debug log for PT2 (dynamo, inductor, AOTAutograd)",
    )
    parser.add_argument(
        "--quantization",
        choices=["int8dynamic", "int8weightonly", "int4weightonly"],
        help="Apply quantization to the model before running it",
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
        "--torchinductor_post_grad_batch_fusion",
        action="store_true",
        help="Enable post grad horizontal batch fusion",
    )

    # inductor boolean configs
    inductor_config_dict = torch._inductor.config.shallow_copy_dict()
    for inductor_config_key in INDUCTOR_CONFIG_KEYS:
        inductor_config_key_arg = inductor_config_key.replace(".", "-")
        parser.add_argument(
            f"--pt2-{inductor_config_key_arg}",
            action="store_true",
            default=inductor_config_dict[inductor_config_key],
        )
        parser.add_argument(
            f"--no-pt2-{inductor_config_key_arg}",
            action="store_false",
            default=None,
        )
    args, extra_args = parser.parse_known_args(dynamo_args)
    # --torchdynamo inductor and --inductor are equivalent
    if args.torchdynamo == "inductor":
        args.inductor = True
    if args.inductor:
        args.torchdynamo = "inductor"
    return args, extra_args

def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    if args.inductor:
        optimize_ctx = functools.partial(
            torch.compile,
            backend="inductor",
            fullgraph=args.nopython,
            mode=args.inductor_compile_mode,
        )
        if args.dynamic_batch_only:
            args.dynamic_shapes = True
            torch._dynamo.config.assume_static_by_default = True
        if args.dynamic_shapes:
            if not args.dynamic_batch_only:
                torch._dynamo.config.assume_static_by_default = False
        if args.pt2_debug_log:
            import logging
            torch._logging.set_logs(dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG)
        # Load inductor configs
        if bool(args.torchinductor_post_grad_batch_fusion):
            torch._inductor.config.post_grad_fusion_options["batch_linear_post_grad"] = {}
        if compile_threads := args.torchinductor_compile_threads:
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
        # Deal with boolean inductor configs
        inductor_config_dict = torch._inductor.config.shallow_copy_dict()
        for inductor_config_key in INDUCTOR_CONFIG_KEYS:
            inductor_config_key_arg = inductor_config_key.replace(".", "_")
            if getattr(args, f"no_pt2_{inductor_config_key_arg}", None) == False:
                torch._inductor.config.__setattr__(inductor_config_key, False)
            else:
                torch._inductor.config.__setattr__(inductor_config_key,
                    getattr(args, f"pt2_{inductor_config_key_arg}", inductor_config_dict[inductor_config_key]))

        if args.quantization:
            import torchao
            from torchao.quantization import (
                change_linear_weights_to_int8_dqtensors,
                change_linear_weights_to_int8_woqtensors,
                change_linear_weights_to_int4_woqtensors
            )
            torch._dynamo.config.automatic_dynamic_shapes = False
            torch._dynamo.config.force_parameter_static_shapes = False
            torch._dynamo.config.cache_size_limit = 1000
            assert "cuda" in model.device
            module, example_inputs = model.get_module()
            if args.quantization == "int8dynamic":
                torch._inductor.config.force_fuse_int_mm_with_mul = True
                change_linear_weights_to_int8_dqtensors(module)
            elif args.quantization == "int8weightonly":
                torch._inductor.config.use_mixed_mm = True
                change_linear_weights_to_int8_woqtensors(module)
            elif args.quantization == "int4weightonly":
                change_linear_weights_to_int4_woqtensors(module)

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
            model.forward = optimize_ctx(model.forward)
        else:
            model.train = optimize_ctx(model.train)
    else:
        model.eval = optimize_ctx(model.eval)

    torch._dynamo.reset()
