from typing import List
import torch
import argparse

from torchbenchmark.util.backends import create_backend
from torchbenchmark.util.env_check import is_hf_model


def parse_torch_trt_args(backend_args: List[str]):
    """Parses CLI-provided backend arguments to extract Torch-TRT keywords

    Returns kwargs dictionary and remainder arguments which were unrecognized
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--truncate_long_and_double",
        default=None,
        action="store_true",
        help="Whether to automatically truncate long and double operations",
    )
    arg_parser.add_argument(
        "--workspace_size", type=int, help="Size of workspace allotted to TensorRT"
    )
    arg_parser.add_argument(
        "--min_block_size",
        type=int,
        help="Minimum number of operations in an accelerated TRT block",
    )
    arg_parser.add_argument(
        "--ir",
        type=str,
        help="Which internal representation to use: {'ts', 'dynamo_compile', 'fx_ts_compat', ...}",
    )
    args, unknown = arg_parser.parse_known_args(backend_args)

    # Remove unspecified arguments from the args dictionary
    # (Only pass through user-specified args)
    parsed_args = vars(args)
    for key in list(parsed_args.keys()):
        if parsed_args[key] is None:
            del parsed_args[key]

    return parsed_args, unknown


@create_backend
def fx2trt(model: "torchbenchmark.util.model.BenchmarkModel", backend_args: List[str]):
    FP16 = True if model.dargs.precision == "fp16" else False
    HF_MODEL = True if is_hf_model(model) else False
    assert (
        model.device == "cuda" and model.test == "eval"
    ), f"fx2trt only works on CUDA inference tests."

    def _fx2trt():
        from torch_tensorrt.fx import compile
        from torch_tensorrt.fx.utils import LowerPrecision

        module, example_inputs = model.get_module()
        precision = LowerPrecision.FP16 if FP16 else LowerPrecision.FP32

        if HF_MODEL:
            from transformers.utils.fx import symbolic_trace as hf_symbolic_trace

            traced_model = hf_symbolic_trace(
                module, batch_size=model.batch_size, sequence_lenghth=model.max_length
            )
            trt_model = compile(
                traced_model,
                example_inputs,
                max_batch_size=model.batch_size,
                lower_precision=precision,
                explicit_batch_dimension=True,
                max_workspace_size=20 << 30,
            )
        else:
            trt_model = compile(
                module=module,
                input=example_inputs,
                max_batch_size=model.batch_size,
                lower_precision=precision,
            )
        model.set_module(trt_model)

    return _fx2trt, backend_args


@create_backend
def torch_trt(
    model: "torchbenchmark.util.model.BenchmarkModel", backend_args: List[str]
):
    """Backend for Torch-TRT

    Can be directly invoked from the command line, for example via:
    python run.py resnet18 -d cuda -t eval --backend torch_trt --precision fp32 --truncate_long_and_double

    Options include:
        --truncate_long_and_double: Whether to automatically truncate long and double operations
        --min_block_size: Minimum number of operations in an accelerated TRT block
        --workspace_size: Size of workspace allotted to TensorRT
        --ir: Which internal representation to use: {"ts", "dynamo_compile", "fx_ts_compat", ...}
    """
    FP16 = True if model.dargs.precision == "fp16" else False
    assert (
        model.device == "cuda" and model.test == "eval"
    ), f"Torch-TRT only works on CUDA inference tests."

    # Extract relevant Torch-TRT arguments from the provided CLI arguments
    torch_trt_kwargs, backend_args = parse_torch_trt_args(backend_args)

    def _torch_trt():
        """Helper function for invoking Torch-TRT"""
        import torch_tensorrt

        module, example_inputs = model.get_module()
        torch_dtype_precision = torch.half if FP16 else torch.float32

        print(
            f"Compiling {model.name} with batch size {model.batch_size}, precision {model.dargs.precision}, "
            + f"and {'default' if 'ir' not in torch_trt_kwargs else torch_trt_kwargs['ir']} IR"
        )

        trt_module = torch_tensorrt.compile(
            module,
            inputs=example_inputs,
            enabled_precisions={torch_dtype_precision},
            **torch_trt_kwargs,
        )
        model.set_module(trt_module)

    return _torch_trt, backend_args
