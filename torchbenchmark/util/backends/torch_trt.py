import torch
from typing import Tuple

def enable_torchtrt(precision: str, model: torch.nn.Module, example_inputs: Tuple[torch.tensor]) -> torch.nn.Module:
    import torch_tensorrt
    if precision == "fp16":
        torchtrt_dtype = torch_tensorrt.dtype.half
        torch_dtype = torch.half
    elif precision == "fp32":
        torchtrt_dtype = torch_tensorrt.dtype.float
        torch_dtype = torch.float32
    else:
        raise NotImplementedError("torch_tensorrt only supports fp32 or fp16 precision")
    trt_input = [torch_tensorrt.Input(shape=example_inputs[0].shape, dtype=torch_dtype)]

    return torch_tensorrt.compile(model, inputs=trt_input, enabled_precisions=torchtrt_dtype)