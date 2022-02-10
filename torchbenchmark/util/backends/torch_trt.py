import torch
from typing import Tuple

def enable_torchtrt(eval_input: Tuple[torch.tensor], eval_fp16: bool, eval_model: torch.nn.Module) -> torch.nn.Module:
    import torch_tensorrt
    trt_input = [torch_tensorrt.Input(eval_input[0].shape)]
    if eval_fp16:
        enabled_precisions = torch_tensorrt.dtype.half
    else:
        enabled_precisions = torch_tensorrt.dtype.float
    return torch_tensorrt.compile(eval_model, inputs=trt_input, enabled_precisions=enabled_precisions)