import torch
from typing import Tuple, Optional

def enable_fx2trt(max_batch_size: int, fp16: bool, model: torch.nn.Module, example_inputs: Tuple[torch.tensor],
                  is_hf_model: bool=False, hf_max_length: Optional[int]=None) -> torch.nn.Module:
    from fx2trt_oss.fx.lower import lower_to_trt
    from fx2trt_oss.fx.utils import LowerPrecision
    if fp16:
        precision = LowerPrecision.FP16
    else:
        precision = LowerPrecision.FP32
    # special enablement for huggingface models
    if is_hf_model:
        from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
        traced_model = hf_symbolic_trace(
            model,
            batch_size=max_batch_size,
            sequence_length=hf_max_length,
        )
        return lower_to_trt(
            traced_model,
            example_inputs,
            max_batch_size=max_batch_size,
            lower_precision=precision,
            explicit_batch_dimension=True,
            max_workspace_size=20 << 30,
        )
    return lower_to_trt(module=model, input=example_inputs, \
                        max_batch_size=max_batch_size, lower_precision=precision)
