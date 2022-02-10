import torch
from fx2trt_oss.fx import LowerSetting
from fx2trt_oss.fx.lower import Lowerer

def enable_fx2trt(max_batch_size: int, fp16: bool, model: torch.nn.Module, example_inputs: Tuple[torch.tensor]) -> torch.nn.Module:
    from torchbenchmark.util.fx2trt import lower_to_trt
    return lower_to_trt(module=model, input=example_inputs, \
                        max_batch_size=max_batch_size, fp16_mode=fp16)

"""
The purpose of this example is to demostrate the onverall flow of lowering a PyTorch model
to TensorRT conveniently with lower.py.
"""
def lower_to_trt(
    module: torch.nn.Module,
    input,
    max_batch_size: int = 2048,
    max_workspace_size=1 << 25,
    explicit_batch_dimension=False,
    fp16_mode=True,
    enable_fuse=True,
    verbose_log=False,
    timing_cache_prefix="",
    save_timing_cache=False,
    cuda_graph_batch_size=-1,
) -> torch.nn.Module:
    """
    Takes in original module, input and lowering setting, run lowering workflow to turn module
    into lowered module, or so called TRTModule.

    Args:
    module: Original module for lowering.
    input: Input for module.
    max_batch_size: Maximum batch size (must be >= 1 to be set, 0 means not set)
    max_workspace_size: Maximum size of workspace given to TensorRT.
    explicit_batch_dimension: Use explicit batch dimension in TensorRT if set True, otherwise use implicit batch dimension.
    fp16_mode: fp16 config given to TRTModule.
    enable_fuse: Enable pass fusion during lowering if set to true. l=Lowering will try to find pattern defined
    in fx2trt_oss.fx.passes from original module, and replace with optimized pass before apply lowering.
    verbose_log: Enable verbose log for TensorRT if set True.
    timing_cache_prefix: Timing cache file name for timing cache used by fx2trt.
    save_timing_cache: Update timing cache with current timing cache data if set to True.
    cuda_graph_batch_size: Cuda graph batch size, default to be -1.

    Returns:
    A torch.nn.Module lowered by TensorRT.
    """
    lower_setting = LowerSetting(
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size,
        explicit_batch_dimension=explicit_batch_dimension,
        fp16_mode=fp16_mode,
        enable_fuse=enable_fuse,
        verbose_log=verbose_log,
        timing_cache_prefix=timing_cache_prefix,
        save_timing_cache=save_timing_cache,
    )
    lowerer = Lowerer.create(lower_setting=lower_setting)
    return lowerer(module, input)
