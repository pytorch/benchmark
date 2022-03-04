import torch
from typing import Tuple, Optional

def enable_hf_fx2trt_no_lower(traced_model, input_all):
    import fx2trt_oss.tracer.acc_tracer.acc_tracer as acc_tracer

    from fx2trt_oss.fx.fx2trt import (
        TRTInterpreter,
        InputTensorSpec,
    )
    from fx2trt_oss.fx.tools.trt_splitter import (
        TRTSplitter,
        TRTSplitterSetting,
    )
    from fx2trt_oss.fx.trt_module import TRTModule
    def get_submod_inputs(mod, submod, inputs):
        acc_inputs = None

        def get_input(self, inputs):
            nonlocal acc_inputs
            acc_inputs = inputs

        handle = submod.register_forward_pre_hook(get_input)
        mod(*inputs)
        handle.remove()
        return acc_inputs

    # Trace with acc_tracer
    acc_model = acc_tracer.trace(traced_model, input_all)
    # Split out unsupported ops
    splitter_setting = TRTSplitterSetting()
    splitter_setting.use_implicit_batch_dim = False
    splitter = TRTSplitter(acc_model, input_all, settings=splitter_setting)
    splitter.node_support_preview()
    split_mod = splitter()

    for name, _ in split_mod.named_children():
        if "_run_on_acc" in name:
            print("replace submod=", name)
            submod = getattr(split_mod, name)
            # Get submodule inputs for fx2trt
            acc_inputs = get_submod_inputs(split_mod, submod, input_all)

            # fx2trt replacement
            interp = TRTInterpreter(
                submod,
                InputTensorSpec.from_tensors(acc_inputs),
                explicit_batch_dimension=True,
            )
            r = interp.run(
                max_workspace_size=1 << 30,
                strict_type_constraints=True,
            )
            trt_mod = TRTModule(*r)
            setattr(split_mod, name, trt_mod)
    return split_mod

def enable_fx2trt(max_batch_size: int, fp16: bool, model: torch.nn.Module, example_inputs: Tuple[torch.tensor],
                  is_hf_model: bool=False, hf_max_length: Optional[int]=None) -> torch.nn.Module:
    # special enablement for huggingface models
    if is_hf_model:
        from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
        traced_model = hf_symbolic_trace(
            model,
            batch_size=max_batch_size,
            sequence_length=hf_max_length,
        )
        # return lower_to_trt(
        #     traced_model,
        #     example_inputs,
        #     max_batch_size=max_batch_size,
        #     fp16_mode=True,
        #     explicit_batch_dimension=True,
        #     max_workspace_size=20 << 30,
        # )
        return enable_hf_fx2trt_no_lower(traced_model=traced_model, input_all=example_inputs)
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
    from fx2trt_oss.fx import LowerSetting
    from fx2trt_oss.fx.lower import Lowerer
    lower_setting = LowerSetting(
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size,
        explicit_batch_dimension=explicit_batch_dimension,
        fp16_mode=fp16_mode,
        enable_fuse=enable_fuse,
        verbose_log=verbose_log,
        timing_cache_prefix=timing_cache_prefix,
        save_timing_cache=save_timing_cache,
        cuda_graph_batch_size=cuda_graph_batch_size,
    )
    lowerer = Lowerer.create(lower_setting=lower_setting)
    return lowerer(module, input)
