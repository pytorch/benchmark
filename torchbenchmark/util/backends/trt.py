from typing import List
import torch

from torchbenchmark.util.backends import create_backend 
from torchbenchmark.util.env_check import is_hf_model

@create_backend
def fx2trt(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    FP16 = True if model.dargs.precision == "fp16" else False
    HF_MODEL = True if is_hf_model(model) else False
    def _fx2trt():
        from torch_tensorrt.fx import compile
        from torch_tensorrt.fx.utils import LowerPrecision
        module, example_inputs = model.get_module()
        precision = LowerPrecision.FP16 if FP16 else LowerPrecision.FP32

        if HF_MODEL:
            from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
            traced_model = hf_symbolic_trace(
                module,
                batch_size = model.batch_size,
                sequence_lenghth = model.max_length
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
            trt_model = compile(module=module,
                                input=example_inputs,
                                max_batch_size=model.batch_size,
                                lower_precision=precision)
        model.set_module(trt_model)
    return _fx2trt, backend_args

@create_backend
def torch_trt(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    FP16 = True if model.dargs.precision == "fp16" else False
    def _torch_trt():
        import torch_tensorrt
        module, example_inputs = model.get_module()
        if FP16:
            torchtrt_dtype = torch_tensorrt.dtype.half
            torch_dtype = torch.half
        else:
            torchtrt_dtype = torch_tensorrt.dtype.float
            torch_dtype = torch.float32
        trt_input = [torch_tensorrt.Input(shape=example_inputs[0].shape, dtype=torch_dtype)]
        trt_module = torch_tensorrt.compile(module, inputs=trt_input, enabled_precisions=torchtrt_dtype)
        model.set_module(trt_module)
    return _torch_trt, backend_args
