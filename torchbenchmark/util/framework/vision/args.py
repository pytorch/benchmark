import torch
import argparse
from torchbenchmark.util.model import BenchmarkModel
from typing import List, Tuple

from torchbenchmark.util.backends.fuser import enable_fuser

def parse_args(model: BenchmarkModel, extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # by default, enable half precision for inference
    parser.add_argument("--eval-fp16", action='store_false', help="enable eval fp16")
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    parser.add_argument("--torch_tensorrt", action='store_true', help="enable torch_tensorrt")
    parser.add_argument("--flops", action='store_true', help="enable flops counting")
    parser.add_argument("--cudagraph", action='store_true', help="enable CUDA Graph. Currently only implemented for train.")
    parser.add_argument("--fuser", type=str, default="", help="enable nvfuser")
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.batch_size = model.batch_size
    # only enable fp16 in GPU inference
    if args.device == "cpu":
        args.eval_fp16 = False
        args.fuser = None
    # sanity checks
    assert not (args.fx2trt and args.torch_tensorrt), "User cannot enable torch_tensorrt and fx2trt at the same time."
    return args

def apply_args(model: BenchmarkModel, args: argparse.Namespace):
    if args.flops:
        from fvcore.nn import FlopCountAnalysis
        model.flops = FlopCountAnalysis(model.model, tuple(model.example_inputs)).total()
    # apply eval_fp16
    if args.eval_fp16:
        model.model, model.example_inputs = enable_fp16(model.model, model.example_inputs)
    # apply fx2trt for eval
    if args.fx2trt:
        assert args.device == 'cuda', "fx2trt is only available with CUDA."
        assert not args.jit, "fx2trt with JIT is not available."
        model.model = enable_fx2trt(args.batch_size, args.eval_fp16, model.model, model.example_inputs)
    # apply torch_tensorrt for eval
    if args.torch_tensorrt:
        assert args.device == 'cuda', "torch_tensorrt is only available with CUDA."
        model.model = enable_torchtrt(model.example_inputs, args.eval_fp16, model.model)
    # apply cuda graph for train
    if args.cudagraph:
        enable_cudagraph(model, model.example_inputs)
    # apply nvfuser
    if args.fuser:
        enable_fuser(args.fuser)

def enable_torchtrt(eval_input: Tuple[torch.tensor], eval_fp16: bool, model: torch.nn.Module) -> torch.nn.Module:
    import torch_tensorrt
    trt_input = [torch_tensorrt.Input(eval_input[0].shape)]
    if eval_fp16:
        enabled_precisions = torch_tensorrt.dtype.half
    else:
        enabled_precisions = torch_tensorrt.dtype.float
    return torch_tensorrt.compile(model, inputs=trt_input, enabled_precisions=enabled_precisions)

def enable_cudagraph(model: BenchmarkModel, example_inputs: Tuple[torch.tensor]):
    optimizer = model.optimizer
    loss_fn = model.loss_fn
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model.model(*example_inputs)
            loss = loss_fn(y_pred, model.example_outputs)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)
    # capture
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_y_pred = model.model(*example_inputs)
        static_loss = loss_fn(static_y_pred, model.example_outputs)
        static_loss.backward()
        optimizer.step()
    model.g = g

def enable_fp16(model: torch.nn.Module, example_input: Tuple[torch.tensor]) -> Tuple[torch.nn.Module, Tuple[torch.tensor]]:
    return model.half(), (example_input[0].half(),)

def enable_fx2trt(max_batch_size: int, fp16: bool, model: torch.nn.Module, example_inputs: Tuple[torch.tensor]) -> torch.nn.Module:
    from torchbenchmark.util.fx2trt import lower_to_trt
    return lower_to_trt(module=model, input=example_inputs, \
                        max_batch_size=max_batch_size, fp16_mode=fp16)
