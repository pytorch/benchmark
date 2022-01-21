import argparse
import torch
import torch.optim as optim
from torchbenchmark.util.model import BenchmarkModel
from typing import List, Tuple

def parse_args(model: BenchmarkModel, extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # by default, enable half precision for inference
    parser.add_argument("--eval-fp16", action='store_false', help="enable eval fp16")
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    parser.add_argument("--flops", action='store_true', help="enable flops counting")
    parser.add_argument("--train_cudagraph", action='store_true', help="enable CUDA Graph for train")
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.train_bs = model.train_bs
    args.eval_bs = model.eval_bs
    # only enable fp16 in GPU inference
    if args.device == "cpu":
        args.eval_fp16 = False
    return args

def apply_args(model: BenchmarkModel, args: argparse.Namespace):
    if args.flops:
        from fvcore.nn import FlopCountAnalysis
        model.train_flops = FlopCountAnalysis(model.model, tuple(model.example_inputs)).total()
        model.eval_flops = FlopCountAnalysis(model.eval_model, tuple(model.eval_example_inputs)).total()
    # apply eval_fp16
    if args.eval_fp16:
        model.eval_model, model.eval_example_inputs = enable_fp16(model.eval_model, model.eval_example_inputs)
    # apply fx2trt for eval
    if args.fx2trt:
        assert args.device == 'cuda', "fx2trt is only available with CUDA."
        assert not args.jit, "fx2trt with JIT is not available."
        model.eval_model = enable_fx2trt(args.eval_bs, args.eval_fp16, model.eval_model, model.eval_example_inputs)
    # apply cuda graph for train
    if args.train_cudagraph:
        enable_cudagraph(model, model.example_inputs)

def enable_cudagraph(model: BenchmarkModel, example_input: Tuple[torch.tensor]):
    # setup input and output
    optimizer = optim.Adam(model.model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model.model(*example_input)
            loss = loss_fn(y_pred, model.example_output)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)
    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_y_pred = model.model(*example_input)
        static_loss = loss_fn(static_y_pred, model.example_output)
        static_loss.backward()
        optimizer.step()
    model.g = g

def enable_fp16(model: torch.nn.Module, example_input: Tuple[torch.tensor]) -> Tuple[torch.nn.Module, Tuple[torch.tensor]]:
    return model.half(), (example_input[0].half(),)

def enable_fx2trt(max_batch_size: int, fp16: bool, model: torch.nn.Module, example_inputs: Tuple[torch.tensor]) -> torch.nn.Module:
    from torchbenchmark.util.fx2trt import lower_to_trt
    return lower_to_trt(module=model, input=example_inputs, \
                        max_batch_size=max_batch_size, fp16_mode=fp16)
