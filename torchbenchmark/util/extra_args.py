import torch
import argparse
from torchbenchmark.util.model import BenchmarkModel
from typing import List, Tuple

def parse_args(model: BenchmarkModel, extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # by default, enable half precision for inference
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.train_bs = model.train_bs
    args.eval_bs = model.eval_bs
    return args

def apply_args(model: BenchmarkModel, args: argparse.Namespace):
   pass

def enable_torchtrt(eval_input: Tuple[torch.tensor], eval_fp16: bool, eval_model: torch.nn.Module) -> torch.nn.Module:
    import torch_tensorrt
    trt_input = [torch_tensorrt.Input(eval_input[0].shape)]
    if eval_fp16:
        enabled_precisions = torch_tensorrt.dtype.half
    else:
        enabled_precisions = torch_tensorrt.dtype.float
    return torch_tensorrt.compile(eval_model, inputs=trt_input, enabled_precisions=enabled_precisions)

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
