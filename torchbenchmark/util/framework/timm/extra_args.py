import argparse
from torchbenchmark.util.model import BenchmarkModel
from typing import List, Tuple

from torchbenchmark.util.framework.vision.args import enable_fp16, enable_fx2trt, enable_tensortrt

def parse_args(model: BenchmarkModel, extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # by default, enable half precision for inference
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    parser.add_argument("--torch_tensorrt", action='store_true', help="enable torch_tensorrt")
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.train_bs = model.train_bs
    args.eval_bs = model.eval_bs
    args.eval_fp16 = False
    # sanity checks
    assert not (args.fx2trt and args.torch_tensorrt), "User cannot enable torch_tensorrt and fx2trt at the same time."
    return args

def apply_args(model: BenchmarkModel, args: argparse.Namespace):
    # apply fx2trt for eval
    if args.fx2trt:
        assert args.device == 'cuda', "fx2trt is only available with CUDA."
        assert not args.jit, "fx2trt with JIT is not available."
        model.eval_model = enable_fx2trt(args.eval_bs, args.eval_fp16, model.eval_model, model.eval_example_inputs)
    # apply torch_tensorrt for eval
    if args.torch_tensorrt:
        assert args.device == 'cuda', "torch_tensorrt is only available with CUDA."
        assert not args.jit, "torch_tensorrt with JIT is not available."
        model.eval_model = enable_tensortrt(model.eval_example_inputs, args.eval_fp16, model.eval_model)
