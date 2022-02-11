import argparse
from typing import List
from torchbenchmark.util.backends.fx2trt import enable_fx2trt

# Dispatch arguments based on model type
def parse_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.batch_size = model.batch_size
    if not (model.device == "cuda" and model.test == "eval"):
        args.fx2trt = False
    if hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL:
        args.cudagraph = False
    return args

def apply_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    # apply fx2trt
    if args.fx2trt:
        assert not args.jit, "fx2trt with JIT is not available."
        module, exmaple_inputs = model.get_module()
        model.set_module(enable_fx2trt(args.batch_size, fp16=False, model=module, example_inputs=exmaple_inputs))

