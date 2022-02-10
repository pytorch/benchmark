import torch
import argparse
from typing import List, Tuple

# Dispatch arguments based on model type
def parse_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # by default, enable half precision for inference
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.batch_size = model.batch_size
    # CUDA Graph is only supported for torchvision models
    args.cudagraph = None
    return args

def apply_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
   pass

def enable_fp16(model: torch.nn.Module, example_input: Tuple[torch.tensor]) -> Tuple[torch.nn.Module, Tuple[torch.tensor]]:
    return model.half(), (example_input[0].half(),)

