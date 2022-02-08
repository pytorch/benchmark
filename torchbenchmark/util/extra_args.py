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
    # some models don't support train or eval tests, therefore they don't have these attributes
    args.train_bs = model.train_bs if hasattr(model, 'train_bs') else None
    args.eval_bs = model.eval_bs if hasattr(model, 'eval_bs') else None
    # CUDA Graph is only supported for torchvision models
    args.cudagraph = None
    return args

def apply_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
   pass

def enable_fp16(model: torch.nn.Module, example_input: Tuple[torch.tensor]) -> Tuple[torch.nn.Module, Tuple[torch.tensor]]:
    return model.half(), (example_input[0].half(),)

