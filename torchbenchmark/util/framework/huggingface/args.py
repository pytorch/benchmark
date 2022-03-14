import argparse
import torch
from torchbenchmark.util.model import BenchmarkModel
from typing import List, Dict, Tuple

def add_bool_arg(parser: argparse.ArgumentParser, name: str, default_value: bool=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default_value})

def parse_args(model: BenchmarkModel, extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # by default, enable half precision for inference
    add_bool_arg(parser, "eval_fp16", default_value=True)
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    # disable fp16 when device is CPU
    if args.device == "cpu":
        args.eval_fp16 = False
    return args

def apply_args(model: BenchmarkModel, args: argparse.Namespace):
    # apply eval_fp16
    if args.eval_fp16:
        model.model, model.example_inputs = enable_eval_fp16(model.model, model.example_inputs)

def enable_eval_fp16(model: torch.nn.Module, example_input: Dict[str, torch.tensor]) -> Tuple[torch.nn.Module, Dict[str, torch.tensor]]:
    return model.half(), {'input_ids': example_input['input_ids'].half()}