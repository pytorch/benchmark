import torch
import argparse

from torchbenchmark.util.backends import create_backend
from typing import List

def parse_torchscript_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # enable ofi by default
    parser.add_argument("--no-ofi", action='store_true', help="disable optimize_for_inference")
    args = parser.parse_args(args)
    return args

@create_backend
def torchscript(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    model.jit = True
    args = parse_torchscript_args(backend_args)
    # customized jit callback function
    if hasattr(model, 'jit_callback'):
        if args.no_ofi:
            raise NotImplementedError("Customized jit callback doesn't support options.")
        model.jit_callback()
        return
    module, example_inputs = model.get_module()
    if hasattr(torch.jit, '_script_pdt'):
        module = torch.jit._script_pdt(module, example_inputs=[example_inputs, ])
    else:
        module = torch.jit.script(module, example_inputs=[example_inputs, ])
    if model.test == "eval" and not args.no_ofi:
        module = torch.jit.optimize_for_inference(module)
    model.set_module(module)
