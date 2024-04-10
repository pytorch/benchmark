import argparse
from typing import List

import torch

from torchbenchmark.util.backends import create_backend


def parse_torchscript_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # enable ofi by default
    parser.add_argument(
        "--no-ofi", action="store_true", help="disable optimize_for_inference"
    )
    parser.add_argument(
        "--fuser",
        type=str,
        default="",
        choices=["fuser0", "fuser1", "fuser2", "fuser3"],
        help="enable fuser",
    )
    args, unknown_args = parser.parse_known_args(args)
    return args, unknown_args


@create_backend
def torchscript(
    model: "torchbenchmark.util.model.BenchmarkModel", backend_args: List[str]
):
    model.jit = True
    backend_args, extra_args = parse_torchscript_args(backend_args)
    if model.device == "cpu" and backend_args.fuser == "fuser2":
        raise NotImplementedError(f"{backend_args.fuser} only works with GPU.")
    if model.test != "eval" and backend_args.fuser == "fuser3":
        raise NotImplementedError(f"{backend_args.fuser} only works with eval mode.")
    if backend_args.fuser:
        model.add_context(lambda: torch.jit.fuser(backend_args.fuser))

    def _torchscript():
        # customized jit callback function
        if hasattr(model, "jit_callback"):
            if backend_args.no_ofi:
                raise NotImplementedError(
                    "Customized jit callback doesn't support options."
                )
            model.jit_callback()
            return
        module, example_inputs = model.get_module()
        if hasattr(torch.jit, "_script_pdt"):
            module = torch.jit._script_pdt(
                module,
                example_inputs=[
                    example_inputs,
                ],
            )
        else:
            module = torch.jit.script(
                module,
                example_inputs=[
                    example_inputs,
                ],
            )
        if model.test == "eval" and not backend_args.no_ofi:
            if backend_args.fuser != "fuser3":
                module = torch.jit.optimize_for_inference(module)
            else:
                module = torch.jit.freeze(module)
        model.set_module(module)

    return _torchscript, extra_args
