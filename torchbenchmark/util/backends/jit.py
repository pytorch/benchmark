import torch
from typing import Tuple

def enable_jit(model: torch.nn.Module, example_inputs: Tuple[torch.Tensor], test: str, optimize_for_inference: bool=True) -> torch.ScriptModule:
    if hasattr(torch.jit, '_script_pdt'):
        model = torch.jit._script_pdt(model, example_inputs=[example_inputs, ])
    else:
        model = torch.jit.script(model, example_inputs=[example_inputs, ])
    if test == "eval" and optimize_for_inference:
        model = torch.jit.optimize_for_inference(model)
    assert isinstance(model, torch.ScriptModule)
    return model
