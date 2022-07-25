from torchdynamo.optimizations.backends import create_backend
import torch
import torch_blade
from contextlib import contextmanager
from torch_blade import optimize as blade_optimize
from typing import Tuple

@contextmanager
def opt_disc_config():
    torch_config = torch_blade.config.Config()
    try:
        with torch_config:
             yield
    finally:
        pass

@create_backend
def blade_optimize_dynamo(subgraph):
    with opt_disc_config(), torch.no_grad():
        optimized_model = blade_optimize(
            subgraph.model.eval(),
            allow_tracing=True,
            model_inputs=tuple(subgraph.example_inputs),
        )

    # with open(f'model.code.py', 'a') as writer:
    #     writer.write(str(optimized_model.code))
    # with open(f'model.graph.txt', 'a') as writer:
    #     writer.write(str(optimized_model.graph))

    return optimized_model

def blade_optimize_script(model: torch.nn.Module, example_inputs: Tuple[torch.Tensor], ):
    with opt_disc_config(), torch.no_grad():
        optimized_model = blade_optimize(
            model.eval(),
            allow_tracing=True,
            model_inputs=tuple(example_inputs),
        )
    return optimized_model