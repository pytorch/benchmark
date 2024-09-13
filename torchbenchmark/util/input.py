from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch.utils._pytree import tree_map


@dataclass
class ModelInputDescriptor:
    pass


def input_cast(cond, action, example_inputs):
    """Traverse the input batch pytree, and cast tensor with `action` if it satisfies `cond`."""
    if isinstance(example_inputs, torch.Tensor) and cond(example_inputs):
        return action(example_inputs)
    elif isinstance(example_inputs, (tuple, list, dict)):
        return tree_map(lambda x: input_cast(cond, action, x), example_inputs)
    elif (
        example_inputs is None
        or isinstance(example_inputs, str)
        or isinstance(example_inputs, int)
        or isinstance(example_inputs, float)
    ):
        # Do not touch primitive types
        return example_inputs
    elif isinstance(example_inputs, torch.Tensor):
        return example_inputs
    elif isinstance(example_inputs, torch.nn.Module):
        return example_inputs
    else:
        raise RuntimeError(f"Unsupported input type: {type(example_inputs)}")


def input_filter(cond, example_inputs):
    """Traverse the input batch pytree, and return the first element that satisfies cond."""
    if isinstance(example_inputs, dict):
        return input_filter(cond, example_inputs.items())
    elif isinstance(example_inputs, (tuple, list)):
        return next(
            item for item in example_inputs if input_filter(cond, item) is not None
        )
    elif cond(example_inputs):
        return example_inputs
    else:
        return None
