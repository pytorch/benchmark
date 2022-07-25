"""
PyTorch benchmark env check utils.
This file may be loaded without torch packages installed, e.g., in OnDemand CI.
"""
import importlib
import copy
import os
from typing import List, Dict, Tuple, Optional

MAIN_RANDOM_SEED = 1337
# rounds for stableness tests
STABLENESS_CHECK_ROUNDS: int = 3
# rounds for correctness tests
CORRECTNESS_CHECK_ROUNDS: int = 2

def set_random_seed():
    import torch
    import random
    import numpy
    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)

def get_pkg_versions(packages: List[str]) -> Dict[str, str]:
    versions = {}
    for module in packages:
        module = importlib.import_module(module)
        versions[module] = module.__version__
    return versions

def has_native_amp() -> bool:
    import torch
    try:
        if getattr(torch.cuda.amp, 'autocast') is not None:
            return True
    except AttributeError:
        pass
    return False

def stableness_check(model: 'torchbenchmark.util.model.BenchmarkModel', cos_sim=True, deepcopy=True, rounds=STABLENESS_CHECK_ROUNDS) -> Tuple['torch.Tensor']:
    """Get the eager output. Run eager mode a couple of times to guarantee stableness.
       If the result is not stable, raise RuntimeError. """
    assert model.test=="eval", "We only support stableness check for inference."
    previous_result = None
    for _i in range(rounds):
        set_random_seed()
        # some models are stateful and will give different outputs
        # on the same input if called multiple times
        try:
            if deepcopy:
                copy_model = copy.deepcopy(model)
            else:
                copy_model = model
        except RuntimeError:
            # if the model is not copy-able, don't copy it
            copy_model = model
        if previous_result == None:
            previous_result = copy_model.invoke()
        else:
            cur_result = copy_model.invoke()
            if not same(previous_result, cur_result, cos_similarity=cos_sim):
                raise RuntimeError("Model returns unstable result. Please report a bug.")
            del cur_result
    return previous_result

def correctness_check(model: 'torchbenchmark.util.model.BenchmarkModel', cos_sim=True, deepcopy=True, rounds=CORRECTNESS_CHECK_ROUNDS, atol=1e-4, rtol=1e-4) -> bool:
    assert model.test=="eval", "We only support correctness check for inference."
    for _i in range(rounds):
        # some models are stateful and will give different outputs
        # on the same input if called multiple times
        set_random_seed()
        try:
            if deepcopy:
                copy_model = copy.deepcopy(model)
            else:
                copy_model = model
        except RuntimeError:
            # if the model is not copy-able, don't copy it
            copy_model = model
        cur_result = copy_model.invoke()

        if not same(model.eager_output, cur_result, cos_similarity=cos_sim, atol=atol, rtol=rtol):
            return False
        del cur_result
    return True

def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types

def is_numpy_int_type(value):
    import numpy as np
    return istype(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )


def is_numpy_float_type(value):
    import numpy as np
    return istype(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )

# copied from https://github.com/pytorch/torchdynamo/blob/main/torchdynamo/utils.py#L411
def same(a, b, cos_similarity=False, atol=1e-4, rtol=1e-4, equal_nan=False):
    """Check correctness to see if a and b match"""
    import torch
    import math
    if isinstance(a, (list, tuple, torch.nn.ParameterList, torch.Size)):
        assert isinstance(b, (list, tuple)), f"type mismatch {type(a)} {type(b)}"
        return len(a) == len(b) and all(
            same(ai, bi, cos_similarity, atol, rtol, equal_nan) for ai, bi in zip(a, b)
        )
    elif isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(
            b.keys()
        ), f"keys mismatch {set(a.keys())} == {set(b.keys())}"
        for k in a.keys():
            if not (same(a[k], b[k], cos_similarity, atol, rtol, equal_nan=equal_nan)):
                print("Accuracy failed for key name", k)
                return False
        return True
    elif isinstance(a, torch.Tensor):
        if a.is_sparse:
            assert b.is_sparse
            a = a.to_dense()
            b = b.to_dense()
        assert isinstance(b, torch.Tensor), f"type mismatch {type(a)} {type(b)}"
        if cos_similarity:
            # TRT will bring error loss larger than current threshold. Use cosine similarity as replacement
            a = a.flatten().to(torch.float32)
            b = b.flatten().to(torch.float32)
            res = torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-6)
            if res < 0.99:
                print(f"Similarity score={res.cpu().detach().item()}")
            return res >= 0.99
        else:
            return torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)
    elif isinstance(a, (str, int, type(None), bool, torch.device)):
        return a == b
    elif isinstance(a, float):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    elif is_numpy_int_type(a) or is_numpy_float_type(a):
        return (type(a) is type(b)) and (a == b)
    elif type(a).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        assert type(a) is type(b)
        return all(
            same(getattr(a, key), getattr(b, key), cos_similarity, atol, rtol, equal_nan)
            for key in a.__dict__.keys()
        )
    else:
        raise RuntimeError(f"unsupported type: {type(a).__name__}")
