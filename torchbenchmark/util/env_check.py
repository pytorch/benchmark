"""
PyTorch benchmark env check utils.
This file may be loaded without torch packages installed, e.g., in OnDemand CI.
"""
import importlib
import copy
import warnings
from typing import List, Dict, Tuple, Optional

import torch
from torch._dynamo.utils import same, clone_inputs

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
    old_test = model.test
    model.test = "eval"
    opt_saved = None
    if hasattr(model, "opt"):
        opt_saved = model.opt
        model.opt = None
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
    model.test = old_test
    if opt_saved:
        model.opt = opt_saved
    return previous_result

def correctness_check(model: 'torchbenchmark.util.model.BenchmarkModel', cos_sim=True, deepcopy=True, rounds=CORRECTNESS_CHECK_ROUNDS, tol=1e-4) -> bool:
    old_test = model.test
    model.test = "eval"
    opt_saved = None
    if hasattr(model, "opt"):
        opt_saved = model.opt
        model.opt = None

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

        equal_nan = hasattr(model, "EQUAL_NAN") and model.EQUAL_NAN
        if not same(model.eager_output, cur_result, fp64_ref=model.eager_output_fp64, cos_similarity=cos_sim, tol=tol, equal_nan=equal_nan):
            return False

        del cur_result
    model.test = old_test
    if opt_saved:
        model.opt = opt_saved

    if model.test == "train":
        if not hasattr(model, "model") or not hasattr(model.model, "named_parameters"):
            warnings.warn(UserWarning("model doesn't have model or model.named_parameters. Skipping train correctness check."))
        if not hasattr(model, "eager_model_after_one_train_iteration"):
            warnings.warn(UserWarning("model doesn't have eager_model_after_one_train_iteration. Skipping train correctness check."))
        model.invoke()
        eager_named_params = dict(model.eager_model_after_one_train_iteration.named_parameters())
        eager_fp64_named_params = dict(model.eager_model_fp64.named_parameters()) if model.eager_model_fp64 is not None else {}

        for name, param in model.model.named_parameters():
            if not param.requires_grad:
                continue

            if name in eager_named_params:
                fp64_grad = getattr(eager_fp64_named_params.get(name, None), "grad", None)
                # backward typically requires higher error margin.
                # 400 times bigger may sound too big to be useful but still better than not checking at all.
                if not same(eager_named_params[name].grad, param.grad, fp64_ref=fp64_grad, cos_similarity=cos_sim, tol=tol):
                    import torch
                    if not isinstance(param.grad, torch.Tensor):
                        print(f"model with dynamo does not have grad of param {name}")
                    else:
                        print(f"grad of param {name} after running with dynamo doesn't have gradient matching with eager mode")
                    return False
                break
            else:
                print(f"param {name} in model with dynamo not found in the eager model")
                return False

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

def is_numpy_ndarray(value):
    import numpy as np
    return istype(
        value,
        (np.ndarray, ),
    )
