"""
PyTorch benchmark env check utils.
This file may be loaded without torch packages installed, e.g., in OnDemand CI.
"""
import copy
import importlib
import os
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

MAIN_RANDOM_SEED = 1337
# rounds for stableness tests
STABLENESS_CHECK_ROUNDS: int = 3
# rounds for correctness tests
CORRECTNESS_CHECK_ROUNDS: int = 2
# Use the list from
# https://github.com/pytorch/pytorch/blob/6c7410ddc350fea625e47744da9d6be7ec74b628/benchmarks/dynamo/common.py#L2247
UNSUPPORTED_USE_DETERMINISTIC_ALGORITHMS = [
    "alexnet",
    "Background_Matting",
    "pytorch_CycleGAN_and_pix2pix",
    "pytorch_unet",
    "Super_SloMo",
    "vgg16",
]
# Use the list from
# https://github.com/pytorch/pytorch/blob/6c7410ddc350fea625e47744da9d6be7ec74b628/benchmarks/dynamo/torchbench.py#L382
USE_GRAD_IN_INFERENCE = [
    "maml"
]
HAS_NUMPY = True

log = logging.getLogger(__name__)

def pick_grad(name: str, is_training: bool):
    import torch
    if is_training or name in USE_GRAD_IN_INFERENCE:
        return torch.enable_grad()
    else:
        return torch.no_grad()

def set_random_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""
    import torch
    import random
    import numpy

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = MAIN_RANDOM_SEED
        import torch.cuda

        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)
    torch.manual_seed = deterministic_torch_manual_seed

def save_deterministic_mode(name: str) -> Dict[str, Any]:
    determinism_dict = {}
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        determinism_dict["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    determinism_dict["torch.use_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    determinism_dict["torch.backends.cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
    determinism_dict["torch.backends.cudnn.benchmark"] = torch.backends.cudnn.benchmark
    determinism_dict["torch.backends.cuda.matmul.allow_tf32"] = torch.backends.cuda.matmul.allow_tf32

    if not name in UNSUPPORTED_USE_DETERMINISTIC_ALGORITHMS:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return determinism_dict

def load_deterministic_mode(determinism_dict: Dict[str, Any]) -> None:
    import torch
    torch.use_deterministic_algorithms(determinism_dict["torch.use_deterministic_algorithms"])
    torch.backends.cudnn.allow_tf32 = determinism_dict["torch.backends.cudnn.allow_tf32"]
    torch.backends.cudnn.benchmark = determinism_dict["torch.backends.cudnn.benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = determinism_dict["torch.backends.cuda.matmul.allow_tf32"]
    if "CUBLAS_WORKSPACE_CONFIG" in determinism_dict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = determinism_dict["CUBLAS_WORKSPACE_CONFIG"]
    elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]

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

def is_timm_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'TIMM_MODEL') and model.TIMM_MODEL

def is_torchvision_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL

def is_hf_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'HF_MODEL') and model.HF_MODEL

def is_fambench_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'FAMBENCH_MODEL') and model.FAMBENCH_MODEL

def is_staged_train_test(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'forward') and hasattr(model, 'backward') and hasattr(model, 'optimizer_step')

def _get_forward_result(model: 'torchbenchmark.util.model.BenchmarkModel', is_training: bool):
    if is_training:
        module, example_inputs = model.get_module()
        return module(*example_inputs)
    return model.invoke()

def stableness_check(model: 'torchbenchmark.util.model.BenchmarkModel', cos_sim=True, deepcopy=True, rounds=STABLENESS_CHECK_ROUNDS) -> Tuple['torch.Tensor']:
    """Get the eager output. Run eager mode a couple of times to guarantee stableness.
       If the result is not stable, raise RuntimeError. """
    opt_saved = None
    if hasattr(model, "opt"):
        opt_saved = model.opt
        model.opt = None

    previous_result = None
    is_training = model.test == "train"
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
            previous_result = _get_forward_result(copy_model, is_training)
        else:
            cur_result = _get_forward_result(copy_model, is_training)
            if not same(previous_result, cur_result, cos_similarity=cos_sim):
                raise RuntimeError("Model returns unstable result. Please report a bug.")
            del cur_result
    if opt_saved:
        model.opt = opt_saved
    return previous_result

def correctness_check(model: 'torchbenchmark.util.model.BenchmarkModel', cos_sim=True, deepcopy=True, rounds=CORRECTNESS_CHECK_ROUNDS, tol=1e-4) -> bool:
    import torch

    opt_saved = None
    if hasattr(model, "opt"):
        opt_saved = model.opt
        model.opt = None

    is_training = model.test == "train"
    # It looks we don't run backward here and also dynamo may have
    # an issue with memory usage: https://fburl.com/workplace/cgxzsdhz
    # with pick_grad(model.name, is_training):
    with torch.no_grad():
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
            cur_result = _get_forward_result(copy_model, is_training)

            equal_nan = hasattr(model, "EQUAL_NAN") and model.EQUAL_NAN
            if not same(model.eager_output, cur_result, cos_similarity=cos_sim, tol=tol, equal_nan=equal_nan):
                # Restore the original model test if eval correctness doesn't pass
                model.opt = opt_saved if opt_saved else model.opt
                return False

            del cur_result

    model.opt = opt_saved if opt_saved else model.opt

    if is_training:
        if not hasattr(model, "model") or not hasattr(model.model, "named_parameters"):
            warnings.warn(UserWarning("model doesn't have model or model.named_parameters. Skipping train correctness check."))
            return True
        if not hasattr(model, "eager_model_after_one_train_iteration"):
            warnings.warn(UserWarning("model doesn't have eager_model_after_one_train_iteration. Skipping train correctness check."))
            return True
        model.invoke()
        for name, param in model.model.named_parameters():
            if not param.requires_grad:
                continue
            found = False
            for name_ref, param_ref in model.eager_model_after_one_train_iteration.named_parameters():
                if name_ref == name:
                    found = True
                    # backward typically requires higher error margin.
                    # 400 times bigger may sound too big to be useful but still better than not checking at all.
                    if not same(param_ref.grad, param.grad, cos_similarity=cos_sim, tol=tol*40):
                        import torch
                        if not isinstance(param.grad, torch.Tensor):
                            print(f"model with dynamo does not have grad of param {name}")
                        else:
                            print(f"grad of param {name} after running with dynamo doesn't have gradient matching with eager mode")
                            print(f"grad of param:\n{param.grad}\neager grad:\n{param_ref.grad}")
                        return False
                    break
            if not found:
                print(f"param {name} in model with dynamo not found in the eager model")
                return False
    return True

def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types

def is_numpy_int_type(value):
    if HAS_NUMPY:
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
    else:
        return False


def is_numpy_float_type(value):
    if HAS_NUMPY:
        import numpy as np
        return istype(
            value,
            (
                np.float16,
                np.float32,
                np.float64,
            ),
        )
    else:
        return False


def is_numpy_ndarray(value):
    if HAS_NUMPY:
        import numpy as np
        return istype(value, np.ndarray)
    else:
        return False


def rmse(ref, res):
    """
    Calculate root mean squared error
    """
    import torch
    return torch.sqrt(torch.mean(torch.square(ref - res)))

def same(
    ref,
    res,
    fp64_ref=None,
    cos_similarity=False,
    tol=1e-4,
    equal_nan=False,
    exact_dtype=True,
    relax_numpy_equality=False,
    ignore_non_fp=False,
    log_error=log.error,
):
    """Check correctness to see if ref and res match"""
    import math
    import torch
    if fp64_ref is None:
        fp64_ref = ref
    if isinstance(ref, (list, tuple, torch.nn.ParameterList, torch.Size)):
        assert isinstance(res, (list, tuple)), f"type mismatch {type(ref)} {type(res)}"
        return len(ref) == len(res) and all(
            same(
                ai,
                bi,
                fp64_refi,
                cos_similarity,
                tol,
                equal_nan,
                exact_dtype,
                relax_numpy_equality,
                ignore_non_fp,
                log_error=log_error,
            )
            for ai, bi, fp64_refi in zip(ref, res, fp64_ref)
        )
    elif isinstance(ref, dict):
        assert isinstance(res, dict)
        assert set(ref.keys()) == set(
            res.keys()
        ), f"keys mismatch {set(ref.keys())} == {set(res.keys())}"
        for k in sorted(ref.keys()):
            if not (
                same(
                    ref[k],
                    res[k],
                    fp64_ref[k],
                    cos_similarity=cos_similarity,
                    tol=tol,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                    relax_numpy_equality=relax_numpy_equality,
                    ignore_non_fp=ignore_non_fp,
                    log_error=log_error,
                )
            ):
                log_error("Accuracy failed for key name %s", k)
                return False
        return True
    elif isinstance(ref, torch.Tensor):
        assert not isinstance(ref, torch._subclasses.FakeTensor)
        assert not isinstance(res, torch._subclasses.FakeTensor)

        if ref.is_sparse:
            assert res.is_sparse
            ref = ref.to_dense()
            res = res.to_dense()
        assert isinstance(res, torch.Tensor), f"type mismatch {type(ref)} {type(res)}"
        if exact_dtype:
            if ref.dtype != res.dtype:
                log_error("dtype mismatch %s, %s", ref.dtype, res.dtype)
                return False
            if ref.dtype == torch.bool:
                if ignore_non_fp:
                    return True
                # triton stores bool as int8, so add this for more accurate checking
                r = torch.allclose(
                    ref.to(dtype=torch.uint8),
                    res.to(dtype=torch.uint8),
                    atol=tol,
                    rtol=tol,
                    equal_nan=equal_nan,
                )
                if not r:
                    log_error("Accuracy failed: uint8 tensor did not match")
                return r
        if cos_similarity:
            ref = ref.flatten().to(torch.float32)
            res = res.flatten().to(torch.float32)
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=True):
                # early exit that handles zero/nan better
                # cosine_similarity(zeros(10), zeros(10), dim=0) is 0
                return True
            score = torch.nn.functional.cosine_similarity(ref, res, dim=0, eps=1e-6)
            if score < 0.99:
                log.warning("Similarity score=%s", score.cpu().detach().item())
            return score >= 0.99
        else:
            if not exact_dtype:
                ref = ref.to(res.dtype)

            # First try usual allclose
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=equal_nan):
                return True

            # Check error from fp64 version
            if fp64_ref.dtype == torch.float64:
                ref_error = rmse(fp64_ref, ref).item()
                res_error = rmse(fp64_ref, res).item()
                multiplier = 2.0

                if (
                    fp64_ref.numel() < 1000
                    or (ref.ndim == 4 and ref.shape[-1] == ref.shape[-2] == 1)
                    # large tol means a benchmark has been specified as REQUIRE_HIGHER_TOLERANCE
                    or tol >= 2 * 1e-2
                ):
                    # In the presence of noise, noise might dominate our error
                    # metric for smaller tensors.
                    # Similary, for 1x1 kernels, there seems to be high noise with amp.
                    multiplier = 3.0

                passes_test = res_error <= (multiplier * ref_error + tol / 10.0)
                if not passes_test:
                    log_error(
                        "RMSE (res-fp64): %.5f, (ref-fp64): %.5f and shape=%s",
                        res_error,
                        ref_error,
                        res.size(),
                    )
                    # import pdb; pdb.set_trace()
                return passes_test

            if ignore_non_fp:
                return True

            log_error("Accuracy failed: allclose not within tol=%s", tol)
            return False
    elif isinstance(ref, (str, int, type(None), bool, torch.device)):
        if ignore_non_fp:
            return True
        r = ref == res
        if not r:
            log_error("Accuracy failed (%s): %s != %s", type(ref), ref, res)
        return r
    elif isinstance(ref, float):
        r = math.isclose(ref, res, rel_tol=tol, abs_tol=tol)
        if not r:
            log_error(
                "Accuracy failed (float): %s != %s (within tol=%s)", ref, res, tol
            )
        return r
    elif is_numpy_int_type(ref) or is_numpy_float_type(ref):
        if relax_numpy_equality and not (
            is_numpy_int_type(res) or is_numpy_float_type(res)
        ):
            ref = ref.item()
        r = (type(ref) is type(res)) and (ref == res)
        if not r:
            log_error("Accuracy failed (numpy): %s != %s", ref, res)
        return r
    elif is_numpy_ndarray(ref):
        return (type(ref) is type(res)) and (ref == res).all()
    elif type(ref).__name__ in (
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
        assert type(ref) is type(res)
        return all(
            same(
                getattr(ref, key),
                getattr(res, key),
                getattr(fp64_ref, key),
                cos_similarity=cos_similarity,
                tol=tol,
                equal_nan=equal_nan,
                exact_dtype=exact_dtype,
                relax_numpy_equality=relax_numpy_equality,
                ignore_non_fp=ignore_non_fp,
                log_error=log_error,
            )
            for key in ref.__dict__.keys()
        )
    else:
        raise RuntimeError(f"unsupported type: {type(ref).__name__}")
