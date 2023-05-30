"""
PyTorch benchmark env check utils.
This file may be loaded without torch packages installed, e.g., in OnDemand CI.
"""
import copy
import importlib
import os
import argparse
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

def _check_correctness_p(
    model: 'torchbenchmark.util.model.BenchmarkModel',
    opt_args: argparse.Namespace,
) -> bool:
    "If correctness check should be enabled."
    # if the model doesn't support correctness check (like detectron2), skip it
    if hasattr(model, 'SKIP_CORRECTNESS_CHECK') and model.SKIP_CORRECTNESS_CHECK:
        return False
    # always check correctness with torchdynamo
    if model.dynamo:
        return True
    opt_args_dict = vars(opt_args)
    for k in opt_args_dict:
        if opt_args_dict[k]:
            return True
    return False

def save_deterministic_dict(name: str):
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

def load_deterministic_dict(determinism_dict: Dict[str, bool]):
    if "CUBLAS_WORKSPACE_CONFIG" in determinism_dict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = determinism_dict["CUBLAS_WORKSPACE_CONFIG"]
    elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]
    import torch
    torch.use_deterministic_algorithms(determinism_dict["torch.use_deterministic_algorithms"])
    torch.backends.cudnn.allow_tf32 = determinism_dict["torch.backends.cudnn.allow_tf32"]
    torch.backends.cudnn.benchmark = determinism_dict["torch.backends.cudnn.benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = determinism_dict["torch.backends.cuda.matmul.allow_tf32"]

def check_accuracy(tbmodel: 'torchbenchmark.util.model.BenchmarkModel') -> Optional[str]:
    import torch
    should_check_correctness = _check_correctness_p(tbmodel, tbmodel.opt_args)
    if not should_check_correctness:
        return "pass_due_to_skip"
    model, example_inputs = tbmodel.get_module()
    name = tbmodel.name
    current_device = tbmodel.device
    cosine = False
    is_training = tbmodel.test == "train"
    accuracy_status = "pass"
    # Collect the fp64 reference outputs to be used later for accuracy checking.
    fp64_outputs = None
    try:
        model_fp64, inputs_fp64 = cast_to_fp64(
            deepcopy_and_maybe_ddp(model),
            clone_inputs(example_inputs),
        )
        init_optimizer(name, current_device, model_fp64.parameters())
        fp64_outputs = run_n_iterations(model_fp64, inputs_fp64)
    except Exception:
        log.warning(
            "fp64 golden ref were not generated for %s. Setting accuracy check to cosine",
            tbmodel.name,
        )
        cosine = True
        fp64_outputs = None
    tolerance, cos_similarity = get_tolerance_and_cosine_flag(
            is_training, current_device, name
    )
     # Cast the model to float16/float32 as necessary
    model, example_inputs = maybe_cast(model, example_inputs)
    with pick_grad(name, self.args.training):
        # Get results of native pytorch
        reset_rng_state()
        try:
            model_copy = deepcopy_and_maybe_ddp(model)
            init_optimizer(name, current_device, model_copy.parameters())
            correct_result = self.run_n_iterations(
                model_copy, clone_inputs(example_inputs)
            )
        except Exception as e:
            accuracy_status = (
                "eager_1st_run_OOM"
                if isinstance(e, torch.cuda.OutOfMemoryError)
                else "eager_1st_run_fail"
            )
            log.exception(e)
            return accuracy_status

        # Rerun native pytorch
        reset_rng_state()
        try:
            model_copy = deepcopy_and_maybe_ddp(model)
            init_optimizer(name, current_device, model_copy.parameters())
            correct_rerun_result = run_n_iterations(
                model_copy, clone_inputs(example_inputs)
            )
        except Exception as e:
            accuracy_status = (
                "eager_2nd_run_OOM"
                if isinstance(e, torch.cuda.OutOfMemoryError)
                else "eager_2nd_run_fail"
            )
            return accuracy_status
        # Two eager runs should have exactly same result
        is_same = True
        try:
            if (
                name not in skip_accuracy_check_as_eager_non_deterministic
                and not same(
                    correct_result,
                    correct_rerun_result,
                    fp64_ref=None,
                    cos_similarity=False,
                    tol=0,
                    equal_nan=equal_nan,
                )
            ):
                is_same = False
        except Exception as e:
            # Sometimes torch.allclose may throw RuntimeError
            is_same = False

        if not is_same:
            accuracy_status = "eager_two_runs_differ"
            return accuracy_status

        correct_rerun_result = None

        # Run with Dynamo
        # Sometime CI fails with random triton compilation failure which will be skipped for now
        # TODO: revisit this after switching to new Triton runtime
        reset_rng_state()
        torch._dynamo.reset()
        try:
            model_copy = deepcopy_and_maybe_ddp(model)
            init_optimizer(name, current_device, model_copy.parameters())
            optimized_model_iter_fn = optimize_ctx(self.run_n_iterations)
            new_result = optimized_model_iter_fn(model_copy, example_inputs)
        except Exception as e:
            log.exception(e)
            if (
                isinstance(e, BackendCompilerFailed)
                and (
                    "Internal Triton PTX codegen error" in str(e)
                    or "cubin" in str(e)
                )
            ):
                accuracy_status = "pass_due_to_skip"
                return accuracy_status
            else:
                accuracy_status = (
                    "OOM"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "fail_to_run"
                )
                return accuracy_status

        if name in skip_accuracy_check_as_eager_non_deterministic:
            return "pass_due_to_skip"

        try:
            if not same(
                correct_result,
                new_result,
                fp64_outputs,
                equal_nan=equal_nan,
                cos_similarity=cos_similarity,
                tol=tolerance,
            ):
                is_same = False
        except Exception as e:
            # Sometimes torch.allclose may throw RuntimeError
            is_same = False

        if not is_same:
            accuracy_status = "fail_accuracy"
            return accuracy_status

        return accuracy_status

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
