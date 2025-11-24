"""
PyTorch benchmark env check utils.
This file may be loaded without torch packages installed, e.g., in OnDemand CI.
"""

import argparse
import copy
import logging
import os
import shutil
from collections.abc import Mapping
from contextlib import contextmanager, ExitStack
from typing import Any, Dict, List, Optional


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
    "sam",
    "sam_fast",
    "Super_SloMo",
    "vgg16",
    "mtml_ctr_instagram_model",
    "resnet50_quantized_qat",
    "mobilenet_v2_quantized_qat",
]
CI_SKIP_OPTIMIZER = {
    # TIMM
    "convmixer_768_32",  # accuracy
    "hrnet_w18",  # Stack issue in fx
    # TorchBench
    "dlrm",  # symbolic shapes error
    # HF
    "pnasnet5large",  # Stack issue in fx
    "MobileBertForMaskedLM",  # Stack issue in fx
    "MobileBertForQuestionAnswering",  # Stack issue in fx
    "PegasusForConditionalGeneration",  # OOM
}


# Need lower tolerance on GPU. GPU kernels have non deterministic kernels for these models.
REQUIRE_HIGHER_TOLERANCE = {
    "alexnet",
    "densenet121",
    "hf_Albert",
    "vgg16",
    "mobilenet_v3_large",
    "nvidia_deeprecommender",
    "timm_efficientdet",
}
# These models need >1e-3 tolerance
REQUIRE_EVEN_HIGHER_TOLERANCE = {
    "soft_actor_critic",
    "tacotron2",
}
REQUIRE_HIGHER_FP16_TOLERANCE = {
    "drq",
}
REQUIRE_HIGHER_BF16_TOLERANCE = {
    "doctr_reco_predictor",
    "drq",
    "hf_Whisper",
}
REQUIRE_COSINE_TOLERACE = {
    # Just keeping it here even though its empty, if we need this in future.
}
SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS = {
    # Models that deterministic algorithms can not be turned on for eager mode.
    "Background_Matting",
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_maskrcnn",
    "stable_diffusion_unet",
}
# Use the list from
# https://github.com/pytorch/pytorch/blob/6c7410ddc350fea625e47744da9d6be7ec74b628/benchmarks/dynamo/torchbench.py#L382
USE_GRAD_IN_INFERENCE = ["maml"]
HAS_NUMPY = True

log = logging.getLogger(__name__)


class DummyGradScaler:
    def scale(self, loss):
        return loss


@contextmanager
def nested(*contexts):
    """
    Chain and apply a list of contexts
    """
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield contexts


@contextmanager
def fresh_inductor_cache(parallel_compile=False):
    INDUCTOR_DIR = f"/tmp/torchinductor_{os.environ['USER']}"
    if os.path.exists(INDUCTOR_DIR):
        shutil.rmtree(INDUCTOR_DIR)
    if parallel_compile:
        old_parallel_compile_threads = os.environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS", None
        )
        cpu_count: Optional[int] = os.cpu_count()
        if cpu_count is not None and cpu_count > 1:
            cpu_count = min(32, cpu_count)
            log.warning(f"Set env var TORCHINDUCTOR_COMPILE_THREADS to {cpu_count}")
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(cpu_count)
    yield
    # clean up parallel compile directory and env
    if parallel_compile and "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        if old_parallel_compile_threads:
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = old_parallel_compile_threads
        else:
            del os.environ["TORCHINDUCTOR_COMPILE_THREADS"]
    if os.path.exists(INDUCTOR_DIR):
        shutil.rmtree(INDUCTOR_DIR)


@contextmanager
def fresh_triton_cache():
    """
    Run with a fresh triton cache.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        old = os.environ.get("TRITON_CACHE_DIR", None)
        os.environ["TRITON_CACHE_DIR"] = tmpdir
        old_cache_manager = os.environ.get("TRITON_CACHE_MANAGER", None)
        os.environ.pop("TRITON_CACHE_MANAGER", None)
        yield
        if old:
            os.environ["TRITON_CACHE_DIR"] = old
        else:
            del os.environ["TRITON_CACHE_DIR"]
        if old_cache_manager:
            os.environ["TRITON_CACHE_MANAGER"] = old_cache_manager


def pick_grad(name: str, is_training: bool):
    import torch

    if is_training or name in USE_GRAD_IN_INFERENCE:
        return torch.enable_grad()
    else:
        return torch.no_grad()


def set_random_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""
    import random

    import numpy
    import torch

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = MAIN_RANDOM_SEED
        import torch.cuda

        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed)

        import torch.xpu

        if not torch.xpu._is_in_bad_fork():
            torch.xpu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)
    torch.manual_seed = deterministic_torch_manual_seed


def get_pkg_versions(packages: List[str]) -> Dict[str, str]:
    import subprocess
    import sys

    versions = {}
    for module in packages:
        cmd = [sys.executable, "-c", f"import {module}; print({module}.__version__)"]
        version = subprocess.check_output(cmd).decode().strip()
        versions[module] = version
    return versions


def has_native_amp() -> bool:
    import torch

    try:
        if getattr(torch.cuda.amp, "autocast") is not None:
            return True
    except AttributeError:
        pass
    return False


def is_timm_model(model: "torchbenchmark.util.model.BenchmarkModel") -> bool:
    return hasattr(model, "TIMM_MODEL") and model.TIMM_MODEL


def is_torchvision_model(model: "torchbenchmark.util.model.BenchmarkModel") -> bool:
    return hasattr(model, "TORCHVISION_MODEL") and model.TORCHVISION_MODEL


def is_hf_model(model: "torchbenchmark.util.model.BenchmarkModel") -> bool:
    return hasattr(model, "HF_MODEL") and model.HF_MODEL


def is_fambench_model(model: "torchbenchmark.util.model.BenchmarkModel") -> bool:
    return hasattr(model, "FAMBENCH_MODEL") and model.FAMBENCH_MODEL


def is_staged_train_test(model: "torchbenchmark.util.model.BenchmarkModel") -> bool:
    return (
        hasattr(model, "forward")
        and hasattr(model, "backward")
        and hasattr(model, "optimizer_step")
    )


def save_deterministic_dict(name: str):
    determinism_dict = {}
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        determinism_dict["CUBLAS_WORKSPACE_CONFIG"] = os.environ[
            "CUBLAS_WORKSPACE_CONFIG"
        ]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    determinism_dict["torch.use_deterministic_algorithms"] = (
        torch.are_deterministic_algorithms_enabled()
    )
    determinism_dict["torch.backends.cudnn.allow_tf32"] = (
        torch.backends.cudnn.allow_tf32
    )
    determinism_dict["torch.backends.cudnn.benchmark"] = torch.backends.cudnn.benchmark
    determinism_dict["torch.backends.cuda.matmul.allow_tf32"] = (
        torch.backends.cuda.matmul.allow_tf32
    )

    if not name in UNSUPPORTED_USE_DETERMINISTIC_ALGORITHMS:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return determinism_dict


def load_deterministic_dict(determinism_dict: Dict[str, bool]):
    if "CUBLAS_WORKSPACE_CONFIG" in determinism_dict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = determinism_dict[
            "CUBLAS_WORKSPACE_CONFIG"
        ]
    elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]
    import torch

    torch.use_deterministic_algorithms(
        determinism_dict["torch.use_deterministic_algorithms"]
    )
    torch.backends.cudnn.allow_tf32 = determinism_dict[
        "torch.backends.cudnn.allow_tf32"
    ]
    torch.backends.cudnn.benchmark = determinism_dict["torch.backends.cudnn.benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = determinism_dict[
        "torch.backends.cuda.matmul.allow_tf32"
    ]


def cast_to(dtype, model, inputs):
    import torch
    from torch.utils._pytree import tree_map

    # cast model and inputs to fp16
    if dtype == torch.float16:
        model = model.half()
    else:
        model = model.to(dtype)

    inputs = tree_map(
        lambda x: (
            x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
        ),
        inputs,
    )
    return model, inputs


def collect_results(model, prediction, loss, example_inputs):
    import torch

    results = []
    results.append(prediction)
    results.append(loss)
    # if isinstance(loss, torch.Tensor) and loss.item() > 1:
    #     log.warning(
    #         f"High loss value alert - {loss:.2f}. Can result in unstable gradients."
    #     )

    grads = dict()
    params = dict()
    for name, param in model.named_parameters():
        # if isinstance(model, eval_frame.OptimizedModule):
        #     name = remove_optimized_module_prefix(name)
        param_copy = param
        grad = param.grad
        # Treat None and zero grad as same
        if param.grad is None:
            grad = torch.zeros_like(param)
        grads[name + ".grad"] = grad
        params[name] = param_copy
    results.append(grads)
    results.append(params)
    buffers = dict()
    for name, buffer in model.named_buffers():
        # if isinstance(model, eval_frame.OptimizedModule):
        #     name = remove_optimized_module_prefix(name)
        buffers[name] = buffer
    results.append(buffers)
    for example in example_inputs:
        if isinstance(example, (tuple, list)):
            for inp in example:
                if isinstance(inp, torch.Tensor):
                    results.append(inp.grad)
        else:
            if isinstance(example, torch.Tensor):
                results.append(example.grad)
    return results


def clone_input(x, *, dtype=None):
    """copy while preserving strides"""
    import torch

    # TODO: this is questionable
    if isinstance(x, torch._subclasses.FakeTensor):
        # this func fails on fake tensors in __torch_dispatch__
        return x

    def torch_clone(x):
        y = torch.clone(x)
        if x.is_leaf:
            y.requires_grad_(x.requires_grad)
        if x.is_leaf and x.grad is not None:
            y.grad = clone_input(x.grad, dtype=dtype)
        if hasattr(x, "_dynamo_dynamic_indices"):
            y._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()
        return y

    with torch.no_grad():
        if x.device.type == "xla":
            # Access data_ptr() for a xla tensor will cause crash
            return torch_clone(x)

        needed_size = sum(
            (shape - 1) * stride for shape, stride in zip(x.size(), x.stride())
        )
        if x.is_quantized:
            result = torch.empty_quantized((needed_size + 32,), x)
        else:
            result = torch.empty(
                needed_size + 32, dtype=dtype or x.dtype, device=x.device
            )
        cache_line_offset = (
            (x.data_ptr() - result.data_ptr()) % 32
        ) // x.element_size()
        result.as_strided_(x.size(), x.stride(), cache_line_offset)
        try:
            result.copy_(x.clone())
            if x.is_leaf:
                result.requires_grad_(x.requires_grad)
            if x.is_leaf and x.grad is not None:
                result.grad = clone_input(x.grad, dtype=dtype)
        except RuntimeError:
            # RuntimeError: unsupported operation: more than one element of the written-to
            # tensor refers to a single memory location. Please clone() the tensor before
            # performing the operation.
            return torch_clone(x)
        if hasattr(x, "_dynamo_dynamic_indices"):
            result._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()
        return result


def clone_inputs(example_inputs):
    import torch

    if isinstance(example_inputs, Mapping):
        res = dict(example_inputs)
        for key, value in res.items():
            assert isinstance(value, torch.Tensor)
            res[key] = clone_input(value)
        return res

    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = clone_input(res[i])
    return res


def init_optimizer(name, device, params, is_training):
    import torch

    if device == "cuda" and is_training and name not in CI_SKIP_OPTIMIZER:
        optimizer = torch.optim.SGD(params, lr=0.01)
    else:
        optimizer = None
    return optimizer


def reduce_to_scalar_loss(out):
    """Reduce the output of a model to get scalar loss"""
    import torch

    if isinstance(out, torch.Tensor):
        # Mean does not work on integer tensors
        return out.sum() / out.numel()
    elif isinstance(out, (list, tuple)):
        return sum([reduce_to_scalar_loss(x) for x in out]) / len(out)
    elif type(out).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
    ):
        return reduce_to_scalar_loss(out.logits)
    elif type(out).__name__ == "SquashedNormal":
        return out.mean.sum()
    elif isinstance(out, dict):
        return sum([reduce_to_scalar_loss(value) for value in out.values()]) / len(
            out.keys()
        )
    elif out == None:
        return 0.0
    raise NotImplementedError("Don't know how to reduce", type(out))


def compute_loss(pred):
    return reduce_to_scalar_loss(pred)


def optimizer_zero_grad(optimizer, mod):
    if optimizer is not None:
        optimizer.zero_grad(True)
    else:
        mod.zero_grad(True)


def optimizer_step(optimizer):
    if optimizer is not None:
        optimizer.step()


def forward_pass(mod, inputs, contexts, _collect_outputs=True):
    cloned_inputs = clone_inputs(inputs)
    with nested(*contexts):
        if isinstance(cloned_inputs, Mapping):
            return mod(**inputs)
        else:
            return mod(*inputs)


def forward_and_backward_pass(mod, inputs, contexts, optimizer, collect_outputs=True):
    cloned_inputs = clone_inputs(inputs)
    optimizer_zero_grad(optimizer, mod)
    with nested(*contexts):
        if isinstance(cloned_inputs, Mapping):
            pred = mod(**cloned_inputs)
        else:
            pred = mod(*cloned_inputs)
        loss = compute_loss(pred)
    DummyGradScaler().scale(loss).backward(retain_graph=True)
    optimizer_step(optimizer)
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None


def run_n_iterations(
    mod,
    inputs,
    contexts,
    optimizer=None,
    is_training=False,
    iterations=STABLENESS_CHECK_ROUNDS,
):
    def _model_iter_fn(mod, inputs, contexts, optimizer, collect_outputs):
        if is_training:
            return forward_and_backward_pass(
                mod, inputs, contexts, optimizer, collect_outputs
            )
        else:
            return forward_pass(mod, inputs, contexts, collect_outputs)

    for _ in range(iterations - 1):
        _model_iter_fn(mod, inputs, contexts, optimizer, collect_outputs=False)
    return _model_iter_fn(mod, inputs, contexts, optimizer, collect_outputs=True)


def get_tolerance_and_cosine_flag(model, is_training, current_device, name):
    tolerance = 1e-4
    cosine = model.dargs.use_cosine_similarity
    # Increase the tolerance for torch allclose
    if model.dargs.precision == "fp16" or model.dargs.precision == "amp":
        if name in REQUIRE_HIGHER_FP16_TOLERANCE:
            return 1e-2, cosine
        return 1e-3, cosine

    if model.dargs.precision == "bf16":
        if name in REQUIRE_HIGHER_BF16_TOLERANCE:
            return 1e-2, cosine

    if is_training and current_device == "cuda":
        tolerance = 1e-3
        if name in REQUIRE_COSINE_TOLERACE:
            cosine = True
        elif name in REQUIRE_HIGHER_TOLERANCE:
            tolerance = 1e-3
        elif name in REQUIRE_EVEN_HIGHER_TOLERANCE:
            tolerance = 8 * 1e-2
    return tolerance, cosine


def skip_accuracy_check_as_eager_non_deterministic(is_training):
    if is_training:
        return SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS
    return set()


def check_accuracy(tbmodel: "torchbenchmark.util.model.BenchmarkModel") -> str:
    import functools

    import torch
    from torch._dynamo.utils import same
    from torch.utils._pytree import tree_map

    def _equal_nan_p(precision):
        equal_nan = True
        if precision == "fp32":
            equal_nan = False
        return equal_nan

    def reset_rng_state():
        set_random_seed()

    def deepcopy_model(model, is_deepcopy):
        if not is_deepcopy:
            return model
        try:
            return copy.deepcopy(model)
        except TypeError:
            return model

    def maybe_cast(tbmodel, model, example_inputs):
        model = deepcopy_model(model, tbmodel.DEEPCOPY)
        example_inputs = clone_inputs(example_inputs)
        if tbmodel.dargs.precision == "fp32":
            model, example_inputs = cast_to(torch.float32, model, example_inputs)
        elif tbmodel.dargs.precision == "fp16":
            model, example_inputs = cast_to(torch.float16, model, example_inputs)
        elif tbmodel.dargs.precision == "bf16":
            model, example_inputs = cast_to(torch.bfloat16, model, example_inputs)
        return model, example_inputs

    model, example_inputs = tbmodel.get_module()
    name = tbmodel.name
    current_device = tbmodel.device
    optimizer = None
    is_training = tbmodel.test == "train"
    is_deepcopy = tbmodel.DEEPCOPY
    accuracy_status = "pass"
    contexts = []
    equal_nan = _equal_nan_p(tbmodel.dargs.precision)

    if tbmodel.device == "cuda" and tbmodel.dargs.precision == "amp" and is_training:
        contexts.append(torch.cuda.amp.autocast)
    elif (
        tbmodel.dargs.precision == "amp"
        and tbmodel.dargs.precision == "bf16"
        and tbmodel.device == "cpu"
    ):
        contexts.append(torch.cpu.amp.autocast)

    # Collect the fp64 reference outputs to be used later for accuracy checking.
    fp64_outputs = None
    try:
        model_fp64, inputs_fp64 = cast_to(
            torch.float64,
            deepcopy_model(model, is_deepcopy=True),
            clone_inputs(example_inputs),
        )
        optimizer = init_optimizer(
            name, current_device, model_fp64.parameters(), is_training
        )
        fp64_outputs = run_n_iterations(
            model_fp64, inputs_fp64, contexts, optimizer, is_training
        )
        fp64_outputs = tree_map(
            lambda x: (
                x.to(torch.float64)
                if isinstance(x, torch.Tensor) and x.is_floating_point()
                else x
            ),
            fp64_outputs,
        )
    except Exception:
        log.warning(
            "fp64 golden ref were not generated for %s. Setting accuracy check to cosine",
            tbmodel.name,
        )
        tbmodel.dargs.use_cosine_similarity = True
        fp64_outputs = None

    tolerance, cos_similarity = get_tolerance_and_cosine_flag(
        tbmodel, is_training, current_device, name
    )
    # Cast the model to float16/float32 as necessary
    model, example_inputs = maybe_cast(tbmodel, model, example_inputs)
    with pick_grad(name, is_training):
        # Get results of native pytorch
        reset_rng_state()
        try:
            model_copy = deepcopy_model(model, is_deepcopy)
            optimizer = init_optimizer(
                name, current_device, model_copy.parameters(), is_training
            )
            correct_result = run_n_iterations(
                model_copy,
                clone_inputs(example_inputs),
                contexts,
                optimizer,
                is_training,
            )
        except Exception as e:
            accuracy_status = (
                "eager_1st_run_OOM"
                if isinstance(e, torch.cuda.OutOfMemoryError)
                else "eager_1st_run_fail"
            )
            print(e)
            log.exception(e)
            return accuracy_status

        # Rerun native pytorch
        reset_rng_state()
        try:
            model_copy = deepcopy_model(model, is_deepcopy)
            optimizer = init_optimizer(
                name, current_device, model_copy.parameters(), is_training
            )
            correct_rerun_result = run_n_iterations(
                model_copy,
                clone_inputs(example_inputs),
                contexts,
                optimizer,
                is_training,
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
            if name not in skip_accuracy_check_as_eager_non_deterministic(
                is_training
            ) and not same(
                correct_result,
                correct_rerun_result,
                fp64_ref=None,
                cos_similarity=False,
                tol=0,
                equal_nan=equal_nan,
            ):
                is_same = False
        except Exception as e:
            # Sometimes torch.allclose may throw RuntimeError
            is_same = False

        if not is_same:
            accuracy_status = "eager_two_runs_differ"
            return accuracy_status

        if (
            not hasattr(tbmodel.opt_args, "torchdynamo")
            or not tbmodel.opt_args.torchdynamo
        ):
            return accuracy_status

        correct_rerun_result = None

        # Run with Dynamo
        # Sometime CI fails with random triton compilation failure which will be skipped for now
        # TODO: revisit this after switching to new Triton runtime
        reset_rng_state()
        torch._dynamo.reset()
        optimize_ctx = functools.partial(
            torch.compile,
            backend=tbmodel.opt_args.torchdynamo,
        )
        try:
            model_copy = deepcopy_model(model, is_deepcopy)
            optimizer = init_optimizer(
                name, current_device, model_copy.parameters(), is_training
            )
            optimized_model_iter_fn = optimize_ctx(run_n_iterations)
            new_result = optimized_model_iter_fn(
                model_copy, example_inputs, contexts, optimizer, is_training
            )
        except Exception as e:
            log.exception(e)
            accuracy_status = (
                "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "fail_to_run"
            )
            return accuracy_status

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
