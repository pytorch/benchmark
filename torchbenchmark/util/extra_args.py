import argparse
import enum
from typing import List, Optional, Tuple
from torchbenchmark.util.backends import list_backends, BACKENDS
from torchbenchmark.util.env_check import is_staged_train_test

TEST_STAGE = enum.Enum('TEST_STAGE', ['FORWARD', 'BACKWARD', 'OPTIMIZER', 'ALL'])
AVAILABLE_PRECISIONS = ["fp32", "tf32", "fp16", "amp", "fx_int8", "bf16","amp_fp16", "amp_bf16"]
QUANT_ENGINES = ["x86", "fbgemm", "qnnpack", "onednn"]


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default_value: bool=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default_value})

def check_precision(model: 'torchbenchmark.util.model.BenchmarkModel', precision: str) -> bool:
    if precision == "fp16":
        return model.device == 'cuda' and hasattr(model, "enable_fp16_half")
    if precision == "tf32":
        return model.device == "cuda"
    if precision == "amp":
        return True
    if precision == "fx_int8":
        return model.device == 'cpu' and hasattr(model, "enable_fx_int8")
    if precision == "bf16":
        return model.device == 'cpu' and hasattr(model, "enable_bf16")
    if precision == "amp_fp16":
        if model.test == 'eval' and model.device == 'cuda':
            return True
        if model.test == 'train' and model.device == 'cuda':
            return hasattr(model, 'enable_amp') or is_staged_train_test(model)
    if precision == "amp_bf16":
        return model.device == 'cpu'
    assert precision == "fp32", f"Expected precision to be one of {AVAILABLE_PRECISIONS}, but get {precision}"
    return True

def check_memory_layout(model: 'torchbenchmark.util.model.BenchmakModel', channels_last: bool) -> bool:
    if channels_last:
        return hasattr(model, 'enable_channels_last')
    return True

def check_distributed_trainer(model: 'torchbenchmark.util.model.BenchmakModel', distributed_trainer: Optional[str]) -> bool:
    if not model.test == "train" and distributed_trainer:
        return False
    return True

def get_precision_default(model: 'torchbenchmark.util.model.BenchmarkModel') -> str:
    if hasattr(model, "DEFAULT_EVAL_CUDA_PRECISION") and model.test == 'eval' and model.device == 'cuda':
        return model.DEFAULT_EVAL_CUDA_PRECISION
    if hasattr(model, "DEFAULT_TRAIN_CUDA_PRECISION") and model.test == 'train' and model.device == 'cuda':
        return model.DEFAULT_TRAIN_CUDA_PRECISION
    return "fp32"

def parse_decoration_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed",
        choices=["ddp", "ddp_no_static_graph", "fsdp"],
        default=None,
        help="Enable distributed trainer",
    )
    parser.add_argument(
        "--distributed_wrap_fn",
        type=str,
        default=None,
        help="Path to function that will apply distributed wrapping fn(model, dargs.distributed)",
    )
    parser.add_argument("--precision", choices=AVAILABLE_PRECISIONS, default=get_precision_default(model), help=f"choose precisions from {AVAILABLE_PRECISIONS}")
    parser.add_argument("--channels-last", action='store_true', help="enable channels-last memory layout")
    parser.add_argument("--accuracy", action="store_true", help="Check accuracy of the model only instead of running the performance test.")
    parser.add_argument("--use_cosine_similarity", action='store_true', help="use cosine similarity for correctness check")
    parser.add_argument("--quant-engine", choices=QUANT_ENGINES, default='x86', help=f"choose quantization engine for fx_int8 precision from {QUANT_ENGINES}")
    dargs, opt_args = parser.parse_known_args(extra_args)
    if not check_precision(model, dargs.precision):
        raise NotImplementedError(f"precision value: {dargs.precision}, "
                                  "fp16 is only supported if the model implements the `enable_fp16_half()` callback function."
                                  "amp is only supported if cuda+eval, or if `enable_amp` implemented,"
                                  "or if model uses staged train interfaces (forward, backward, optimizer).")
    if not check_memory_layout(model, dargs.channels_last):
        raise NotImplementedError(f"Specified channels_last: {dargs.channels_last} ,"
                                  f" but the model doesn't implement the enable_channels_last() interface.")
    if not check_distributed_trainer(model, dargs.distributed):
        raise NotImplementedError(f"We only support distributed trainer {dargs.distributed} for train tests, "
                                  f"but get test: {model.test}")
    return (dargs, opt_args)

def apply_decoration_args(model: 'torchbenchmark.util.model.BenchmarkModel', dargs: argparse.Namespace):
    if dargs.channels_last:
        model.enable_channels_last()
    if dargs.precision == "fp16":
        model.enable_fp16_half()
    elif dargs.precision == "tf32":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif dargs.precision == "amp":
        # model handles amp itself if it has 'enable_amp' callback function (e.g. pytorch_unet)
        if hasattr(model, "enable_amp"):
            model.enable_amp()
    elif dargs.precision == "fx_int8":
        assert model.device == "cpu" and model.test == "eval", f"fx_int8 only work for eval mode on cpu device."
        model.enable_fx_int8(dargs.quant_engine)
    elif dargs.precision == "bf16":
        assert model.device == "cpu", f"bf16 only work on cpu device."
        model.enable_bf16()
    elif dargs.precision == "amp_fp16":
        assert model.device == "cuda", f"{model.device} has no fp16 autocast."
        if model.test == "eval":
            import torch
            model.add_context(lambda: torch.cuda.amp.autocast(dtype=torch.float16))
        elif model.test == "train":
            # the model must implement staged train test
            assert is_staged_train_test(model), f"Expected model implements staged train test (forward, backward, optimizer)."
            import torch
            model.add_context(lambda: torch.cuda.amp.autocast(dtype=torch.float16), stage=TEST_STAGE.FORWARD)
    elif dargs.precision == "amp_bf16":
        import torch
        model.add_context(lambda: torch.cpu.amp.autocast(dtype=torch.bfloat16))
    elif not dargs.precision == "fp32":
        assert False, f"Get an invalid precision option: {dargs.precision}. Please report a bug."

# Dispatch arguments based on model type
def parse_opt_args(model: 'torchbenchmark.util.model.BenchmarkModel', opt_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list_backends(), help="enable backends")
    args, extra_args = parser.parse_known_args(opt_args)
    if args.backend:
        backend = BACKENDS[args.backend]
        model._enable_backend, extra_args = backend(model, backend_args=extra_args)
    return args, extra_args

def apply_opt_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    if args.backend:
        model._enable_backend()
