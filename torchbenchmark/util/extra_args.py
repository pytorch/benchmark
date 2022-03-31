import argparse
from typing import List, Optional, Tuple
from torchbenchmark.util.backends.fx2trt import enable_fx2trt
from torchbenchmark.util.backends.jit import enable_jit
from torchbenchmark.util.backends.torch_trt import enable_torchtrt

def enable_opt_args(opt_args: argparse.Namespace) -> bool:
    "Check if any of the optimizations is enabled."
    opt_args_dict = vars(opt_args)
    for k in opt_args_dict:
        if opt_args_dict[k]:
            return True
    return False

def add_bool_arg(parser: argparse.ArgumentParser, name: str, default_value: bool=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default_value})

def is_timm_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'TIMM_MODEL') and model.TIMM_MODEL

def is_torchvision_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL

def is_hf_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'HF_MODEL') and model.HF_MODEL

def get_hf_maxlength(model: 'torchbenchmark.util.model.BenchmarkModel') -> Optional[int]:
    return model.max_length if is_hf_model(model) else None

def check_fp16(model: 'torchbenchmark.util.model.BenchmarkModel', fp16: str) -> bool:
    if fp16 == "half":
        return model.test == 'eval' and model.device == 'cuda' and hasattr(model, "enable_fp16_half")
    if fp16 == "amp":
        return model.test == 'eval' and model.device == 'cuda'
    return True

# torchvision models uses fp16 half mode by default, others use fp32
def get_fp16_default(model: 'torchbenchmark.util.model.BenchmarkModel') -> str:
    if hasattr(model, "DEFAULT_EVAL_CUDA_PRECISION") and model.test == 'eval' and model.device == 'cuda':
        return model.DEFAULT_EVAL_CUDA_PRECISION
    return "no"

def parse_decoration_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", choices=["no", "half", "amp"], default=get_fp16_default(model), help="enable fp16 modes from: no fp16, half, or amp")
    dargs, opt_args = parser.parse_known_args(extra_args)
    if not check_fp16(model, dargs.fp16):
        raise NotImplementedError(f"fp16 value: {dargs.fp16}, fp16 is only supported on CUDA inference tests, "
                                  f"fp16 (half mode) is only supported if the model implements `enable_fp16_half()` callback function.")
    return (dargs, opt_args)

def apply_decoration_args(model: 'torchbenchmark.util.model.BenchmarkModel', dargs: argparse.Namespace):
    if dargs.fp16 == "half":
        model.enable_fp16_half()
    elif dargs.fp16 == "amp":
        # model handles amp itself if it has 'enable_amp' callback function
        if hasattr(model, "enable_amp"):
            model.enable_amp()
        else:
            import torch
            model.add_context(lambda: torch.cuda.amp.autocast(dtype=torch.float16))
    elif not dargs.fp16 == "no":
        assert False, f"Get an invalid fp16 value: {dargs.fp16}. Please report a bug."

# Dispatch arguments based on model type
def parse_opt_args(model: 'torchbenchmark.util.model.BenchmarkModel', opt_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    parser.add_argument("--fuser", type=str, default="", choices=["fuser0", "fuser1", "fuser2"], help="enable fuser")
    parser.add_argument("--torch_trt", action='store_true', help="enable torch_tensorrt")
    args = parser.parse_args(opt_args)
    args.jit = model.jit
    if model.device == "cpu" and args.fuser:
        raise NotImplementedError("Fuser only works with GPU.")
    if not (model.device == "cuda" and model.test == "eval"):
        if args.fx2trt or args.torch_trt:
            raise NotImplementedError("TensorRT only works for CUDA inference tests.")
    if is_torchvision_model(model):
        args.cudagraph = False
    return args

def apply_opt_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    if args.fuser:
        import torch
        model.add_context(lambda: torch.jit.fuser(args.fuser))
    if args.jit:
        # model can handle jit code themselves through the 'jit_callback' callback function
        if hasattr(model, 'jit_callback'):
            model.jit_callback()
        else:
            # if model doesn't have customized jit code, use the default jit script code
            module, exmaple_inputs = model.get_module()
            model.set_module(enable_jit(model=module, example_inputs=exmaple_inputs, test=model.test))
    if args.fx2trt:
        if args.jit:
            raise NotImplementedError("fx2trt with JIT is not available.")
        module, exmaple_inputs = model.get_module()
        fp16 = not (model.dargs.fp16 == "no")
        model.set_module(enable_fx2trt(model.batch_size, fp16=fp16, model=module, example_inputs=exmaple_inputs,
                                       is_hf_model=is_hf_model(model), hf_max_length=get_hf_maxlength(model)))
    if args.torch_trt:
        module, exmaple_inputs = model.get_module()
        precision = 'fp16' if not model.dargs.fp16 == "no" else 'fp32'
        model.set_module(enable_torchtrt(precision=precision, model=module, example_inputs=exmaple_inputs))
