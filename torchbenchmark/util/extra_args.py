import argparse
from typing import List, Optional
from torchbenchmark.util.backends.fx2trt import enable_fx2trt
from torchbenchmark.util.backends.fuser import enable_fuser
from torchbenchmark.util.backends.jit import enable_jit
from torchbenchmark.util.backends.torch_trt import enable_torchtrt
from torchbenchmark.util.env_check import correctness_check
from torchbenchmark.util.framework.vision.args import enable_fp16_half

def add_bool_arg(parser: argparse.ArgumentParser, name: str, default_value: bool=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default_value})

def is_torchvision_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL

def is_hf_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'HF_MODEL') and model.HF_MODEL

def get_hf_maxlength(model: 'torchbenchmark.util.model.BenchmarkModel') -> Optional[int]:
    return model.max_length if is_hf_model(model) else None

def check_fp16(model: 'torchbenchmark.util.model.BenchmarkModel', fp16: str) -> bool:
    if fp16 == "half":
        return is_torchvision_model(model) and model.test == 'eval' and model.device == 'cuda'
    if fp16 == "amp":
        is_cuda_eval_test = (model.test == 'eval' and model.device == 'cuda')
        support_amp = hasattr(model, "enable_amp")
        return is_cuda_eval_test or support_amp
    return True

# torchvision models uses fp16 half mode by default, others use fp32
def get_fp16_default(model: 'torchbenchmark.util.model.BenchmarkModel') -> str:
    if is_torchvision_model(model) and model.test == 'eval' and model.device == 'cuda':
        return "half"
    return "no"

# Dispatch arguments based on model type
def parse_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    parser.add_argument("--fuser", type=str, default="", help="enable fuser")
    parser.add_argument("--torch_trt", action='store_true', help="enable torch_tensorrt")
    parser.add_argument("--fp16", choices=["no", "half", "amp"], default=get_fp16_default(model), help="enable fp16 modes from: no fp16, half, or amp")
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.test = model.test
    args.batch_size = model.batch_size
    if args.device == "cpu":
        args.fuser = None
    if not check_fp16(model, args.fp16):
        raise NotImplementedError(f"fp16 value: {args.fp16}, fp16 (amp mode) is only supported by CUDA inference tests, "
                                  f"fp16 (half mode) is only supported by torchvision CUDA inference tests.")
    if not (model.device == "cuda" and model.test == "eval"):
        if args.fx2trt or args.torch_trt:
            raise NotImplementedError("TensorRT only works for CUDA inference tests.")
    if hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL:
        args.cudagraph = False
    return args

def apply_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    if args.fuser:
        enable_fuser(args.fuser)
    if args.fp16 and not args.fp16 == "no":
        if args.test == "eval":
            model.eager_output = model.invoke()
        if args.fp16 == "half":
            model.model, model.example_inputs = enable_fp16_half(model.model, model.example_inputs)
        elif args.fp16 == "amp":
            # check if the model has native amp support
            if hasattr(model, "enable_amp"):
                model.enable_amp()
            else:
                import torch
                model.add_context(torch.cuda.amp.autocast(dtype=torch.float16))
        else:
            assert False, f"Get invalid fp16 value: {args.fp16}. Please report a bug."
        if args.test == "eval":
            model.output = model.invoke()
            model.correctness = correctness_check(model.eager_output, model.output)
            del model.eager_output
            del model.output
    if args.jit:
        if args.test == "eval":
            model.eager_output = model.invoke()
        # model can handle jit code themselves through 'jit_callback' function
        if hasattr(model, 'jit_callback'):
            model.jit_callback()
        else:
            # if model doesn't have customized jit code, use the default jit script code
            module, exmaple_inputs = model.get_module()
            model.set_module(enable_jit(model=module, example_inputs=exmaple_inputs, test=args.test))
        if args.test == "eval":
            model.output = model.invoke()
            model.correctness = correctness_check(model.eager_output, model.output)
            del model.eager_output
            del model.output
    if args.fx2trt:
        model.eager_output = model.invoke()
        if args.jit:
            raise NotImplementedError("fx2trt with JIT is not available.")
        module, exmaple_inputs = model.get_module()
        # get the output tensor of eval
        model.eager_output = model.eval()
        model.set_module(enable_fx2trt(args.batch_size, fp16=args.fp16, model=module, example_inputs=exmaple_inputs,
                                       is_hf_model=is_hf_model(model), hf_max_length=get_hf_maxlength(model)))
        model.output = model.eval()
        model.correctness = correctness_check(model.eager_output, model.output)
        del model.eager_output
        del model.output
    if args.torch_trt:
        model.eager_output = model.invoke()
        module, exmaple_inputs = model.get_module()
        precision = 'fp16' if args.fp16 is not "no" else 'fp32'
        model.set_module(enable_torchtrt(precision=precision, model=module, example_inputs=exmaple_inputs))
        model.output = model.invoke()
        model.correctness = correctness_check(model.eager_output, model.output)
        del model.eager_output
        del model.output
