import argparse
from typing import List
from torchbenchmark.util.backends.fx2trt import enable_fx2trt
from torchbenchmark.util.backends.fuser import enable_fuser
from torchbenchmark.util.backends.jit import enable_jit
from torchbenchmark.util.backends.torch_trt import enable_torchtrt
from torchbenchmark.util.env_check import correctness_check
from torchbenchmark.util.framework.vision.args import enable_fp16

def add_bool_arg(parser: argparse.ArgumentParser, name: str, default_value: bool=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default_value})

def is_torchvision_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL

def allow_fp16(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return is_torchvision_model(model) and model.test == 'eval' and model.device == 'cuda'

# Dispatch arguments based on model type
def parse_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    parser.add_argument("--fuser", type=str, default="", help="enable fuser")
    parser.add_argument("--torch_trt", action='store_true', help="enable torch_tensorrt")
    # TODO: Enable fp16 for all model inference tests
    # fp16 is only True for torchvision models running CUDA inference tests
    # otherwise, it is False
    fp16_default_value = False
    if allow_fp16(model):
        fp16_default_value = True
    add_bool_arg(parser, "fp16", fp16_default_value)
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.test = model.test
    args.batch_size = model.batch_size
    if args.device == "cpu":
        args.fuser = None
    if not allow_fp16(model) and args.fp16:
        raise NotImplementedError("fp16 is only implemented for torchvision models inference tests on CUDA.")
    if not (model.device == "cuda" and model.test == "eval"):
        if args.fx2trt or args.torch_trt:
            raise NotImplementedError("TensorRT only works for CUDA inference tests.")
    if hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL:
        args.cudagraph = False
    return args

def apply_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    if args.fuser:
        enable_fuser(args.fuser)
    if args.fp16:
        assert allow_fp16(model), "Eval fp16 is only available on CUDA for torchvison models."
        model.model, model.example_inputs = enable_fp16(model.model, model.example_inputs)
    if args.jit:
        # model can handle jit code themselves through 'jit_callback' function
        if hasattr(model, 'jit_callback'):
            model.jit_callback()
        else:
            # if model doesn't have customized jit code, use the default jit script code
            module, exmaple_inputs = model.get_module()
            model.set_module(enable_jit(model=module, example_inputs=exmaple_inputs, test=args.test))
    if args.fx2trt:
        if args.jit:
            raise NotImplementedError("fx2trt with JIT is not available.")
        module, exmaple_inputs = model.get_module()
        # get the output tensor of eval
        model.eager_output = model.eval()
        model.set_module(enable_fx2trt(args.batch_size, fp16=args.fp16, model=module, example_inputs=exmaple_inputs))
        model.output = model.eval()
        model.correctness = correctness_check(model.eager_output, model.output)
    if args.torch_trt:
        module, exmaple_inputs = model.get_module()
        precision = 'fp16' if args.fp16 else 'fp32'
        # get the output tensor of eval
        model.eager_output = model.eval()
        model.set_module(enable_torchtrt(precision=precision, model=module, example_inputs=exmaple_inputs))
        model.output = model.eval()
        model.correctness = correctness_check(model.eager_output, model.output)

