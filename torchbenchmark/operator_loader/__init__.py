from typing import List, Any, Generator, Optional
import argparse
from .operator_inp_utils import OperatorInputsLoader, to_channels_last
from torch._ops import OpOverload
from torch._dynamo.backends.cudagraphs import cudagraphs_inner
from torch._inductor.compile_fx import compile_fx
from torch._inductor.utils import gen_gm_and_inputs
from torch.utils._pytree import tree_map_only
import sys
import types
import torch
from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)

timm_loader = OperatorInputsLoader.get_timm_loader()
huggingface_loader = OperatorInputsLoader.get_huggingface_loader()
torchbench_loader = OperatorInputsLoader.get_torchbench_loader()

def parse_args(extra_args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--channel-list",
        action='store_true',
        help="Flag to enable channel list benchmarking.",
    )
    return parser.parse_args(extra_args)

def list_operators() -> List[OpOverload]:
    all_ops = (
        list(timm_loader.get_all_ops())
        + list(huggingface_loader.get_all_ops())
        + list(torchbench_loader.get_all_ops())
    )
    # remove duplicate operators
    all_ops = list(set(all_ops))
    return all_ops


def load_opbench_by_name_from_loader(op_name: str):
    all_ops = list_operators()
    if op_name not in all_ops:
        raise ValueError(f"{op_name} is not found in the operator loader.")

def create_operator_class(op_eval: OpOverload):

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        native_args = parse_args(extra_args)
        self.channel_list = native_args.channel_list
        self.device = tb_args.device
        self.huggingface_loader = huggingface_loader
        self.torchbench_loader = torchbench_loader
        self.timm_loader = timm_loader


    def get_input_iter(self) -> Generator:
        inps_gens = [self.huggingface_loader, self.torchbench_loader, self.timm_loader]
        for inp_gen in inps_gens:
            for inp in inp_gen:
                args, kwargs = inp
                if self.channel_list:
                    args, kwargs = tree_map_only(torch.Tensor, to_channels_last, (args, kwargs))
                gm, gm_args = gen_gm_and_inputs(self.op_eval, args, kwargs)
                torch.jit._builtins._register_builtin(
                    torch.ops.aten.convolution_backward.default, "aten::convolution_backward"
                )
                if self.device == 'cuda':
                    cudagraph_eager = cudagraphs_inner(gm, gm_args, copy_outputs=False, copy_inputs=False)
                    self.eager_op = cudagraph_eager
                    compiled_fn = compile_fx(gm, gm_args)
                    cudagraph_compiled = cudagraphs_inner(compiled_fn, gm_args, copy_outputs=False, copy_inputs=False)
                    self.inductor_op = cudagraph_compiled
                else:
                    self.eager_op = gm
                    self.inductor_op = gm

                yield gm_args

    @register_benchmark(baseline=True)
    def eager(self, input):
        return lambda: self.eager_op(input)

    @register_benchmark()
    def inductor(self, input):
        return lambda: self.inductor_op(input)

    class_attrs = {
        'eager': eager,
        'inductor': inductor,
        "get_input_iter": get_input_iter,
    }
    new_class = type("Operator", (BenchmarkOperator,), class_attrs)
    new_class.op_eval = op_eval
    return new_class

def dynamically_create_native_operator_classes(op_eval: OpOverload, args: argparse.Namespace):
    """
    To keep same with custom operators, we dynamically create operator classes here.
    """
    class_name = f"native_{str(op_eval).replace('.', '_')}"
    # create a new module for each operator
    op_name_module = types.ModuleType(f"operator_loader.{class_name}")
    sys.modules[f"operator_loader.{class_name}"] = op_name_module
    op_class = create_operator_class(op_eval)
    op_name_module.Operator = op_class
