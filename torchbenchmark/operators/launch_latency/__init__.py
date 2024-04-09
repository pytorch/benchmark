import torch
import triton
import triton.language as tl

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)


@triton.jit
def nop_kernel():
    pass


@triton.jit
def nop_with_args_kernel(
    t1,
    t2,
    t3,
    t4,
    t5,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
):
    pass


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["walltime"]

    def get_input_iter(self):
        yield tuple()
        targs = [torch.zeros(1, device="cuda") for _ in range(5)]
        iargs = [1 for _ in range(9)]
        cargs = [32 for _ in range(5)]
        yield tuple([*targs, *iargs, *cargs])

    def get_x_val(self, example_inputs) -> float:
        return len(example_inputs)

    @register_benchmark()
    def nop_triton_kernel(self, *args):
        if len(args) == 0:
            return lambda: nop_kernel[1,]()
        return lambda: nop_with_args_kernel[1,](*args)

    @register_benchmark(baseline=True)
    def nop_python_function(self, *args):
        def nop():
            pass

        return nop
