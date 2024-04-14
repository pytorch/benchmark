import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor import triton_heuristics
from torch._inductor.codecache import AsyncCompile

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


@torch.compile
def trivial_add_kernel(*args):
    return sum([torch.tensor(1.0, device="cuda"), *args])


async_compile = AsyncCompile()

inductor_nop = async_compile.triton(
    "inductor_nop",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_heuristics

@triton_heuristics.pointwise(
    size_hints=[1],
    triton_meta={'signature': {0: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(), equal_to_1=())]},
)
@triton.jit
def inductor_nop(x):
    pass
""",
    device_str="cuda",
)


inductor_nop_args = async_compile.triton(
    "inductor_nop_args",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_heuristics

@triton_heuristics.pointwise(
    size_hints=[1],
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(5, 6, 7, 8, 9, 10, 11, 12, 13))]},
)
@triton.jit
def inductor_nop_args(t1, t2, t3, t4, t5, i1, i2, i3, i4, i5, i6, i7, i8, i9):
    pass
""",
    device_str="cuda",
)


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

    @register_benchmark()
    def nop_triton_compiled_kernel_run(self, *args):
        if len(args) == 0:
            bin = nop_kernel[1,]()

        else:
            bin = nop_with_args_kernel[1,](*args)
            args = args[:-5]  # remove tl.constexpr args
        function = bin.function
        metadata = (
            bin.packed_metadata if hasattr(bin, "packed_metadata") else bin.metadata
        )
        if hasattr(triton.compiler.CompiledKernel, "launch_metadata"):
            return lambda: bin.run(
                1, 1, 1, 0, function, metadata, None, None, None, *args
            )
        else:
            return lambda: bin.run(
                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, function, None, None, metadata, *args
            )

    @register_benchmark()
    def nop_inductor_kernel_run(self, *args):
        stream = get_raw_stream(0)
        grid = triton_heuristics.grid(1)

        if len(args) == 0:
            return lambda: inductor_nop.run(1, grid=grid, stream=stream)
        args = args[:-5]
        return lambda: inductor_nop_args.run(*args, grid=grid, stream=stream)

    @register_benchmark()
    def nop_inductor_kernel(self, *args):
        return lambda: trivial_add_kernel(*args)

    @register_benchmark(baseline=True)
    def nop_python_function(self, *args):
        def nop():
            pass

        return nop
