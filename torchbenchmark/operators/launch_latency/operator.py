import triton.language as tl
from torch import zeros
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)
from triton.compiler import CompiledKernel

from .kernels import nop_kernel, nop_with_args_kernel, trivial_add_kernel


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["walltime"]

    def get_input_iter(self):
        yield tuple()
        targs = [zeros(1, device="cuda") for _ in range(5)]
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
        if hasattr(CompiledKernel, "launch_metadata"):
            return lambda: bin.run(
                1, 1, 1, 0, function, metadata, None, None, None, *args
            )
        else:
            return lambda: bin.run(
                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, function, None, None, metadata, *args
            )

    @register_benchmark()
    def nop_inductor_kernel(self, *args):
        return lambda: trivial_add_kernel(*args)

    @register_benchmark(baseline=True)
    def nop_python_function(self, *args):
        def nop():
            pass

        return nop
