import os
from typing import Generator, List

import torch
import triton
from .kernels import triton_add_kernel

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)


class Operator(BenchmarkOperator):

    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics):
        gbps = (
            lambda ms: 3
            * example_inputs[0].element_size()
            * example_inputs[0].numel()
            / ms
            * 1e-6
        )
        return list(map(gbps, metrics.latency))

    @register_benchmark()
    def triton_add(self, x: torch.Tensor, y: torch.Tensor):

        # We need to preallocate the output.
        output = torch.empty_like(x)
        n_elements = output.numel()
        # The SPMD launch grid denotes the number of kernel instances that run in parallel.
        # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
        # In this case, we use a 1D grid where the size is the number of blocks:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.

        def _inner():
            return_val = triton_add_kernel[grid](
                x, y, output, n_elements, BLOCK_SIZE=1024
            )
            return return_val

        return _inner

    @register_benchmark(baseline=True)
    def torch_add(self, x: torch.Tensor, y: torch.Tensor):
        return lambda: x + y

    def get_x_vals(self) -> List[int]:
        return [2**i for i in range(12, 28, 1)]

    def get_x_val(self, example_inputs):
        return len(example_inputs[0])

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["size"],  # Argument names to use as an x-axis for the plot.
                x_vals=self.x_vals,  # Different possible values for `x_name`.
                x_log=True,  # x axis is logarithmic.
                line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
                line_vals=[
                    "torch_add",
                    "triton_add",
                ],  # Possible values for `line_arg`.
                line_names=["Torch", "Triton"],  # Label name for the lines.
                styles=[("blue", "-"), ("green", "-")],  # Line styles.
                ylabel="GB/s",  # Label name for the y-axis.
                plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
                args={},  # Values for function arguments not in `x_names` and `y_name`.
            )
        )
        def _plot(size, provider):

            gbps, max_gbps, min_gbps = self.output.get_y_vals(size, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/vector_add")

    def get_input_iter(self) -> Generator:
        for size in self.get_x_vals():
            x = torch.rand(size, device=self.device, dtype=self.dtype)
            y = torch.rand(size, device=self.device, dtype=self.dtype)
            yield x, y
        while True:
            yield None
