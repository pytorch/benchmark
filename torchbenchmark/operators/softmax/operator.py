from typing import Generator, List

import torch
import triton
import triton.language as tl

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)


class Operator(BenchmarkOperator):

    @register_benchmark()
    def triton_softmax(self, x):
        n_rows, n_cols = x.shape
        # The block size is the smallest power of two greater than the number of columns in `x`
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        # Another trick we can use is to ask the compiler to use more threads per row by
        # increasing the number of warps (`num_warps`) over which each row is distributed.
        # You will see in the next tutorial how to auto-tune this value in a more natural
        # way so you don't have to come up with manual heuristics yourself.
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        # Allocate output
        y = torch.empty_like(x)

        # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
        # f the input matrix
        def _inner():
            Operator.softmax_kernel[(n_rows,)](
                y,
                x,
                x.stride(0),
                y.stride(0),
                n_cols,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return y

        return _inner

    @triton.jit
    def softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        # The rows of the softmax are independent, so we parallelize across those
        row_idx = tl.program_id(0)
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    @register_benchmark(baseline=True)
    def naive_softmax(self, x):
        """Compute row-wise softmax of X using native pytorch
        We subtract the maximum element in order to avoid overflows. Softmax is invariant to
        this shift.
        """

        def _inner():
            # read  MN elements ; write M  elements
            x_max = x.max(dim=1)[0]
            # read MN + M elements ; write MN elements
            z = x - x_max[:, None]
            # read  MN elements ; write MN elements
            numerator = torch.exp(z)
            # read  MN elements ; write M  elements
            denominator = numerator.sum(dim=1)
            # read MN + M elements ; write MN elements
            ret = numerator / denominator[:, None]
            # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
            return ret

        return _inner

    def get_input_iter(self):
        M = 4096
        for i in range(2, 100):
            N = 128 * i
            yield (torch.randn([M, N], dtype=self.dtype, device=self.device),)

    def get_x_val(self, example_inputs) -> int:
        shape = example_inputs[0].size()
        return shape[1]

    @register_metric()
    def gbps(
        self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics
    ) -> float:
        return (
            2
            * example_inputs[0].nelement()
            * example_inputs[0].element_size()
            * 1e-9
            / (metrics.latency * 1e-3)
        )

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N"],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "triton_softmax",
                    "naive_softmax",
                ],  # possible values for `line_arg``
                line_names=[
                    "Triton",
                    "Torch (native)",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-"), ("green", "--")],  # line styles
                ylabel="GB/s",  # label name for the y-axis
                plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={
                    "M": 4096
                },  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(M, N, provider):
            gbps, max_gbps, min_gbps = self.output.get_y_vals(N, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_softmax")
