import torch
import torch.nn.functional as F
import triton
from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from . import tutorial


class Operator(BenchmarkOperator):
    @register_benchmark()
    def layer_norm_fwd(self, *args):
        return lambda: tutorial.layer_norm(*args)

    @register_benchmark(baseline=True)
    def layer_norm_fwd_baseline(self, *args):
        return lambda: F.layer_norm(*args)

    def get_input_iter(self):
        M = 4096
        eps = 1e-5
        for N in [512 * i for i in range(2, 32)]:
            x_shape = (M, N)
            w_shape = (x_shape[-1],)
            x = -2.3 + 0.5 * torch.randn(x_shape, dtype=self.dtype, device="cuda")
            weight = torch.rand(
                w_shape, dtype=self.dtype, device="cuda", requires_grad=True
            )
            bias = torch.rand(
                w_shape, dtype=self.dtype, device="cuda", requires_grad=True
            )
            yield (x, w_shape, weight, bias, eps)

    def get_x_val(self, args):
        _, N = args[0].shape
        return N

    @register_metric()
    def gbps(self, fn_name, args, metrics: BenchmarkOperatorMetrics) -> float:
        x = args[0]

        def gbps(ms):
            return 2 * x.numel() * x.element_size() / ms * 1e-6

        return list(map(gbps, metrics.latency))

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=[
                    "layer_norm_fwd",
                    "layer_norm_fwd_baseline",
                ],
                line_names=[
                    "Triton",
                    "Torch",
                ],
                styles=[("blue", "-"), ("green", "-")],
                ylabel="GB/s",
                plot_name="layer-norm-fwd",
                args={"M": 4096},
            )
        )
        def _plot(M, N, provider):
            gbps, max_gbps, min_gbps = self.output.get_y_vals(N, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_layer_norm")
