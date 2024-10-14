import argparse
import csv
import os
import statistics
from typing import Any, Callable, Generator, List, Optional, Tuple

import numpy
import torch
import torch._inductor.config as inductor_config
import triton

from torchbenchmark import REPO_PATH

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    IS_FBCODE,
    llama_shapes,
    register_benchmark,
    register_metric,
    register_x_val,
)

from .kernels import matmul as kernels
from .partition_k import matmul_partition_k
from .persistent_matmul import (
    matmul_persistent,
    matmul_tma_persistent,
    matmul_tma_persistent_cached,
)
from .triton_matmul import (
    matmul as triton_tutorial_matmul,
    matmul_kernel as triton_tutorial_matmul_kernel,
)

if inductor_config.is_fbcode():
    from hammer.ops.triton.triton_matmul import triton_matmul as hstu_triton_matmul

    HAS_HAMMER = True
else:
    HAS_HAMMER = False

try:
    torch.ops.load_library(
        "//pytorch/benchmark/torchbenchmark/operators/gemm/cutlass:colfax_gemm_lib"
    )
    colfax_gemm = torch.ops.cutlass.colfax_gemm
except (ImportError, IOError, AttributeError) as e:
    colfax_gemm = None

BUILDIN_SHAPES = [
    (256, 256, 256, None),
    (384, 384, 384, None),
    (512, 512, 512, None),
    (640, 640, 640, None),
    (768, 768, 768, None),
    (896, 896, 896, None),
    (1024, 1024, 1024, None),
    (1152, 1152, 1152, None),
    (1280, 1280, 1280, None),
    (1408, 1408, 1408, None),
    (1536, 1536, 1536, None),
    (1664, 1664, 1664, None),
    (1792, 1792, 1792, None),
    (1920, 1920, 1920, None),
    (2048, 2048, 2048, None),
    (2176, 2176, 2176, None),
    (2304, 2304, 2304, None),
    (2432, 2432, 2432, None),
    (2560, 2560, 2560, None),
    (2688, 2688, 2688, None),
    (2816, 2816, 2816, None),
    (2944, 2944, 2944, None),
    (3072, 3072, 3072, None),
    (3200, 3200, 3200, None),
    (3328, 3328, 3328, None),
    (3456, 3456, 3456, None),
    (3584, 3584, 3584, None),
    (3712, 3712, 3712, None),
    (3840, 3840, 3840, None),
    (3968, 3968, 3968, None),
    (4096, 4096, 4096, None),
]

SPLIT_K_SHAPES = [
    (m, m, k, None)
    for m in [16 * i for i in range(1, 5)]
    for k in [4096 * i for i in range(1, 9)]
]


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchBench Gemm operator Benchmark")
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--bias", type=int)
    parser.add_argument("--input", type=str)
    parser.add_argument("--splitk", action="store_true", default=False)
    parser.add_argument("--llama", action="store_true", default=False)
    parser.add_argument("--layout", type=str, default="tn")
    args = parser.parse_args(args)
    return args


def read_shapes_from_csv(csv_path: str) -> List[List[int]]:
    input_file_path = os.path.join(
        REPO_PATH, "torchbenchmark", "operators", "gemm", csv_path
    )
    shapes = []
    with open(input_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = [
                int(row.get(f)) if row.get(f) else None for f in ("M", "N", "K", "Bias")
            ]
            shapes.append(shape)
    return shapes


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["speedup", "tflops"]
    DEFAULT_PRECISION = "fp16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.use_cuda_graphs = False
        gemm_args = parse_args(self.extra_args)
        self.layout = gemm_args.layout
        if gemm_args.input:
            self.shapes = read_shapes_from_csv(gemm_args.input)
        elif gemm_args.splitk:
            self.shapes = SPLIT_K_SHAPES
        elif gemm_args.llama:
            self.shapes = llama_shapes()
        elif gemm_args.m and gemm_args.k and gemm_args.n:
            self.shapes = [(gemm_args.m, gemm_args.n, gemm_args.k, gemm_args.bias)]
        else:
            self.shapes = BUILDIN_SHAPES

    @register_benchmark()
    def triton_tutorial_matmul(self, a, b, bias) -> Callable:
        if not bias == None:
            return lambda: triton_tutorial_matmul(a, b) + bias
        else:
            return lambda: triton_tutorial_matmul(a, b)

    @register_benchmark()
    def matmul_partition_k(self, a, b, bias) -> Callable:
        if not bias == None:
            return lambda: matmul_partition_k(a, b) + bias
        else:
            return lambda: matmul_partition_k(a, b)

    @register_benchmark()
    def triton_persistent_matmul(self, a, b, bias) -> Callable:
        if not bias == None:
            return lambda: matmul_persistent(a, b) + bias
        else:
            return lambda: matmul_persistent(a, b)

    @register_benchmark(enabled=not IS_FBCODE)
    def triton_tma_persistent_matmul(self, a, b, bias) -> Callable:
        b = b.T.contiguous()
        if not bias == None:
            return lambda: matmul_tma_persistent(a, b) + bias
        else:
            return lambda: matmul_tma_persistent(a, b)

    @register_benchmark(enabled=not IS_FBCODE)
    def triton_tma_persistent_cached_matmul(self, a, b, bias) -> Callable:
        b = b.T.contiguous()
        if not bias == None:
            return lambda: matmul_tma_persistent_cached(a, b) + bias
        else:
            return lambda: matmul_tma_persistent_cached(a, b)

    @register_benchmark(enabled=torch.version.cuda is not None)
    def triton_ops_matmul(self, a, b, bias) -> Callable:
        if bias is None:
            return lambda: kernels.matmul(a, b)
        return lambda: kernels.matmul(a, b) + bias

    @register_benchmark(baseline=True)
    def aten_matmul(self, a, b, bias) -> Callable:
        if not bias == None:
            return lambda: torch.matmul(a, b) + bias
        else:
            return lambda: torch.matmul(a, b)

    @register_benchmark(enabled=HAS_HAMMER)
    def hstu_triton_matmul(self, a, b, bias) -> Callable:
        if not bias == None:
            return lambda: hstu_triton_matmul(a, b) + bias
        else:
            return lambda: hstu_triton_matmul(a, b)

    @register_benchmark(enabled=bool(colfax_gemm))
    def colfax_cutlass_matmul(self, a, b, bias) -> Callable:
        assert colfax_gemm, f"colfax_gemm operator is not available."
        if not bias == None:
            return lambda: colfax_gemm(a, b, alpha=1.0, beta=1.0) + bias
        else:
            return lambda: colfax_gemm(a, b, alpha=1.0, beta=1.0)

    @register_benchmark()
    def pt2_triton_matmul(self, a, b, bias) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            autotune_fallback_to_aten=False,
        ):
            if bias is not None:
                f = lambda a, b: a.matmul(b) + bias
            else:
                f = lambda a, b: a.matmul(b)
            compiled = torch.compile(f, dynamic=False)
            compiled(a, b)
        return lambda: compiled(a, b)

    @register_benchmark()
    def pt2_cutlass_matmul(self, a, b, bias) -> Callable:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="CUTLASS",
            autotune_fallback_to_aten=False,
        ):
            if bias is not None:
                f = lambda a, b: a.matmul(b) + bias
            else:
                f = lambda a, b: a.matmul(b)
            # cutlass needs to know the static shape, so set dynamic to False
            compiled = torch.compile(f, dynamic=False)
            compiled(a, b)
        return lambda: compiled(a, b)

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        # x-value: computation intensity
        a, w, bias = example_inputs
        m, k = a.size()
        k, n = w.size()
        return (m, n, k)

    @register_metric()
    def gbps(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, w, bias = example_inputs
        numel = a.numel() + w.numel() + (torch.mm(a, w).numel())
        numel = numel * a.element_size() / 1e9
        return numel / metrics.latency * 1e3

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, w, bias = example_inputs
        m, k = a.size()
        k, n = w.size()
        if not bias == None:
            flops = m * k * 2 * n + 2 * m * n
        else:
            flops = m * k * 2 * n
        return flops / metrics.latency / 1e12 * 1e3

    @staticmethod
    def _scaled_randn(*args, scale: float, **kwargs) -> torch.Tensor:
        """
        This provides more numerically stable inputs for GEMMs. The +1
        eliminates very small values that could result in denormals, and the
        scale (which should be set to K in an M*N*K GEMM) reduces the size of
        the absolute error.

        In particular, for a given element in the output tensor, the cumulative
        error is eps * 2 * K, where eps is the smallest precision representable
        in the dtype. By scaling the element by K, we avoid the error growing
        with the size of the tensor.
        """
        return (torch.randn(*args, **kwargs) + 1) / scale

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            m, n, k, bias = shape
            a = self._scaled_randn(
                (m, k), scale=k, device=self.device, dtype=self.dtype
            )
            w = self._scaled_randn(
                (k, n), scale=k, device=self.device, dtype=self.dtype
            )
            # Convert inputs to column-major if layout is "n" (non-transposed)
            if self.layout[0] == "n":
                a = a.T.contiguous().T
            if self.layout[1] == "n":
                w = w.T.contiguous().T
            if not bias == None:
                bias = torch.randn(
                    (bias), device=self.device, dtype=self.dtype
                ).requires_grad_(False)
            yield a, w, bias

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        return torch.allclose(output, baseline_output)

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=[
                    "m",
                    "n",
                    "k",
                ],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "aten_matmul",
                    "triton_tutorial_matmul",
                    "triton_kernels_matmul",
                    "hstu_triton_matmul",
                ],  # possible values for `line_arg``
                line_names=[
                    "ATen GEMM",
                    "Triton Tutorial GEMM",
                    "triton/kernels/matmul",
                    "HSTU Triton GEMM",
                ],  # label name for the lines
                styles=[
                    ("blue", "-"),
                    ("green", "-"),
                    ("red", "-"),
                    ("yellow", "-"),
                ],  # line styles
                ylabel="tflops",  # label name for the y-axis
                plot_name="gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(m, n, k, provider):
            tflops = self.output.get_y_vals((m, n, k), provider, "tflops")
            return tflops

        save_path = "/tmp/test_gemm"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
