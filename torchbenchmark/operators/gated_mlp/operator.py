from typing import Generator
import torch

try:
    import mirage as mi
    HAS_MIRAGE = True
except:
    HAS_MIRAGE = False

from torchbenchmark.util.triton_op import (
    BenchmarkOperator,
    register_benchmark
)

class Operator(BenchmarkOperator):

    DEFAULT_PRECISION = "fp16"

    def __init__(self, tb_args, extra_args):
        if HAS_MIRAGE:
            self.mirage_optimized_graph = self._get_mi_optimized_graph()
        self.baseline = self._get_baseline()


    def _get_mi_optimized_graph(self):
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(8, 4096), dtype=mi.float16)
        W1 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
        W2 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
        D1 = graph.matmul(X, W1)
        D2 = graph.matmul(X, W2)
        O = graph.mul(graph.silu(D1), graph.silu(D2))
        graph.mark_output(O)
        optimized_graph = graph.superoptimize()
        return optimized_graph


    @register_benchmark(baseline=True, ci=False)
    def pt2(self, x, w1, w2):
        def _baseline(x, w1, w2):
            D1 = torch.matmul(x, w1)
            D2 = torch.matmul(x, w2)
            m = torch.nn.silu()
            F1 = m(D1)
            F2 = m(D2)
            O = torch.mul(F1, F2)
            return O
        compiled = torch.compile(_baseline)
        compiled(x, w1, w2)
        return lambda: compiled(x, w1, w2)


    @register_benchmark(ci=HAS_MIRAGE)
    def mirage(self, x, w1, w2):
        return lambda: self.mirage_optimized_graph([x, w1, w2])


    def get_input_iter(self) -> Generator:
        x = torch.randn(8, 4096, dtype=torch.float16, device=self.device)
        w1 = torch.randn(4096, 4096, dtype=torch.float16, device=self.device)
        w2 = torch.randn(4096, 4096, dtype=torch.float16, device=self.device)
        yield (x, w1, w2)
        return
