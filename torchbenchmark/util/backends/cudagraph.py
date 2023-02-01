import torch
from torchbenchmark.util.backends import create_backend
from typing import List

WARMUP_ITER = 3

@create_backend
def cudagraph(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    def _cudagraph():
        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(WARMUP_ITER):
                model.invoke()
        torch.cuda.current_stream().wait_stream(s)
        # capture
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            model.invoke()
        def _run_cudagraph(g=cuda_graph):
            g.replay()
        model.invoke = _run_cudagraph
    return _cudagraph, backend_args
