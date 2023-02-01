import torch
from torchbenchmark.util.backends import create_backend
from typing import List

WARMUP_ITER = 3

@create_backend
def cudagraph(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    cudagraph_func_name = f"cudagraph_{model.test}"
    assert hasattr(model, cudagraph_func_name), f"CUDA Graph only works on models implement {cudagraph_func_name}()"
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
        model.g = cuda_graph
        model.invoke = getattr(model, cudagraph_func_name)
    return _cudagraph, backend_args
