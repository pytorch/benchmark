import torch
from torchbenchmark.util.backends import create_backend
from typing import List

WARMUP_ITER = 3

@create_backend
def cudagraph(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    cudagraph_func_name = f"cudagraph_{model.test}"
    assert hasattr(model, cudagraph_func_name), f"CUDA Graph only works on models implement {cudagraph_func_name}()"
    if model.test == "train":
        assert hasattr(model, "SKIP_ZERO_GRAD"), f"The model must support skipping zero grad in its train test."
    def _cudagraph():
        # CUDAGraph can't be copied/pickled, disable copying in correctness checking
        model.DEEPCOPY = False
        model.SKIP_ZERO_GRAD = True
        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(WARMUP_ITER):
                model.opt.zero_grad(set_to_none=True)
                model.invoke()
        torch.cuda.current_stream().wait_stream(s)
        # capture
        cuda_graph = torch.cuda.CUDAGraph()
        model.opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(cuda_graph):
            model.invoke()
        model.g = cuda_graph
        if model.test == "train":
            model.train = getattr(model, cudagraph_func_name)
        else:
            model.eval = getattr(model, cudagraph_func_name)
    return _cudagraph, backend_args
