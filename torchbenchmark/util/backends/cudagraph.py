import torch
from typing import Tuple

def enable_cudagraph(model: 'torchbenchmark.util.model.BenchmarkModel', example_inputs: Tuple[torch.tensor]):
    optimizer = model.optimizer
    loss_fn = model.loss_fn
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model.model(*example_inputs)
            loss = loss_fn(y_pred, model.example_outputs)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)
    # capture
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_y_pred = model.model(*example_inputs)
        static_loss = loss_fn(static_y_pred, model.example_outputs)
        static_loss.backward()
        optimizer.step()
    model.g = g