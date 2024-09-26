from typing import Callable

import torch


class cuda_profiler_range:
    def __init__(self, use_cuda_profiler_range):
        self.use_cuda_profiler_range = use_cuda_profiler_range

    def __enter__(self):
        if self.use_cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()

    def __exit__(self, *exc_info):
        if self.use_cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()


def do_bench_in_task(
    fn: Callable,
    grad_to_none=None,
    range_name: str = "",
    warmup: bool = False,
    warmup_time: int = 25,
    use_cuda_profiler_range: bool = False,
) -> None:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    """

    fn()
    torch.cuda.synchronize()

    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    if warmup:
        # Estimate the runtime of the function
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            cache.zero_()
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # compute number of warmup and repeat
        n_warmup = max(1, int(warmup / estimate_ms))
        # Warm-up
        for _ in range(n_warmup):
            fn()

    # we don't want `fn` to accumulate gradient values
    # if it contains a backward pass. So we clear the
    # provided gradients
    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None
    # we clear the L2 cache before run
    cache.zero_()
    with cuda_profiler_range(use_cuda_profiler_range), torch.cuda.nvtx.range(
        range_name
    ):
        fn()
