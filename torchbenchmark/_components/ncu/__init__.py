from typing import Callable


def do_bench_ncu_in_task(
    fn: Callable,
    grad_to_none=None,
    fast_flush=True,
    range_name: str = "",
) -> None:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    :param output_dir: Output directory to store the trace
    :type output_dir: str, optional
    """
    import torch

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # we don't want `fn` to accumulate gradient values
    # if it contains a backward pass. So we clear the
    # provided gradients
    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None
    # we clear the L2 cache before run
    cache.zero_()
    with torch.cuda.nvtx.range(range_name):
        fn()
