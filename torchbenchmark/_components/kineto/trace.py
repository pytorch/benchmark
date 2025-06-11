import random
import string
from datetime import datetime
from functools import partial
from typing import Callable

import torch
import torch.profiler as profiler

DEFAULT_PROFILE_OPTS = {
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": True,
    "with_flops": True,
    "with_modules": True,
}

if not hasattr(torch.version, "git_version"):
    from .fb.run_utils import trace_handler


def do_bench_kineto(
    fn: Callable,
    warmup=25,
    grad_to_none=None,
    fast_flush=True,
    profile_opts=None,
    output_dir=None,
) -> str:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    :param profile_opts: Options to pass into profiler.profile
    :type profile_opts: dict, optional
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
    activity_groups = [
        profiler.ProfilerActivity.CUDA,
        profiler.ProfilerActivity.CPU,
    ]
    if profile_opts is None:
        profile_opts = DEFAULT_PROFILE_OPTS
    prefix = f"torchbench_{fn._name}"
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{''.join(random.choices(string.digits, k=10))}.json"
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=n_warmup, active=1, repeat=1),
        activities=activity_groups,
        record_shapes=profile_opts["record_shapes"],
        profile_memory=profile_opts["profile_memory"],
        with_stack=profile_opts["with_stack"],
        with_flops=profile_opts["with_flops"],
        with_modules=profile_opts["with_modules"],
        on_trace_ready=(
            partial(trace_handler, name)
            if not hasattr(torch.version, "git_version")
            else profiler.tensorboard_trace_handler(output_dir)
        ),
    ) as prof:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for i in range(n_warmup + 1):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            # we clear the L2 cache before run
            cache.zero_()
            fn()
            prof.step()
    if not hasattr(torch.version, "git_version"):
        return f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/test/{name}.gz&bucket=pyper_traces"
    else:
        return output_dir
