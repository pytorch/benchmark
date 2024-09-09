import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch._inductor.utils import maybe_profile


"""
Extensible script to measure PT2-D speedup on MAST

Launch locally:
    torchrun --nnodes=1 --nproc_per_node=8 \
    userbenchmark/mast-sample/main.py --edir=$(pwd)

Launch on MAST: see instructions in fbcode: run_pt2d_oss_benchmark.sh
"""

# Provided by torchx
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def bench(args):
    def fn():
        linear = torch.nn.Linear(100, 200, bias=False, device="cuda")
        x = torch.randn(100, 100, device="cuda")
        out = linear(x)

        # Supports collectives
        dist.all_reduce(out)

        return out

    # Supports torch.compile
    compiled_fn = torch.compile(fn)

    eager_times = []
    compiled_times = []
    n_repeat = 10
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        compiled_fn()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        eager_times.append(t2 - t1)
        compiled_times.append(t3 - t2)

    speedups = [e / c for e, c in zip(eager_times, compiled_times)]
    speedup = np.median(speedups)

    # Supports profile
    with maybe_profile(True) as p:
        eager_out = fn()
        compiled_out = compiled_fn()

    if RANK == 0:
        print(f"eager_out={eager_out}")
        print(f"compiled_out={compiled_out}")

        trace_file = os.path.join(args.edir, f"trace_rank_{RANK}.json")
        print(f"Writing trace to: {trace_file}")
        p.export_chrome_trace(trace_file)

        print(f"{speedup:.3f}x")

        # Supports writing to a local file
        output_file = os.path.join(args.edir, f"metrics_rank_{RANK}.csv")
        print(f"Writing output content to: {output_file}")
        with open(output_file, "w") as f:
            f.write("speedup\n")
            f.write(f"{speedup}\n")


if __name__ == "__main__":
    assert torch.cuda.is_available()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    print(f"Hello from rank={RANK}")

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--edir",
        type=str,
        default=None,
        help="directory to dump job outputs",
    )
    args = parser.parse_args()
    assert args.edir is not None

    bench(args)

    print(f"Bye from rank={RANK}")
