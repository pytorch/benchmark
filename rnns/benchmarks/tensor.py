import torch

if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import Bench
else:
    from .benchmark_common import benchmark_init
    from .common import Bench


def run_tensor(broadcast=True):
    benchmark_init(0, 0, False)

    d = torch.zeros(1000, 1000)
    e = torch.zeros(1)

    def time_broadcast():
        d * e

    def time_no_broadcast():
        d * d

    if broadcast:
        fn = time_broadcast
    else:
        fn = time_no_broadcast

    name = "mul_bcast" if broadcast else "mul_no_bcast"
    iter_timer = Bench(name=name, cuda=False, warmup_iters=2)
    for _ in range(20):
        with iter_timer:
            fn()

    return iter_timer
