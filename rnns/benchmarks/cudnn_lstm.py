import torch
from torch.autograd import Variable
import torch.jit
import torch.nn

import argparse
import pprint
import gc
import time
import sys

if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import Bench, tag
else:
    from .benchmark_common import benchmark_init
    from .common import Bench, tag


def run_cudnn_lstm(cpu=0, gpu=0, batch_size=1, input_size=256, hidden_size=512,
                   layers=1, seq_len=512, warmup=10, benchmark=30, backward=False,
                   skip_cpu_governor_check=False):

    benchmark_init(cpu, gpu, skip_cpu_governor_check)

    def V(x):
        return Variable(x)  # mandatory

    input = V(torch.randn(seq_len, batch_size, input_size).cuda(gpu))
    hx = V(torch.randn(layers, batch_size, hidden_size).cuda(gpu))
    cx = V(torch.randn(layers, batch_size, hidden_size).cuda(gpu))

    lstm = torch.nn.LSTM(input_size, hidden_size, layers).cuda(gpu)
    lstm.flatten_parameters()

    iter_timer = Bench(name='lstm_cudnn', cuda=True, warmup_iters=warmup)

    for i in range(warmup + benchmark):
        gc.collect()
        with iter_timer:
            hx_t, cx_t = lstm(input, (hx, cx))
            if backward:
                hx_t.sum().backward()

    return iter_timer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CuDNN LSTM benchmark.")
    parser.add_argument('--cpu', type=int, default=0, help="CPU to run on")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--input-size', type=int, default=256, help="Input size")
    parser.add_argument('--hidden-size', type=int, default=512, help="Hidden size")
    parser.add_argument('--layers', type=int, default=1, help="Layers")
    parser.add_argument('--seq-len', type=int, default=512, help="Sequence length")
    parser.add_argument('--warmup', type=int, default=10, help="Warmup iterations")
    parser.add_argument('--benchmark', type=int, default=30, help="Benchmark iterations")
    parser.add_argument('--skip-cpu-governor-check', action='store_true',
                        help="Skip checking whether CPU governor is set to `performance`")
    parser.add_argument('--backward', action='store_true', help="time backward")
    args = parser.parse_args()
    pprint.pprint(vars(args))

    run_cudnn_lstm(**vars(args))
