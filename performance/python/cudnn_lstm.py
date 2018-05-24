import benchmark_common

import torch
from torch.autograd import Variable
import torch.jit
import torch.nn

import argparse
import pprint
import gc
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="PyTorch CuDNN LSTM benchmark.")
    parser.add_argument('--cpu',                     type=int,  default=0,    help="CPU to run on")
    parser.add_argument('--gpu',                     type=int,  default=0,    help="GPU to run on")
    parser.add_argument('--batch-size',              type=int,  default=1,    help="Batch size")
    parser.add_argument('--input-size',              type=int,  default=256,  help="Input size")
    parser.add_argument('--hidden-size',             type=int,  default=512,  help="Hidden size")
    parser.add_argument('--layers',                  type=int,  default=1,    help="Layers")
    parser.add_argument('--seq-len',                 type=int,  default=512,  help="Sequence length")
    parser.add_argument('--warmup',                  type=int,  default=10,   help="Warmup iterations")
    parser.add_argument('--benchmark',               type=int,  default=30,   help="Benchmark iterations")
    parser.add_argument('--skip-cpu-governor-check', action='store_true',     help="Skip checking whether CPU governor is set to `performance`")
    args = parser.parse_args()

    benchmark_common.init(args.cpu, args.gpu, args.skip_cpu_governor_check)

    pprint.pprint(vars(args))

    def V(x):
        return Variable(x)  # mandatory

    input = V(torch.randn(args.seq_len, args.batch_size, args.input_size).cuda(args.gpu))
    hx    = V(torch.randn(args.layers, args.batch_size, args.hidden_size).cuda(args.gpu))
    cx    = V(torch.randn(args.layers, args.batch_size, args.hidden_size).cuda(args.gpu))

    lstm = torch.nn.LSTM(args.input_size, args.hidden_size, args.layers).cuda(args.gpu)
    lstm.flatten_parameters()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(args.warmup + args.benchmark):
        gc.collect()
        start.record()
        start_cpu_secs = time.time()  # high precision only for Linux
        lstm(input, (hx, cx))
        end_cpu_secs = time.time()
        end.record()
        torch.cuda.synchronize()
        gpu_msecs = start.elapsed_time(end)
        benchmark_common.print_results_usecs("cudnn_lstm", i, gpu_msecs*1000, (end_cpu_secs - start_cpu_secs)*1000000, args.seq_len)

if __name__ == "__main__":
    main()
