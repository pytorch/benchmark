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
    parser.add_argument('--cpu',          type=int, default=0,    help="CPU to run on")
    parser.add_argument('--gpu',          type=int, default=0,    help="GPU to run on")
    parser.add_argument('--batch-size',   type=int, default=1,    help="Batch size")
    parser.add_argument('--input-size',   type=int, default=256,  help="Input size")
    parser.add_argument('--hidden-size',  type=int, default=512,  help="Hidden size")
    parser.add_argument('--layers',       type=int, default=1,    help="Layers")
    parser.add_argument('--seq-len',      type=int, default=512,  help="Sequence length")
    parser.add_argument('--warmup',       type=int, default=10,   help="Warmup iterations")
    parser.add_argument('--benchmark',    type=int, default=30,   help="Benchmark iterations")
    args = parser.parse_args()

    benchmark_common.init(args.cpu, args.gpu)

    pprint.pprint(vars(args))

    def V(x):
        return Variable(x)  # mandatory

    input = V(torch.randn(args.seq_len, args.batch_size, args.input_size).cuda(device=args.gpu))
    hx    = V(torch.randn(args.layers, args.batch_size, args.hidden_size).cuda(device=args.gpu))
    cx    = V(torch.randn(args.layers, args.batch_size, args.hidden_size).cuda(device=args.gpu))

    lstm = torch.nn.LSTM(args.input_size, args.hidden_size, args.layers).cuda(device_id=args.gpu)
    lstm.flatten_parameters()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(args.warmup + args.benchmark):
        gc.collect()
        start.record()
        start_cpu = time.time()  # high precision only for Linux
        lstm(input, (hx, cx))
        end_cpu = time.time()
        end.record()
        torch.cuda.synchronize()
        msecs = start.elapsed_time(end)
        print("cudnn_lstm({:2d}): {:8.3f} msecs ({:8.3f} msecs cpu)".format(i, msecs, (end_cpu-start_cpu)*1000), file=sys.stderr)

if __name__ == "__main__":
    main()
