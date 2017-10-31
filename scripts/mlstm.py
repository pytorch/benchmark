import benchmark_common

import torch
from torch.autograd import Variable
import torch.jit

import argparse
import pprint
import gc
import time

def mlstm_raw(input, hx, cx, w_xm, w_hm, w_ih, w_mh):
    # w_ih holds W_hx, W_ix, W_ox, W_fx
    # w_mh holds W_hm, W_im, W_om, W_fm

    m = input.mm(w_xm.t()) * hx.mm(w_hm.t())
    gates = input.mm(w_ih.t()) + m.mm(w_mh.t())

    ingate, forgetgate, hiddengate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    outgate = outgate.sigmoid()
    forgetgate = forgetgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * hiddengate)
    hy = (cy * outgate).tanh()

    return hy, cy


def main():
    parser = argparse.ArgumentParser(description="PyTorch LSTM benchmark.")
    parser.add_argument('--cpu',          type=int, default=0,    help="CPU to run on")
    parser.add_argument('--gpu',          type=int, default=0,    help="GPU to run on")
    parser.add_argument('--batch-size',   type=int, default=1,    help="Batch size")
    parser.add_argument('--input-size',   type=int, default=205,  help="Input size")
    parser.add_argument('--hidden-size',  type=int, default=1900, help="Hidden size")
    parser.add_argument('--embed-size',   type=int, default=None, help="Embed size")
    parser.add_argument('--seq-len',      type=int, default=20,   help="Sequence length")
    parser.add_argument('--warmup',       type=int, default=10,   help="Warmup iterations")
    parser.add_argument('--benchmark',    type=int, default=20,   help="Benchmark iterations")
    parser.add_argument('--autograd',     action='store_true',    help="Use autograd")
    parser.add_argument('--jit',          action='store_true',    help="Use JIT compiler (implies --autograd)")
    args = parser.parse_args()

    if args.embed_size is None:
        args.embed_size = args.hidden_size

    if args.jit:
        args.autograd = True

    pprint.pprint(vars(args))

    benchmark_common.init(args.cpu, args.gpu)

    if args.autograd:
        V = Variable
    else:
        V = lambda x: x

    if args.jit:
        mlstm = torch.jit.compile(nderivs=0)(mlstm_raw)
    else:
        mlstm = mlstm_raw

    input = V(torch.randn(args.seq_len, args.batch_size, args.input_size).cuda(device=args.gpu))
    hx    = V(torch.randn(args.batch_size, args.hidden_size).cuda(device=args.gpu))
    cx    = V(torch.randn(args.batch_size, args.hidden_size).cuda(device=args.gpu))
    w_xm  = V(torch.randn(args.embed_size, args.input_size).cuda(device=args.gpu))
    w_hm  = V(torch.randn(args.embed_size, args.hidden_size).cuda(device=args.gpu))
    w_ih  = V(torch.randn(4 * args.hidden_size, args.input_size).cuda(device=args.gpu))
    w_mh  = V(torch.randn(4 * args.hidden_size, args.embed_size).cuda(device=args.gpu))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(args.warmup + args.benchmark):
        gc.collect()
        start.record()
        start_cpu_secs = time.time()  # high precision only for Linux
        for j in range(args.seq_len):
            hx, cx = mlstm(input[j], hx, cx, w_xm, w_hm, w_ih, w_mh)
        end_cpu_secs = time.time()
        end.record()
        torch.cuda.synchronize()
        gpu_msecs = start.elapsed_time(end)
        benchmark_common.print_results_usecs("mlstm", i, gpu_msecs*1000, (end_cpu_secs - start_cpu_secs)*1000000, args.seq_len)

if __name__ == "__main__":
    main()
