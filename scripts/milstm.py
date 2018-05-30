import benchmark_common

import torch
from torch.autograd import Variable
import torch.jit

import argparse
import pprint
import gc
import time

# Benchmark script for the Multiplicative Integration LSTM cell
# Paper reference: https://arxiv.org/abs/1606.06630
#
# Code reference: https://github.com/pytorch/translate/blob/master/pytorch_translate/rnn_cell.py#L27:44
def milstm_raw(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())

    # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
    gates = (alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias)

    # Same as LSTMCell after this point
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()

    return hy, cy


def main():
    parser = argparse.ArgumentParser(description="PyTorch LSTM benchmark.")
    parser.add_argument('--cpu',                     type=int, default=0,     help="CPU to run on")
    parser.add_argument('--gpu',                     type=int, default=0,     help="GPU to run on")
    parser.add_argument('--batch-size',              type=int, default=1,     help="Batch size")
    parser.add_argument('--input-size',              type=int, default=205,   help="Input size")
    parser.add_argument('--hidden-size',             type=int, default=1900,  help="Hidden size")
    parser.add_argument('--embed-size',              type=int, default=None,  help="Embed size")
    parser.add_argument('--seq-len',                 type=int, default=20,    help="Sequence length")
    parser.add_argument('--warmup',                  type=int, default=10,    help="Warmup iterations")
    parser.add_argument('--benchmark',               type=int, default=20,    help="Benchmark iterations")
    parser.add_argument('--autograd',                action='store_true',     help="Use autograd")
    parser.add_argument('--jit',                     action='store_true',     help="Use JIT compiler (implies --autograd)")
    parser.add_argument('--skip-cpu-governor-check', action='store_true',     help="Skip checking whether CPU governor is set to `performance`")
    args = parser.parse_args()

    if args.embed_size is None:
        args.embed_size = args.hidden_size

    if args.jit:
        args.autograd = True

    pprint.pprint(vars(args))

    benchmark_common.init(args.cpu, args.gpu, args.skip_cpu_governor_check)

    if args.autograd:
        V = Variable
    else:
        V = lambda x: x

    if args.jit:
        mlstm = torch.jit.script(milstm_raw)
    else:
        mlstm = milstm_raw


    input = V(torch.randn(args.seq_len, args.batch_size, args.input_size).cuda(device=args.gpu))
    hx = V(torch.randn(args.batch_size, args.hidden_size).cuda(device=args.gpu))
    cx = V(torch.randn(args.batch_size, args.hidden_size).cuda(device=args.gpu))
    ih = V(torch.randn(4 * args.hidden_size, args.input_size).cuda(device=args.gpu))
    hh = V(torch.randn(4 * args.hidden_size, args.hidden_size).cuda(device=args.gpu))
    bias_var = V(torch.randn(4 * args.hidden_size).cuda(device=args.gpu))
    alpha = V(torch.randn(4 * args.hidden_size).cuda(device=args.gpu))
    beta_h = V(torch.randn(4 * args.hidden_size).cuda(device=args.gpu))
    beta_i = V(torch.randn(4 * args.hidden_size).cuda(device=args.gpu))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(args.warmup + args.benchmark):
        gc.collect()
        start.record()
        start_cpu_secs = time.time()  # high precision only for Linux
        for j in range(args.seq_len):
            hx, cx = mlstm(input[j], hx, cx, ih, hh, alpha, beta_i, beta_h, bias_var)
        end_cpu_secs = time.time()
        end.record()
        torch.cuda.synchronize()
        gpu_msecs = start.elapsed_time(end)
        benchmark_common.print_results_usecs("milstm", i, gpu_msecs*1000, (end_cpu_secs - start_cpu_secs)*1000000, args.seq_len)

if __name__ == "__main__":
    main()
