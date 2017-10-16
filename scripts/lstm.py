import benchmark_common

import torch
from torch.autograd import Variable
import torch.jit
from torch._thnn import type2backend

import argparse
import pprint
import gc
import time
import sys



# If you swap the transpose here, you can test the effect of pre-transposing.
# In my experiments it didn't account for much
def t_use(x):
    return x
def t_def(x):
    return x.t()


def fused_lstm(input, hidden, w_ih, w_hh):
    hx, cx = hidden

    input_gate = input.mm(t_use(w_ih))
    hidden_gate = hidden[0].mm(t_use(w_hh))

    backend = type2backend[type(input_gate)]

    hy = input_gate.new()
    cy = input_gate.new()

    backend.LSTMFused_updateOutput(
        backend.library_state,
        input_gate, hidden_gate,
        None, None,
        cx, hy, cy)

    return hy, cy


def unfused_lstm(input, hidden, w_ih, w_hh):
    hx, cx = hidden
    gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh))

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
    parser.add_argument('--cpu',          type=int, default=0,    help="CPU to run on")
    parser.add_argument('--gpu',          type=int, default=0,    help="GPU to run on")
    parser.add_argument('--batch-size',   type=int, default=1,    help="Batch size")
    parser.add_argument('--input-size',   type=int, default=256,  help="Input size")
    parser.add_argument('--hidden-size',  type=int, default=512,  help="Hidden size")
    parser.add_argument('--seq-len',      type=int, default=512,  help="Sequence length")
    parser.add_argument('--warmup',       type=int, default=10,   help="Warmup iterations")
    parser.add_argument('--benchmark',    type=int, default=20,   help="Benchmark iterations")
    parser.add_argument('--autograd',     action='store_true',    help="Use autograd")
    parser.add_argument('--fused',        action='store_true',    help="Use fused cell")
    parser.add_argument('--jit',          action='store_true',    help="Use JIT compiler (implies --autograd)")
    args = parser.parse_args()

    if args.jit:
        args.autograd = True

    assert not (args.jit and args.fused)

    pprint.pprint(vars(args))

    benchmark_common.init(args.cpu, args.gpu)

    if args.autograd:
        V = Variable
    else:
        V = lambda x: x

    if args.fused:
        lstm = fused_lstm
    elif args.jit:
        lstm = torch.jit.compile(nderivs=0)(unfused_lstm)
    else:
        lstm = unfused_lstm

    input = V(torch.randn(args.seq_len, args.batch_size, args.input_size).cuda(device=args.gpu))
    hx    = V(torch.randn(args.batch_size, args.hidden_size).cuda(device=args.gpu))
    cx    = V(torch.randn(args.batch_size, args.hidden_size).cuda(device=args.gpu))
    w_ih  = V(t_def(torch.randn(4 * args.hidden_size, args.input_size)).cuda(device=args.gpu))
    w_hh  = V(t_def(torch.randn(4 * args.hidden_size, args.hidden_size)).cuda(device=args.gpu))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(args.warmup + args.benchmark):
        gc.collect()
        start.record()
        start_cpu = time.time()  # high precision only for Linux
        for j in range(args.seq_len):
            hx, cx = lstm(input[j], (hx, cx), w_ih, w_hh)
        end_cpu = time.time()
        end.record()
        torch.cuda.synchronize()
        msecs = start.elapsed_time(end)
        print("lstm({:2d}): {:8.3f} msecs ({:8.3f} msecs cpu)".format(i, msecs, (end_cpu-start_cpu)*1000), file=sys.stderr)

if __name__ == "__main__":
    main()
