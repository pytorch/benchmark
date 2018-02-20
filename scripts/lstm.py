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
    #return hx.clone(), cx.clone()
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
    parser.add_argument('--cpu',                     type=int, default=0,     help="CPU to run on")
    parser.add_argument('--gpu',                     type=int, default=0,     help="GPU to run on")
    parser.add_argument('--batch-size',              type=int, default=1,     help="Batch size")
    parser.add_argument('--input-size',              type=int, default=256,   help="Input size")
    parser.add_argument('--hidden-size',             type=int, default=512,   help="Hidden size")
    parser.add_argument('--seq-len',                 type=int, default=None,  help="Sequence length")
    parser.add_argument('--warmup',                  type=int, default=10,    help="Warmup iterations")
    parser.add_argument('--benchmark',               type=int, default=20,    help="Benchmark iterations")
    parser.add_argument('--autograd',                action='store_true',     help="Use autograd")
    parser.add_argument('--variable',                action='store_true',     help="Use Variable, but not autograd (measure baseline overhead)")
    parser.add_argument('--fused',                   action='store_true',     help="Use fused cell")
    parser.add_argument('--jit',                     action='store_true',     help="Use JIT compiler (implies --autograd)")
    parser.add_argument('--backward',                action='store_true',     help="Run backwards computation")
    parser.add_argument('--skip-cpu-governor-check', action='store_true',     help="Skip checking whether CPU governor is set to `performance`")
    args = parser.parse_args()

    if args.jit:
        args.autograd = True

    if args.backward:
        args.autograd = True

    # TODO: Support BPTT

    if args.seq_len is None:
        # TODO: Not sure about the wisdom of this
        if args.backward:
            args.seq_len = 32
        else:
            args.seq_len = 512

    assert not (args.jit and args.fused)
    assert not (args.variable and args.autograd)

    pprint.pprint(vars(args))

    benchmark_common.init(args.cpu, args.gpu, args.skip_cpu_governor_check)

    if args.variable:
        V = lambda x, requires_grad=False: Variable(x, requires_grad=False)
    elif args.autograd:
        V = lambda x, requires_grad=False: Variable(x, requires_grad=requires_grad)
    else:
        V = lambda x, requires_grad=False: x

    if args.fused:
        lstm = fused_lstm
    elif args.jit:
        lstm = torch.jit.compile(nderivs=0)(unfused_lstm)
    else:
        lstm = unfused_lstm

    input = V(torch.randn(args.batch_size, args.input_size).cuda(device=args.gpu))
    hx0   = V(torch.randn(args.batch_size, args.hidden_size).cuda(device=args.gpu), requires_grad=True)
    cx0   = V(torch.randn(args.batch_size, args.hidden_size).cuda(device=args.gpu), requires_grad=True)
    w_ih  = V(t_def(torch.randn(4 * args.hidden_size, args.input_size)).cuda(device=args.gpu), requires_grad=True)
    w_hh  = V(t_def(torch.randn(4 * args.hidden_size, args.hidden_size)).cuda(device=args.gpu), requires_grad=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(args.warmup + args.benchmark):
        gc.collect()
        start.record()
        times = []
        start_cpu_secs = time.time()  # high precision only for Linux
        hx, cx = hx0, cx0
        for j in range(args.seq_len):
            times.append(time.time())
            hx, cx = lstm(input, (hx, cx), w_ih, w_hh)
        if args.backward:
            hx.sum().backward()
        end_cpu_secs = time.time()
        end.record()
        torch.cuda.synchronize()
        gpu_msecs = start.elapsed_time(end)
        benchmark_common.print_results_usecs("lstm", i, gpu_msecs*1000, (end_cpu_secs - start_cpu_secs)*1000000, args.seq_len)

if __name__ == "__main__":
    main()
