import argparse
import pprint
import gc
import time
import sys

import torch
from torch.autograd import Variable
import torch.jit
from torch._thnn import type2backend
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
from torch.nn._functions.rnn import LSTMCell
import torch.nn.functional as F

if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import Bench, tag
else:
    from .benchmark_common import benchmark_init
    from .common import Bench, tag

# This file copied from scripts/lstm.py.

# If you swap the transpose here, you can test the effect of pre-transposing.


def t_use(x):
    return x


def t_def(x):
    return x.t()


def fused_lstm(input, hidden, w_ih, w_hh):
    hx, cx = hidden

    input_gate = input.mm(t_use(w_ih))
    hidden_gate = hidden[0].mm(t_use(w_hh))

    backend = type2backend[input_gate.type()]

    hy = input_gate.new()
    cy = input_gate.new()

    backend.LSTMFused_updateOutput(
        backend.library_state,
        input_gate, hidden_gate,
        None, None,
        cx, hy, cy)

    return hy, cy


def fused_autograd_lstm(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    igates = input.mm(t_use(w_ih))
    hgates = hidden[0].mm(t_use(w_hh))
    state = fusedBackend.LSTMFused.apply
    if b_ih is None:
        return state(igates, hgates, hidden[1])
    state(igates, hgates, hidden[1], b_ih, b_hh)


def wrap_hidden(fn):
    def lstm(input, hidden, w_ih, w_hh):
        return fn(input, hidden[0], hidden[1], w_ih, w_hh)

    return lstm


def _unfused_lstm(input, hx, cx, w_ih, w_hh):
    hx, cx
    # return hx.clone(), cx.clone()
    gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh))

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()

    return hy, cy


def run_lstm(cpu=0, gpu=0, batch_size=1, input_size=256, hidden_size=512,
             seq_len=None, warmup=10, benchmark=20, autograd=False,
             variable=False, fused=False, jit=False, backward=False,
             skip_cpu_governor_check=False):
    if jit:
        autograd = True

    if backward:
        autograd = True

    if seq_len is None:
        if backward:
            seq_len = 32
        else:
            seq_len = 512

    assert not (jit and fused)
    assert not (variable and autograd)

    benchmark_init(cpu, gpu, skip_cpu_governor_check)

    if variable:
        def V(x, requires_grad=False):
            return Variable(x, requires_grad=False)
    elif autograd:
        def V(x, requires_grad=False):
            return Variable(x, requires_grad=requires_grad)
    else:
        def V(x, requires_grad=False):
            return x

    input = V(torch.randn(batch_size, input_size).cuda(device=gpu))
    hx0 = V(torch.randn(batch_size, hidden_size).cuda(device=gpu), requires_grad=True)
    cx0 = V(torch.randn(batch_size, hidden_size).cuda(device=gpu), requires_grad=True)
    w_ih = V(t_def(torch.randn(4 * hidden_size, input_size)).cuda(device=gpu), requires_grad=True)
    w_hh = V(t_def(torch.randn(4 * hidden_size, hidden_size)).cuda(device=gpu), requires_grad=True)

    if fused:
        if backward:
            print("using fused_autograd_lstm")
            lstm = fused_autograd_lstm
        else:
            print("using fused_forward_lstm")
            lstm = fused_autograd_lstm
            lstm = fused_lstm
    elif jit:
        print("tracing an unfused lstm")
        lstm = wrap_hidden(torch.jit.trace(input, hx0, cx0, w_ih, w_hh)(_unfused_lstm))
    else:
        print("using unfused lstm")
        lstm = wrap_hidden(_unfused_lstm)

    name = 'lstm_cuda{}{}{}'.format(tag(autograd=autograd), tag(fused=fused),
                                    tag(jit=jit))
    iter_timer = Bench(name=name, cuda=True, warmup_iters=warmup)

    for i in range(warmup + benchmark):
        gc.collect()
        with iter_timer:
            hx, cx = hx0, cx0
            for j in range(seq_len):
                hx, cx = lstm(input, (hx, cx), w_ih, w_hh)
            if backward:
                hx.sum().backward()

    return iter_timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch LSTM benchmark.")
    parser.add_argument('--cpu', type=int, default=0, help="CPU to run on")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--input-size', type=int, default=256, help="Input size")
    parser.add_argument('--hidden-size', type=int, default=512, help="Hidden size")
    parser.add_argument('--seq-len', type=int, default=None, help="Sequence length")
    parser.add_argument('--warmup', type=int, default=10, help="Warmup iterations")
    parser.add_argument('--benchmark', type=int, default=20, help="Benchmark iterations")
    parser.add_argument('--autograd', action='store_true', help="Use autograd")
    parser.add_argument('--variable', action='store_true',
                        help="Use Variable, but not autograd (measure baseline overhead)")
    parser.add_argument('--fused', action='store_true', help="Use fused cell")
    parser.add_argument('--jit', action='store_true', help="Use JIT compiler (implies --autograd)")
    parser.add_argument('--backward', action='store_true', help="Run backwards computation")
    parser.add_argument('--skip-cpu-governor-check', action='store_true',
                        help="Skip checking whether CPU governor is set to `performance`")
    args = parser.parse_args()

    pprint.pprint(vars(args))
    run_lstm(**vars(args))
