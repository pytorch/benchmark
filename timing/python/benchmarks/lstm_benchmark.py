import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time

import torch.jit
from torch._thnn import type2backend
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
from torch.nn._functions.rnn import LSTMCell
import torch.nn.functional as F

from framework import AttrDict, ListBenchmark
import gc


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

    if variable:
        V = lambda x, requires_grad=False: Variable(x, requires_grad=False)
    elif autograd:
        V = lambda x, requires_grad=False: Variable(x, requires_grad=requires_grad)
    else:
        V = lambda x, requires_grad=False: x

    input = V(torch.randn(batch_size, input_size).cuda(device=gpu))
    hx0   = V(torch.randn(batch_size, hidden_size).cuda(device=gpu), requires_grad=True)
    cx0   = V(torch.randn(batch_size, hidden_size).cuda(device=gpu), requires_grad=True)
    w_ih  = V(t_def(torch.randn(4 * hidden_size, input_size)).cuda(device=gpu), requires_grad=True)
    w_hh  = V(t_def(torch.randn(4 * hidden_size, hidden_size)).cuda(device=gpu), requires_grad=True)

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

    cpu_times = []
    cuda_times = []
    for i in range(warmup + benchmark):
        if i > warmup:
            gc.collect()
            start_cpu = time.time()
            start_cuda = torch.cuda.Event(enable_timing=True)
            end_cuda = torch.cuda.Event(enable_timing=True)
            start_cuda.record()

        hx, cx = hx0, cx0
        for j in range(seq_len):
            hx, cx = lstm(input, (hx, cx), w_ih, w_hh)
        if backward:
            hx.sum().backward()

        if i > warmup:
            end_cpu = time.time()
            end_cuda.record()
            torch.cuda.synchronize()
            cuda_times.append(start_cuda.elapsed_time(end_cuda))
            cpu_times.append(end_cpu - start_cpu)

    return cpu_times, cuda_times


class LSTMBench(ListBenchmark):
    args = [
        # Compare variable overhead
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             autograd=False, variable=False, fused=False, jit=False,
             backward=False, warmup=10, benchmark=20),
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             autograd=False, variable=True, fused=False, jit=False,
             backward=False, warmup=10, benchmark=20),
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             autograd=True, variable=False, fused=False, jit=False,
             backward=False, warmup=10, benchmark=20),

        # Compare JIT/fused/None Forward
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             warmup=10, benchmark=20, variable=False,
             autograd=True, fused=False, jit=False, backward=False),
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             warmup=10, benchmark=20, variable=False,
             autograd=True, fused=True, jit=False, backward=False),
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             warmup=10, benchmark=20, variable=False, 
             autograd=True, fused=False, jit=True, backward=False),

        # Compare JIT/fused/None Forward + Backward
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             warmup=10, benchmark=20, variable=False,
             autograd=True, fused=False, jit=False, backward=True),
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             warmup=10, benchmark=20, variable=False,
             autograd=True, fused=True, jit=False, backward=True),
        dict(batch_size=1, input_size=256, hidden_size=512, seq_len=512,
             warmup=10, benchmark=20, variable=False,
             autograd=True, fused=False, jit=True, backward=True),
    ]

    user_counters = {
        "cpu_time_avg": 0,
        "cuda_time_avg": 0,
    }

    def setup(self, state, arg):
        pass

    def benchmark(self, state, arg):
        cpu_times, cuda_times = run_lstm(**arg)

        cpu_time_avg = sum(cpu_times) / arg.benchmark
        cuda_time_avg = sum(cuda_times) / arg.benchmark

        state.cpu_time_avg = "{:3.4f}".format(cpu_time_avg)
        state.cuda_time_avg = "{:3.4f}".format(cuda_time_avg)
