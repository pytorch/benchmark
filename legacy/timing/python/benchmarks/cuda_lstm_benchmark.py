import torch
from torch.autograd import Variable
import time

import torch.jit
from torch._thnn import type2backend

from framework import Benchmark


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
        backend.library_state, input_gate, hidden_gate, None, None, cx, hy, cy
    )

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


class CUDALSTMBench(Benchmark):
    common_arg = dict(
        gpu=0,
        batch_size=1,
        input_size=256,
        hidden_size=512,
        seq_len=512,
        benchmark=20,
        warmup=10,
    )
    args = [
        # Compare variable overhead
        dict(
            autograd=False,
            variable=False,
            fused=False,
            jit=False,
            backward=False,
        ),
        dict(
            autograd=False,
            variable=True,
            fused=False,
            jit=False,
            backward=False,
        ),
        dict(
            autograd=True,
            variable=False,
            fused=False,
            jit=False,
            backward=False,
        ),
        # Compare JIT/fused/None Forward
        dict(
            autograd=True,
            variable=False,
            fused=False,
            jit=False,
            backward=False,
        ),
        dict(
            autograd=True,
            variable=False,
            fused=True,
            jit=False,
            backward=False,
        ),
        dict(
            autograd=True,
            variable=False,
            fused=False,
            jit=True,
            backward=False,
        ),
        # Compare JIT/fused/None Forward + Backward
        dict(
            autograd=True,
            variable=False,
            fused=False,
            jit=False,
            backward=True,
        ),
        dict(
            autograd=True, variable=False, fused=True, jit=False, backward=True
        ),
        dict(
            autograd=True, variable=False, fused=False, jit=True, backward=True
        ),
    ]

    user_counters = {
        "cpu_time_avg": 0,
        "cuda_time_avg": 0,
        "gpu": "0",
        "batch_size": "1",
        "input_size": "256",
        "hidden_size": "512",
        "seq_len": "512",
    }

    def setupRun(self, state, arg_):
        arg = arg_.copy()
        autograd = arg["autograd"]
        variable = arg["variable"]
        fused = arg["fused"]
        jit = arg["jit"]
        backward = arg["backward"]

        arg.update(self.common_arg)

        gpu = arg["gpu"]
        batch_size = arg["batch_size"]
        input_size = arg["input_size"]
        hidden_size = arg["hidden_size"]

        if jit:
            autograd = True

        if backward:
            autograd = True

        assert not (jit and fused)
        assert not (variable and autograd)

        def V(x, requires_grad=False):
            if variable:
                return Variable(x, requires_grad=False)
            elif autograd:
                return Variable(x, requires_grad=requires_grad)
            else:
                return x

        input = V(torch.randn(batch_size, input_size).cuda(device=gpu))
        hx0 = V(
            torch.randn(batch_size, hidden_size).cuda(device=gpu),
            requires_grad=True,
        )
        cx0 = V(
            torch.randn(batch_size, hidden_size).cuda(device=gpu),
            requires_grad=True,
        )
        w_ih = V(
            t_def(torch.randn(4 * hidden_size, input_size)).cuda(device=gpu),
            requires_grad=True,
        )
        w_hh = V(
            t_def(torch.randn(4 * hidden_size, hidden_size)).cuda(device=gpu),
            requires_grad=True,
        )

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
            lstm = wrap_hidden(
                torch.jit.trace(input, hx0, cx0, w_ih, w_hh)(_unfused_lstm)
            )
        else:
            print("using unfused lstm")
            lstm = wrap_hidden(_unfused_lstm)

        state.backward = backward
        state.hx0 = hx0
        state.cx0 = cx0
        state.w_ih = w_ih
        state.w_hh = w_hh
        state.input = input
        state.lstm = lstm
        state.cpu_time = 0
        state.cuda_time = 0
        state.iter = 0

    def benchmark(self, state, arg):
        start_cpu = time.time()
        start_cuda = torch.cuda.Event(enable_timing=True)
        end_cuda = torch.cuda.Event(enable_timing=True)
        start_cuda.record()

        seq_len = self.common_arg["seq_len"]
        hx, cx = state.hx0, state.cx0
        for j in range(seq_len):
            hx, cx = state.lstm(state.input, (hx, cx), state.w_ih, state.w_hh)
        if state.backward:
            hx.sum().backward()

        end_cpu = time.time()
        end_cuda.record()
        torch.cuda.synchronize()
        state.cuda_time += start_cuda.elapsed_time(end_cuda)
        state.cpu_time += end_cpu - start_cpu
        state.iter += 1

    def teardownRun(self, state, arg):

        cpu_time_avg = state.cpu_time / state.iter
        cuda_time_avg = state.cuda_time / state.iter

        state.cpu_time_avg = "{:3.4f}".format(cpu_time_avg)
        state.cuda_time_avg = "{:3.4f}".format(cuda_time_avg)
