from flops import *

import torch
from torch.autograd import Variable
import torch.jit
from torch._thnn import type2backend

import gc
import time
import warnings
import os



# very marginal improvement pre-transposing weights
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


def lstm(input, hidden, w_ih, w_hh):
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


def lstm_flops(input, hidden, w_ih, w_hh):
    flops = 0

    hx, cx = hidden
    gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh))

    flops += mm_flops(input.size(), t_use(w_ih).size())
    flops += mm_flops(hx.size(), t_use(w_hh).size())

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    flops += sigmoid_flops(ingate.size())
    flops += sigmoid_flops(forgetgate.size())
    flops += tanh_flops(cellgate.size())
    flops += sigmoid_flops(outgate.size())

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()
    flops += cy.numel() * 4 + tanh_flops(cy.size())

    print(flops)

    return hy, cy


# NB: Be careful with this when benchmarking backward; backward
# uses multiple threads
def cpu_pin(cpu):
    os.sched_setaffinity(0, (cpu, ))


def check_cpu_governor(cpu):
    fp = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor".format(cpu)
    try:
        with open(fp, 'r') as f:
            gov = f.read().rstrip()
            if gov != "performance":
                warnings.warn("CPU {} governor is {} which could lead to variance in performance\n"
                              "Run 'echo performance > {}' as root to turn off power scaling.".format(cpu, gov, fp))
    except IOError as e:
        warnings.warn("Could not find CPU {} governor information in filesystem (are you running on Linux?)\n"
                      "The file '{}' is not readable.\n"
                      "More information:\n\n{}".format(fp, e))


# PyTorch does not natively provide NVML support so we don't check it

def main():
    cpu = 0
    gpu = 0

    cpu_pin(cpu)
    check_cpu_governor(cpu)

    batch_size = 64
    input_size = 256
    hidden_size = 512

    seq_len = 512
    loops = 30
    warmup = 10

    def V(x):
        return x
        #return Variable(x)

    input = V(torch.cuda.FloatTensor(seq_len, batch_size, input_size).normal_())
    hx    = V(torch.cuda.FloatTensor(batch_size, hidden_size).normal_())
    cx    = V(torch.cuda.FloatTensor(batch_size, hidden_size).normal_())
    w_ih  = V(t_def(torch.cuda.FloatTensor(4 * hidden_size, input_size).normal_()))
    w_hh  = V(t_def(torch.cuda.FloatTensor(4 * hidden_size, hidden_size).normal_()))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(warmup + loops):
        gc.collect()
        start.record()
        start_cpu = time.time()  # high precision only for Linux
        for j in range(seq_len):
            hx, cx = lstm(input[j], (hx, cx), w_ih, w_hh)
        end_cpu = time.time()
        end.record()
        torch.cuda.synchronize()
        msecs = start.elapsed_time(end)
        flopc_per_cell = 201916416
        flopc = flopc_per_cell * seq_len
        flops = (flopc / (msecs / 1000)) / 1000000000000
        print("lstm({:2d}): {:8.3f} msecs ({:8.3f} msecs cpu; {:8.3f} TFLOPS)".format(i, msecs, (end_cpu-start_cpu)*1000, flops))

if __name__ == "__main__":
    main()
