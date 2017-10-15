import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import gc
import time
import warnings
import os

def mlstm(input, hx, cx, w_xm, w_hm, w_ih, w_mh):
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

    batch_size = 1
    input_size = 205
    hidden_size = 1900
    embed_size = hidden_size

    iters = 200
    loops = 30
    warmup = 10

    input = Variable(torch.cuda.FloatTensor(batch_size, input_size).normal_())
    hx    = Variable(torch.cuda.FloatTensor(batch_size, hidden_size).normal_())
    cx    = Variable(torch.cuda.FloatTensor(batch_size, hidden_size).normal_())
    w_xm  = Variable(torch.cuda.FloatTensor(embed_size, input_size).normal_())
    w_hm  = Variable(torch.cuda.FloatTensor(embed_size, hidden_size).normal_())
    w_ih  = Variable(torch.cuda.FloatTensor(4 * hidden_size, input_size).normal_())
    w_mh  = Variable(torch.cuda.FloatTensor(4 * hidden_size, embed_size).normal_())

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(warmup + loops):
        gc.collect()
        start.record()
        start_cpu = time.time()  # high precision only for Linux
        for j in range(iters):
            hx, cx = mlstm(input, hx, cx, w_xm, w_hm, w_ih, w_mh)
        end_cpu = time.time()
        end.record()
        torch.cuda.synchronize()
        msecs = start.elapsed_time(end)
        print("mlstm({:2d}): {:8.3f} msecs ({:8.3f} msecs cpu)".format(i, msecs, (end_cpu-start_cpu)*1000))

if __name__ == "__main__":
    main()
