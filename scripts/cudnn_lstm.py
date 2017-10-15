import torch
from torch.autograd import Variable
import torch.jit
import torch.nn

import gc
import time
import warnings
import os


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
    layers = 1

    seq_len = 512
    loops = 30
    warmup = 10

    def V(x):
        return Variable(x)

    input = V(torch.cuda.FloatTensor(seq_len, batch_size, input_size).normal_())
    hx    = V(torch.cuda.FloatTensor(layers, batch_size, hidden_size).normal_())
    cx    = V(torch.cuda.FloatTensor(layers, batch_size, hidden_size).normal_())

    lstm = torch.nn.LSTM(input_size, hidden_size, layers).cuda()
    lstm.flatten_parameters()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(warmup + loops):
        gc.collect()
        start.record()
        start_cpu = time.time()  # high precision only for Linux
        lstm(input, (hx, cx))
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
