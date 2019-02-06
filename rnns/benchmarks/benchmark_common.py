import os
import warnings
import sys

# Copied and pasted from benchmark_common under benchmark/scripts


def benchmark_init(cpu, gpu, skip_cpu_governor_check=False):
    cpu_pin(cpu)
    if not skip_cpu_governor_check:
        check_cpu_governor(cpu)


# NB: Be careful with this when benchmarking backward; backward
# uses multiple threads
def cpu_pin(cpu):
    if not getattr(os, 'sched_setaffinity'):
        warnings.warn("Could not pin to CPU {}; try pinning with 'taskset 0x{:x}'".format(cpu, 1 << cpu))
    else:
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


def print_results_usecs(name, i, gpu_usecs, cpu_usecs, divide_by):
    print("{}({:2d}): {:8.3f} usecs ({:8.3f} usecs cpu)".format(
        name, i, gpu_usecs / divide_by, cpu_usecs / divide_by, file=sys.stderr))
