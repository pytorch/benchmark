import os
import warnings
import sys


def init(cpu, gpu):
    #cpu_pin(cpu)
    check_cpu_governor(cpu)


# NB: Be careful with this when benchmarking backward; backward
# uses multiple threads
#
# WARNING: THIS DOESN'T WORK, THE PYTHON THREAD SPAWNS TOO LATE
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
    print("{}({:4}): {:8.3f} usecs ({:8.3f} usecs cpu)".format(name, i, gpu_usecs/divide_by, cpu_usecs/divide_by, file=sys.stderr))

## {{{ http://code.activestate.com/recipes/511478/ (r1)
import math
import functools

def percentile(N, percent, key=lambda x:x):
    """
    Find the percentile of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile of the values
    """
    if not N:
        return None
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0+d1

# median is 50th percentile.
median = functools.partial(percentile, percent=0.5)
## end of http://code.activestate.com/recipes/511478/ }}}


def print_stats_usecs(name, all_gpu_usecs, all_cpu_usecs, sample_size):
    assert len(all_gpu_usecs) == len(all_cpu_usecs)
    all_gpu_usecs.sort()
    all_cpu_usecs.sort()
    print("----", file=sys.stderr)
    # print_results_usecs(name, " avg",
    #                     sum(all_gpu_usecs), sum(all_cpu_usecs),
    #                     sample_size * len(all_gpu_usecs))
    def print_percentile(p):
        print_results_usecs(name, "{:3.0f}%".format(p * 100),
                            percentile(all_gpu_usecs, p),
                            percentile(all_cpu_usecs, p), sample_size)
    print_percentile(0)
    print_percentile(.25)
    print_percentile(.50)
    print_percentile(.75)
    print_percentile(.99)
    print_percentile(1)
