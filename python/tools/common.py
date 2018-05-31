import os
import warnings
import sys
from itertools import product
import time

PY2 = sys.version_info[0] == 2


def init(cpu, gpu, skip_cpu_governor_check=False):
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
    print("{}({:2d}): {:8.3f} usecs ({:8.3f} usecs cpu)".format(name, i, gpu_usecs/divide_by, cpu_usecs/divide_by, file=sys.stderr))

class over(object):
    def __init__(self, *args):
        self.values = args


class AttrDict(dict):
    def __repr__(self):
        return ', '.join(k + '=' + str(v) for k, v in self.items())

    def __getattr__(self, name):
        return self[name]


def make_params(**kwargs):
    keys = list(kwargs.keys())
    iterables = [kwargs[k].values if isinstance(kwargs[k], over) else (kwargs[k],) for k in keys]
    all_values = list(product(*iterables))
    param_dicts = [AttrDict({k: v for k, v in zip(keys, values)}) for values in all_values]
    return [param_dicts]


class Benchmark(object):
    timer = time.time if PY2 else time.perf_counter
    default_params = []
    params = make_params()
    param_names = ['config']

    # NOTE: subclasses should call prepare instead of setup
    def setup(self, params):
        for k, v in self.default_params.items():
            params.setdefault(k, v)
        self.prepare(params)
