import os
import itertools
import time
import sys
import logging
import subprocess
from glob import glob

PY2 = sys.version_info[0] == 2
timer = time.time if PY2 else time.perf_counter
cpu_timer = time.clock


def grid(d):
    """
    Creates a grid of arguments and calls benchmark with each.
    The arguments must be primited types such as strings or
    numbers and will be shallow(!) copied as part of the setup
    """
    result = []
    keys = list(d.keys())
    values = d.values()
    values = list(itertools.product(*values))
    for value in values:
        arg = {}
        for i, k in enumerate(keys):
            arg[keys[i]] = value[i]
        result.append(arg)
    return result


def show_cpu_info():
    p = subprocess.check_output(
        "cat /proc/cpuinfo | grep name | head -n 1", shell=True
    )
    device_name = str(p).split(":")[1][:-3]
    return "CPU info: %s" % (device_name)


def show_gpu_info():
    p = subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv", shell=True
    )
    device_name = str(p).split("\\n")[1]
    return "GPU info: %s" % (device_name)


def get_cpu_list():
    cpus = []
    for cpu in glob("/sys/devices/system/cpu/cpu*"):
        # filename should end in numbers
        try:
            num = int(os.path.basename(cpu)[3:])
            cpus.append(num)
        except ValueError:
            continue
    return cpus


def check_cpu_governor(cpu):
    logger = logging.getLogger()
    fp = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor".format(cpu)
    fpd = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_driver".format(cpu)
    try:
        with open(fp, "r") as f, open(fpd, "r") as fd:
            gov = f.read().rstrip()
            driver = fd.read().rstrip()
            if gov != "performance" and driver != "intel_pstate":
                logger.warning(
                    "CPU {} governor is {} which could lead to"
                    " variance in performance\n"
                    "Run 'echo performance > {}' as root to turn"
                    " off power scaling.".format(cpu, gov, fp)
                )
    except IOError as e:
        logger.warning(
            "Could not find CPU {} governor information in filesystem"
            " (are you running on Linux?)\n"
            "The file '{}' is not readable.\n"
            "More information:\n\n{}".format(fp, e)
        )
