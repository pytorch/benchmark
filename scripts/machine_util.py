"""Utilities for tuning the machine for better benchmark stability.

Written for Amazon linux and Intel CPU, Nvidia GPU althogh many utilities will overlap.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def read_sys_file(sysfile: Path):
    with open(sysfile, 'r') as f:
        return f.read()

def write_sys_file(sysfile: Path, content: str):
    print(f"Write {content} to {sysfile}")
    with open(sysfile, 'w') as f:
        f.write(content)

def check_intel_turbo_state(turbo_file='/sys/devices/system/cpu/intel_pstate/no_turbo'):
    return int(read_sys_file(turbo_file))

def set_intel_turbo_state(state: int, turbo_file='/sys/devices/system/cpu/intel_pstate/no_turbo'):
    assert state in [0, 1]
    write_sys_file(turbo_file, str(state))

def parse_lscpu_cpu_core_list():
    coreinfo = subprocess.check_output("lscpu --all --parse=CPU,CORE,ONLINE", shell=True).strip().decode().split('\n')
    matched_cpus = 0
    cpu_core = []
    for line in coreinfo[2:]:
        if line[0] == '#':
            continue
        cpu, core, online = line.split(',')
        cpu = int(cpu)
        online = online == "Y"
        core = int(core) if online else None
        if cpu == core:
            matched_cpus += 1
        cpu_core.append((cpu, core, online))
    assert matched_cpus > 0, "Failed to parse lscpu output"
    return cpu_core


def hyper_threading_enabled():
    for cpu, core, online in parse_lscpu_cpu_core_list():
        if cpu != core and online:
            return True
    return False

def set_hyper_threading(enabled=False):
    for cpu, core, online in parse_lscpu_cpu_core_list():
        if cpu != core:
            if not online and not enabled:
                continue
            if online and enabled:
                continue
            virtual_cpu_online_file = f"/sys/devices/system/cpu/cpu{cpu}/online"
            value = "1" if enabled else "0"
            write_sys_file(virtual_cpu_online_file, value)

def get_intel_max_cstate():
    kernel_args = read_sys_file('/proc/cmdline').split()
    for arg in kernel_args:
        if arg.find('intel_idle.max_cstate') == 0:
            return int(arg.split('=')[1])
    return None

def get_isolated_cpus():
    """
    Returns a list of cpus marked as isolated from the kernel scheduler for regular tasks.
    Only tasks scheduled via taskset command can use these cpus, e.g. benchmarking workload.
    """
    kernel_args = read_sys_file('/proc/cmdline').split()
    isolcpus = set()
    for arg in kernel_args:
        if arg.find('isolcpus') == 0:
            arg = arg.split('=')[1]
            chunks = arg.split(',')
            for chunk in chunks:
                if '-' in chunk:
                    start, end = chunk.split('-')
                    for cpu in range(start, end+1):
                        isolcpus.add(cpu)
                else:
                    isolcpus.add(int(chunk))
    return list(isolcpus)

def get_nvidia_graphics_clock(device_id=0):
    clock = subprocess.check_output(f'nvidia-smi --query-gpu="clocks.applications.graphics" -i {device_id} --format=csv,noheader,nounits',
                                       shell=True).strip().decode()
    clock = int(clock)
    return clock

def set_nvidia_graphics_clock(device_id=0, clock=900):
    # nvidia-smi -ac 5001,900 
    raise NotImplementedError("TODO, wrap the above call and check for error")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable_ht", action="store_true", help="Enable HyperThreading")
    parser.add_argument("--configure", action="store_true", help="Apply benchmark tuning to this machine")
    parser.add_argument("--no_verify", action="store_true", help="Skip verifying machine is configured for benchmarking")
    args = parser.parse_args()


    if args.enable_ht:
        set_hyper_threading(True)

    if args.configure:
        set_intel_turbo_state(0)
        set_hyper_threading(False)
        set_nvidia_graphics_clock()

    # if args.verify:
    if not args.no_verify:
        assert 0 == check_intel_turbo_state(), "Turbo Boost is not disabled"
        assert False == hyper_threading_enabled(), "HyperThreading is not disabled"
        assert 1 == get_intel_max_cstate(), "Intel max C-State isn't set to 1, which avoids power-saving modes."
        assert len(get_isolated_cpus()) > 0, "No cpus are isolated for benchmarking with isolcpus"
        assert 900 == get_nvidia_graphics_clock(), "Nvidia gpu clock isn't limited, to increase consistency by reducing throttling"
