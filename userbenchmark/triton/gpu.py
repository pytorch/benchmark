import logging
import os
import subprocess
from contextlib import contextmanager

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

POWER_LIMIT = {
    "NVIDIA PG509-210": "330",
    "NVIDIA A100": "330",
    "NVIDIA H100": "650",
}
FREQ_LIMIT = {
    "NVIDIA PG509-210": "1410",
    "NVIDIA A100": "1410",
    "NVIDIA H100": "1980",
}


def _set_pm():
    command = ["sudo", "nvidia-smi", "-i", CUDA_VISIBLE_DEVICES, "-pm", "1"]
    subprocess.check_call(command)


def _set_power(gpu_info: str):
    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        CUDA_VISIBLE_DEVICES,
        "--power-limit",
        POWER_LIMIT[gpu_info],
    ]
    subprocess.check_call(command)


def _set_clock(gpu_info: str):
    # lgc: lock gpu clocks
    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        CUDA_VISIBLE_DEVICES,
        "-lgc",
        FREQ_LIMIT[gpu_info],
    ]
    subprocess.check_call(command)


def _reset_clock(gpu_info: str):
    # rgc: reset gpu clocks
    command = ["sudo", "nvidia-smi", "-i", CUDA_VISIBLE_DEVICES, "-rgc"]
    subprocess.check_call(command)


def _get_gpu_name() -> str:
    import pynvml

    pynvml.nvmlInit()
    gpu_id = CUDA_VISIBLE_DEVICES.split(",")[0]
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
    return pynvml.nvmlDeviceGetName(handle).decode("utf-8")


@contextmanager
def gpu_lockdown(enabled=True):
    try:
        if enabled:
            logging.info(f"[tritonbench] Locking down GPU {CUDA_VISIBLE_DEVICES}")
            gpu_name = _get_gpu_name()
            assert gpu_name in POWER_LIMIT, f"Unsupported GPU {gpu_name}"
            _set_pm()
            _set_power(gpu_name)
            _set_clock(gpu_name)
        yield
    finally:
        if enabled:
            _reset_clock(gpu_name)
