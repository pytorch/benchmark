"""
Utilities for building pytorch and torch* domain packages
"""
import subprocess
from pathlib import Path
from typing import Dict


def setup_bisection_build_env(env: Dict[str, str]) -> Dict[str, str]:
    env["USE_CUDA"] = "1"
    env["BUILD_CAFFE2_OPS"] = "0"
    # Do not build the test
    env["BUILD_TEST"] = "0"
    env["USE_MKLDNN"] = "1"
    env["USE_MKL"] = "1"
    env["USE_CUDNN"] = "1"
    env["USE_FFMPEG"] = "1"
    # Torchaudio SOX build has failures, skip it
    env["BUILD_SOX"] = "0"
    env["CMAKE_PREFIX_PATH"] = env["CONDA_PREFIX"]
    return env


def _print_info(info: str):
    print(f"===========================   {info}   ===========================", flush=True)

def build_pytorch(src_path: Path, build_env: Dict[str, str]):
    _print_info("BUILDING PYTORCH START")
    command = ["python", "setup.py", "install"]
    subprocess.check_call(command, cwd=src_path, env=build_env)
    _print_info("BUILDING PYTORCH END")

def build_torchdata(src_path: Path, build_env: Dict[str, str]):
    _print_info("BUILDING TORCHDATA START")
    command = ["python", "setup.py", "install"]
    subprocess.check_call(command, cwd=src_path, env=build_env)
    _print_info("BUILDING TORCHDATA END")

def build_torchvision(src_path: Path, build_env: Dict[str, str]):
    _print_info("BUILDING TORCHVISION START")
    command = ["python", "setup.py", "install"]
    subprocess.check_call(command, cwd=src_path, env=build_env)
    _print_info("BUILDING TORCHVISION END")

def build_torchtext(src_path: Path, build_env: Dict[str, str]):
    _print_info("BUILDING TORCHTEXT START")
    command = ["python", "setup.py", "clean", "install"]
    subprocess.check_call(command, cwd=src_path, env=build_env)
    _print_info("BUILDING TORCHTEXT END")

def build_torchaudio(src_path: Path, build_env: Dict[str, str]):
    _print_info("BUILDING TORCHAUDIO START")
    command = ["python", "setup.py", "clean", "develop"]
    subprocess.check_call(command, cwd=src_path, env=build_env)
    _print_info("BUILDING TORCHAUDIO END")
