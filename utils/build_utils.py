"""
Utilities for building pytorch and torch* domain packages
"""
import os
import sys
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

CLEANUP_ROUND = 5

@dataclass
class TorchRepo:
    name: str
    origin_url: str
    main_branch: str
    src_path: Path
    cur_commit: str
    build_command: List[str]

def setup_bisection_build_env(env: Dict[str, str]) -> Dict[str, str]:
    env["USE_CUDA"] = "1"
    env["BUILD_CAFFE2_OPS"] = "0"
    # Do not build the test
    env["BUILD_TEST"] = "0"
    env["USE_MKLDNN"] = "1"
    env["USE_MKL"] = "1"
    env["USE_CUDNN"] = "1"
    # Do not depend on ffmpeg, which requires conda-forge
    env["USE_FFMPEG"] = "0"
    # Torchaudio SOX build has failures, skip it
    env["BUILD_SOX"] = "0"
    # Disable Torchaudio KALDI build
    env["BUILD_KALDI"] = "0"
    env["CMAKE_PREFIX_PATH"] = env["CONDA_PREFIX"]
    return env

def _print_info(info: str):
    print(f"===========================   {info}   ===========================", flush=True)

def build_pytorch_repo(repo: TorchRepo, build_env: Dict[str, str]):
    # Check if version.py exists, if it does, remove it.
    # This is to force pytorch to update the version.py file upon incremental compilation
    version_py_path = os.path.join(repo.src_path.absolute(), "torch/version.py")
    if os.path.exists(version_py_path):
        os.remove(version_py_path)
    try:
        subprocess.check_call(repo.build_command, cwd=repo.src_path.absolute(), env=build_env)
        command_testbuild = [sys.executable, "-c", "'import torch'"]
        subprocess.check_call(command_testbuild, cwd=os.environ["HOME"], env=build_env)
    except subprocess.CalledProcessError:
        _print_info(f"BUILDING {repo.name.upper()} commit {repo.cur_commit} 2ND TRY")
        # Remove the build directory, then try building it again
        build_path = os.path.join(repo.src_path.absolute(), "build")
        if os.path.exists(build_path):
            shutil.rmtree(build_path)
        subprocess.check_call(repo.build_command, cwd=repo.src_path.absolute(), env=build_env)

def build_repo(repo: TorchRepo, build_env: Dict[str, str]):
    _print_info(f"BUILDING {repo.name.upper()} commit {repo.cur_commit} START")
    if repo.name == "pytorch":
        build_pytorch_repo(repo, build_env)
    else:
        subprocess.check_call(repo.build_command, cwd=repo.src_path, env=build_env)
    _print_info(f"BUILDING {repo.name.upper()} commit {repo.cur_commit} END")

def cleanup_torch_packages(pkgs: List[str]=[]):
    if not len(pkgs):
        pkgs = ["torch", "torchvision", "torchaudio", "torchdata"]
    for _ in range(CLEANUP_ROUND):
        command = "pip uninstall -y " + " ".join(pkgs) + " || true"
        subprocess.check_call(command, shell=True)
