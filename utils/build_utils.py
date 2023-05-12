"""
Utilities for building pytorch and torch* domain packages
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

FIRST_TIME_INSTALL_TORCHBENCH = True

@dataclass
class TorchRepo:
    name: str
    src_path: Path
    cur_commit: str
    main_branch: str
    build_command: List[str]

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

def build_repo(repo: TorchRepo, build_env: Dict[str, str]):
    if not repo.name == "torchbench":
        _print_info(f"BUILDING {repo.name.upper()} START")
        subprocess.check_call(repo.build_command, cwd=repo.src_path, env=build_env)
        _print_info(f"BUILDING {repo.name.upper()} END")
    elif FIRST_TIME_INSTALL_TORCHBENCH:
        _print_info(f"BUILDING {repo.name.upper()} START")
        subprocess.check_call(repo.build_command, cwd=repo.src_path, env=build_env)
        FIRST_TIME_INSTALL_TORCHBENCH = False
        _print_info(f"BUILDING {repo.name.upper()} END")
