import argparse
import subprocess
import sys
import os
from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
FBGEMM_PATH = REPO_PATH.joinpath("submodules", "FBGEMM", "fbgemm_gpu")

def install_fbgemm():
    cmd = ["pip", "install", "-r", "requirements.txt"]
    subprocess.check_call(cmd, cwd=str(FBGEMM_PATH.resolve()))
    # Build target A100(8.0) or H100(9.0, 9.0a)
    cmd = [sys.executable, "setup.py", "bdist_wheel", "--package_variant=genai", "-DTORCH_CUDA_ARCH_LIST=8.0;9.0;9.0a"]
    subprocess.check_call(cmd, cwd=str(FBGEMM_PATH.resolve()))

def test_fbgemm():
    cmd = [sys.executable, "-c", '"import fbgemm_gpu.experimental.gen_ai"']
    subprocess.check_call(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbgemm", action="store_true", help="Install FBGEMM GPU")
    args = parser.parse_args()
    if args.fbgemm:
        install_fbgemm()
        test_fbgemm()
