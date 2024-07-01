import argparse
import subprocess
import sys
import os
from pathlib import Path

from utils.cuda_utils import DEFAULT_CUDA_VERSION, CUDA_VERSION_MAP

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
FBGEMM_PATH = REPO_PATH.joinpath("submodules", "FBGEMM", "fbgemm_gpu")

def install_jax(cuda_version=DEFAULT_CUDA_VERSION):
    jax_package_name = CUDA_VERSION_MAP[cuda_version]["jax"]
    jax_nightly_html = "https://storage.googleapis.com/jax-releases/jax_nightly_releases.html"
    # install instruction:
    # https://jax.readthedocs.io/en/latest/installation.html
    # pip install -U --pre jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
    cmd = ["pip", "install", "--pre", jax_package_name, "-f", jax_nightly_html]
    subprocess.check_call(cmd)
    # Test jax installation
    test_cmd = [sys.executable, "-c", "import jax"]
    subprocess.check_call(test_cmd)

def install_fbgemm():
    cmd = ["pip", "install", "-r", "requirements.txt"]
    subprocess.check_call(cmd, cwd=str(FBGEMM_PATH.resolve()))
    # Build target A100(8.0) or H100(9.0, 9.0a)
    cmd = [sys.executable, "setup.py", "install", "--package_variant=genai", "-DTORCH_CUDA_ARCH_LIST=8.0;9.0;9.0a"]
    subprocess.check_call(cmd, cwd=str(FBGEMM_PATH.resolve()))

def test_fbgemm():
    print("Checking fbgemm_gpu installation...", end="")
    cmd = [sys.executable, "-c", "import fbgemm_gpu.experimental.gen_ai"]
    subprocess.check_call(cmd)
    print("OK")

def install_cutlass():
    try:
        from .cutlass_kernels.install import install_colfax_cutlass
    except ImportError:
        try:
            from cutlass_kernels.install import install_colfax_cutlass
        except ImportError:
            from userbenchmark.triton.cutlass_kernels.install import install_colfax_cutlass
    install_colfax_cutlass()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbgemm", action="store_true", help="Install FBGEMM GPU")
    parser.add_argument("--cutlass", action="store_true", help="Install optional CUTLASS kernels")
    parser.add_argument("--jax", action="store_true", help="Install jax nightly")
    parser.add_argument("--test", action="store_true", help="Run test")
    args = parser.parse_args()
    if args.fbgemm:
        if args.test:
            test_fbgemm()
        else:
            install_fbgemm()
    if args.cutlass and not args.test:
        install_cutlass()
    if args.jax and not args.test:
        install_jax()
