import argparse
import subprocess
import sys

from torchbenchmark import REPO_PATH
FBGEMM_PATH = REPO_PATH.joinpath("submodules", "FBGEMM", "fbgemm_gpu")

def install_fbgemm():
    cmd = [sys.executable, "setup.py", "bdist_wheel", "--package_variant=genai"]
    subprocess.check_call(cmd, cwd=FBGEMM_PATH)

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
