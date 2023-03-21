import os
import re
import argparse
import subprocess
from pathlib import Path

# defines the default CUDA version to compile against
DEFAULT_CUDA_VERSION = "11.7"

CUDA_VERSION_MAP = {
    "11.3": {
        "pytorch_url": "cu113",
        "magma_version": "magma-cuda113",
    },
    "11.6": {
         "pytorch_url": "cu116",
         "magma_version": "magma-cuda116",
    },
    "11.7": {
         "pytorch_url": "cu117",
         "magma_version": "magma-cuda117",
    },
    "11.8": {
         "pytorch_url": "cu118",
         "magma_version": "magma-cuda118",
    }
}
PIN_CMAKE_VERSION = "3.22.*"

def _nvcc_output_match(nvcc_output, target_cuda_version):
    regex = 'release (.*),'
    version = re.search(regex, nvcc_output).groups()[0]
    return version == target_cuda_version

def prepare_cuda_env(cuda_version: str, dryrun=False):
    assert cuda_version in CUDA_VERSION_MAP, f"Required CUDA version {cuda_version} doesn't exist in {CUDA_VERSION_MAP.keys()}."
    env = os.environ.copy()
    # step 1: setup CUDA path and environment variables
    cuda_path = Path("/").joinpath("usr", "local", f"cuda-{cuda_version}")
    assert cuda_path.exists() and cuda_path.is_dir(), f"Expected CUDA Library path {cuda_path} doesn't exist."
    cuda_path_str = str(cuda_path.resolve())
    env["CUDA_ROOT"] = cuda_path_str
    env["CUDA_HOME"] = cuda_path_str
    env["PATH"] = f"{cuda_path_str}/bin:{env['PATH']}"
    env["CMAKE_CUDA_COMPILER"] = str(cuda_path.joinpath('bin', 'nvcc').resolve())
    env["LD_LIBRARY_PATH"] = f"{cuda_path_str}/lib64:{cuda_path_str}/extras/CUPTI/lib64:{env['LD_LIBRARY_PATH']}"
    if dryrun:
        print(f"CUDA_HOME is set to {env['CUDA_HOME']}")
    # step 2: test call to nvcc to confirm the version is correct
    test_nvcc = ["nvcc", "--version"]
    if dryrun:
        print(f"Checking nvcc version, command {test_nvcc}")
    else:
        output = subprocess.check_output(test_nvcc, stderr=subprocess.STDOUT, env=env).decode()
        print(f"NVCC version output: {output}")
        assert _nvcc_output_match(output, cuda_version), f"Expected CUDA version {cuda_version}, getting nvcc test result {output}"
    # step 3: install the correct magma version
    install_magma_cmd = ["conda", "install", "-c", "pytorch", CUDA_VERSION_MAP[cuda_version]['magma_version']]
    if dryrun:
        print(f"Installing CUDA magma: {install_magma_cmd}")
    subprocess.check_call(install_magma_cmd, env=env)
    return env

def setup_cuda_softlink(cuda_version: str):
    assert cuda_version in CUDA_VERSION_MAP, f"Required CUDA version {cuda_version} doesn't exist in {CUDA_VERSION_MAP.keys()}."
    cuda_path = Path("/").joinpath("usr", "local", f"cuda-{cuda_version}")
    assert cuda_path.exists() and cuda_path.is_dir(), f"Expected CUDA Library path {cuda_path} doesn't exist."
    current_cuda_path = Path("/").joinpath("usr", "local", "cuda")
    if current_cuda_path.exists():
        assert current_cuda_path.is_symlink(), f"Expected /usr/local/cuda to be a symlink."
        current_cuda_path.unlink()
    os.symlink(str(cuda_path.resolve()), str(current_cuda_path.resolve()))

def install_pytorch_nightly(cuda_version: str, env, dryrun=False):
    uninstall_torch_cmd = ["pip", "uninstall", "-y", "torch", "torchvision", "torchtext", "torchaudio"]
    if dryrun:
        print(f"Uninstall pytorch: {uninstall_torch_cmd}")
    else:
        # uninstall multiple times to make sure the env is clean
        for _loop in range(3):
            subprocess.check_call(uninstall_torch_cmd)
    pytorch_nightly_url = f"https://download.pytorch.org/whl/nightly/{CUDA_VERSION_MAP[cuda_version]['pytorch_url']}/torch_nightly.html"
    install_torch_cmd = ["pip", "install", "--pre", "torch", "torchvision", "torchtext", "torchaudio", "-f",  pytorch_nightly_url]
    if dryrun:
        print(f"Install pytorch nightly: {install_torch_cmd}")
    else:
        subprocess.check_call(install_torch_cmd, env=env)

def install_torch_deps(cuda_version: str):
    # install magma
    magma_pkg = CUDA_VERSION_MAP[cuda_version]["magma_version"]
    cmd = ["conda", "install", "-y", magma_pkg, "-c", "pytorch"]
    subprocess.check_call(cmd)
    # install other dependencies
    torch_deps = ["numpy", "requests", "ninja", "pyyaml", "setuptools", "gitpython", "beautifulsoup4", "regex"]
    cmd = ["conda", "install", "-y"] + torch_deps
    subprocess.check_call(cmd)
    # install unittest-xml-reporting
    cmd = ["pip", "install", "unittest-xml-reporting"]
    subprocess.check_call(cmd)

def install_torch_build_deps(cuda_version: str):
    install_torch_deps(cuda_version=cuda_version)
    # Pin cmake version to stable
    # See: https://github.com/pytorch/builder/pull/1269
    torch_build_deps = ["cffi", "sympy", "typing_extensions", "future", "six", "dataclasses", "tabulate", "tqdm", "mkl", "mkl-include", \
                        f"cmake={PIN_CMAKE_VERSION}"]
    cmd = ["conda", "install", "-y"] + torch_build_deps
    subprocess.check_call(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaver", default=DEFAULT_CUDA_VERSION, help="Specify the default CUDA version")
    parser.add_argument("--setup-cuda-softlink", action="store_true", help="Setup the softlink to /usr/local/cuda")
    parser.add_argument("--install-torch-deps", action="store_true", help="Install pytorch runtime requirements")
    parser.add_argument("--install-torch-build-deps", action="store_true", help="Install pytorch build requirements")
    parser.add_argument("--install-torch-nightly", action="store_true", help="Install pytorch nightlies")
    args = parser.parse_args()
    if args.setup_cuda_softlink:
        setup_cuda_softlink(cuda_version=args.cudaver)
    if args.install_torch_deps:
        install_torch_deps(cuda_version=args.cudaver)
    if args.install_torch_build_deps:
        install_torch_build_deps(cuda_version=args.cudaver)
    if args.install_torch_nightly:
        install_pytorch_nightly(cuda_version=args.cudaver, env=os.environ)
