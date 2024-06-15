import argparse
import subprocess
import sys
import os
from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
FBGEMM_PATH = REPO_PATH.joinpath("submodules", "FBGEMM", "fbgemm_gpu")
CUDA_HOME = "/usr/local/cuda" if not "CUDA_HOME" in os.environ else os.environ["CUDA_HOME"]
FBGEMM_CUTLASS_PATH = FBGEMM_PATH.parent.joinpath("third_party", "cutlass")
COLFAX_CUTLASS_PATH = REPO_PATH.joinpath("submodules", "cutlass-kernels")
COLFAX_CUTLASS_TRITONBENCH_PATH = REPO_PATH.joinpath("userbenchmark", "triton", "cutlass-kernel")

NVCC_GENCODE = "-gencode=arch=compute_90a,code=[sm_90a]"

NVCC_FLAGS = [
    NVCC_GENCODE,
    "--use_fast_math",
    "-forward-unknown-to-host-compiler",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-forward-unknown-to-host-compiler",
    "--use_fast_math",
    "-Xcompiler=-fno-strict-aliasing",
    "-Xcompiler=-fPIE",
    "-Xcompiler=-lcuda",
    "-DNDEBUG",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
]
PREPROCESSOR_FLAGS = [
    f"-I{str(COLFAX_CUTLASS_PATH.joinpath('lib').resolve())}",
    f"-I{str(COLFAX_CUTLASS_PATH.joinpath('include').resolve())}",
    f"-I{str(FBGEMM_CUTLASS_PATH.joinpath('include').resolve())}",
    f"-I{str(FBGEMM_CUTLASS_PATH.joinpath('examples', 'commmon').resolve())}",
    f"-I{str(FBGEMM_CUTLASS_PATH.joinpath('tools', 'util', 'include').resolve())}",
    f"-I{CUDA_HOME}/include",
    f"-Wl,-rpath,'{CUDA_HOME}/lib64'",
    f"-Wl,-rpath,'{CUDA_HOME}/lib'"
]
FMHA_SOURCES = [
    # Source 1
    f"{str(COLFAX_CUTLASS_PATH.joinpath('src', 'fmha', 'fmha_forward.cu').resolve())}",
    # Source 2
    f"{str(COLFAX_CUTLASS_TRITONBENCH_PATH.joinpath('src', 'fmha', 'register_op.cu').resolve())}",
    "-o",
    "fmha_forward_lib",
]

def install_fbgemm():
    cmd = ["pip", "install", "-r", "requirements.txt"]
    subprocess.check_call(cmd, cwd=str(FBGEMM_PATH.resolve()))
    # Build target A100(8.0) or H100(9.0, 9.0a)
    cmd = [sys.executable, "setup.py", "bdist_wheel", "--package_variant=genai", "-DTORCH_CUDA_ARCH_LIST=8.0;9.0;9.0a"]
    subprocess.check_call(cmd, cwd=str(FBGEMM_PATH.resolve()))

def test_fbgemm():
    cmd = [sys.executable, "-c", '"import fbgemm_gpu.experimental.gen_ai"']
    subprocess.check_call(cmd)

def install_cutlass():
    # compile colfax_cutlass kernels
    output_dir = COLFAX_CUTLASS_TRITONBENCH_PATH.joinpath(".data")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["nvcc"]
    cmd.extend(PREPROCESSOR_FLAGS)
    cmd.extend(NVCC_FLAGS)
    cmd.extend(FMHA_SOURCES)
    print(" ".join(cmd))
    print(str(output_dir.resolve()))
    subprocess.check_call(cmd, cwd=str(output_dir.resolve()))
    return str(output_dir.joinpath(FMHA_SOURCES[-1]).resolve())

def test_cutlass():
    so_output = FMHA_SOURCES[-1]
    assert os.path.exists(so_output), f"{so_output} should exist as the built cutlass kernel."
    import torch
    torch.ops.load_library(so_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbgemm", action="store_true", help="Install FBGEMM GPU")
    parser.add_argument("--cutlass", action="store_true", help="Install optional CUTLASS kernels")
    args = parser.parse_args()
    if args.fbgemm:
        install_fbgemm()
        test_fbgemm()
    if args.cutlass:
        install_cutlass()
        test_cutlass()
