import os
from pathlib import Path
import subprocess
import torch

CUDA_HOME = "/usr/local/cuda" if not "CUDA_HOME" in os.environ else os.environ["CUDA_HOME"]
REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent.parent
FBGEMM_PATH = REPO_PATH.joinpath("submodules", "FBGEMM", "fbgemm_gpu")
FBGEMM_CUTLASS_PATH = FBGEMM_PATH.parent.joinpath("third_party", "cutlass")
COLFAX_CUTLASS_PATH = REPO_PATH.joinpath("submodules", "cutlass-kernels")
COLFAX_CUTLASS_TRITONBENCH_PATH = REPO_PATH.joinpath("userbenchmark", "triton", "cutlass_kernels")

TORCH_BASE_PATH = Path(torch.__file__).parent

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
    "-D_GLIBCXX_USE_CXX11_ABI=0",
]
COMPILER_FLAGS = [
    f"-I{str(COLFAX_CUTLASS_PATH.joinpath('lib').resolve())}",
    f"-I{str(COLFAX_CUTLASS_PATH.joinpath('include').resolve())}",
    f"-I{str(FBGEMM_CUTLASS_PATH.joinpath('include').resolve())}",
    f"-I{str(FBGEMM_CUTLASS_PATH.joinpath('examples', 'commmon').resolve())}",
    f"-I{str(FBGEMM_CUTLASS_PATH.joinpath('tools', 'util', 'include').resolve())}",
    f"-I{CUDA_HOME}/include",
    f"-I{str(TORCH_BASE_PATH.joinpath('include').resolve())}",
    f"-I{str(COLFAX_CUTLASS_TRITONBENCH_PATH.joinpath('include').resolve())}",
    f"-Wl,-rpath,'{CUDA_HOME}/lib64'",
    f"-Wl,-rpath,'{CUDA_HOME}/lib'",
]
LINKER_FLAGS = [
    "--shared",
    "-fPIC",
    f"-L{str(TORCH_BASE_PATH.joinpath('lib').resolve())}",
    "-ltorch",
    "-ltorch_cuda",
    "-lc10",
    "-lc10_cuda",
    "-lcuda",
    "-lcudadevrt",
    "-lcudart_static",
    "-lcublas",
    "-lrt",
    "-lpthread",
    "-ldl",
]
FMHA_SOURCES = [
    # Source 1
    f"{str(COLFAX_CUTLASS_PATH.joinpath('src', 'fmha', 'fmha_forward.cu').resolve())}",
    # Source 2
    f"{str(COLFAX_CUTLASS_TRITONBENCH_PATH.joinpath('src', 'fmha', 'register_op.cu').resolve())}",
    "-o",
    "fmha_forward_lib.so",
]


def test_colfax_cutlass(colfax_cutlass_lib: str):
    assert os.path.exists(colfax_cutlass_lib), \
        f"{colfax_cutlass_lib} should exist as the built cutlass kernel."
    torch.ops.load_library(colfax_cutlass_lib)

def install_colfax_cutlass():
    # compile colfax_cutlass kernels
    output_dir = COLFAX_CUTLASS_TRITONBENCH_PATH.joinpath(".data")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["nvcc"]
    cmd.extend(COMPILER_FLAGS)
    cmd.extend(NVCC_FLAGS)
    cmd.extend(FMHA_SOURCES)
    cmd.extend(LINKER_FLAGS)
    print(" ".join(cmd))
    print(str(output_dir.resolve()))
    subprocess.check_call(cmd, cwd=str(output_dir.resolve()))
    colfax_cutlass_lib = str(output_dir.joinpath(FMHA_SOURCES[-1]).resolve())
    test_colfax_cutlass(colfax_cutlass_lib)
