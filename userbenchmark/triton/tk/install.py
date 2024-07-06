import os
from pathlib import Path
import subprocess
import torch

CUDA_HOME = "/usr/local/cuda" if not "CUDA_HOME" in os.environ else os.environ["CUDA_HOME"]
REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent.parent
TK_PATH = REPO_PATH.joinpath("submodules", "ThunderKittens")
TRITONBENCH_TK_PATH = REPO_PATH.joinpath("userbenchmark", "triton", "tk")

NVCC_GENCODE = "-gencode=arch=compute_90a,code=[sm_90a]"

NVCC_FLAGS = [
    NVCC_GENCODE,
    "-DNDEBUG",
    "-Xcompiler=-fPIE",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "--use_fast_math",
    "-forward-unknown-to-host-compiler",
    "-O3",
    "-Xnvlink=--verbose",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
    "--std=c++20",
    "-MD",
    "-MT",
    "-MF",
    "-x",
    "cu",
    "-DKITTENS_HOPPER",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
]
LINKER_FLAGS = [
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
TK_ATTN_H100_FWD_SOURCES = [
    # Source 1
    f"{str(TK_PATH.joinpath('examples', 'attn', 'h100', 'h100_fwd.cu').resolve())}",
    # Source 2
    f"{str(TRITONBENCH_TK_PATH.joinpath('src', 'attn_h100_fwd', 'register_op.cu').resolve())}",
    "-o",
    "attn_h100_fwd.so",
]


def test_tk_attn_h100_fwd(tk_attn_h100_fwd_lib: str):
    assert os.path.exists(tk_attn_h100_fwd_lib), \
        f"{tk_attn_h100_fwd_lib} should exist as the built cutlass kernel."
    torch.ops.load_library(tk_attn_h100_fwd_lib)


def install_tk():
    # compile thunderkitten kernels
    output_dir = TRITONBENCH_TK_PATH.joinpath(".data")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["nvcc"]
    cmd.extend(NVCC_FLAGS)
    cmd.extend(TK_ATTN_H100_FWD_SOURCES)
    cmd.extend(LINKER_FLAGS)
    print(" ".join(cmd))
    print(str(output_dir.resolve()))
    subprocess.check_call(cmd, cwd=str(output_dir.resolve()))
    tk_attn_h100_fwd_lib = str(output_dir.joinpath(TK_ATTN_H100_FWD_SOURCES[-1]).resolve())
    test_tk_attn_h100_fwd(tk_attn_h100_fwd_lib)
