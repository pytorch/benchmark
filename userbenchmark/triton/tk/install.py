import os
from pathlib import Path
import subprocess
import torch
from utils import add_path

CUDA_HOME = "/usr/local/cuda" if not "CUDA_HOME" in os.environ else os.environ["CUDA_HOME"]
REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent.parent
TK_PATH = REPO_PATH.joinpath("submodules", "ThunderKittens")
TK_PYUTILS_PATH = TK_PATH.joinpath("src","common","pyutils")
TRITONBENCH_TK_PATH = REPO_PATH.joinpath("userbenchmark", "triton", "tk")
TORCH_BASE_PATH = Path(torch.__file__).parent

def _sources():
    base_dir = TK_PATH.joinpath('examples', 'attn', 'h100')
    cu = str(base_dir.joinpath("h100_fwd.cu").resolve())
    return [cu, "-o", "tk_attn_h100_fwd.so"]

def cuda_extension(debug, gpu_type):
    _cuda_flags  = [
                    '--use_fast_math',
                    '--generate-line-info',
                    '--restrict', '-std=c++20',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-Xcompiler=-fno-strict-aliasing',
                    '-MD', '-MT', '-MF', '-x', 'cu', '-lrt', '-lpthread', '-ldl',
                    '-lcuda', '-lcudadevrt', '-lcudart_static', '-lcublas',
                    f"-I {str(TK_PATH.resolve())}"
                    ]

    if gpu_type == '4090':
        _cuda_flags.append('-DKITTENS_4090')
        _cuda_flags.append('-arch=sm_89')
    elif gpu_type == 'H100':
        _cuda_flags.append('-DKITTENS_HOPPER')
        _cuda_flags.append('-arch=sm_90a')
    elif gpu_type == 'A100':
        _cuda_flags.append('-DKITTENS_A100')
        _cuda_flags.append('-arch=sm_80')

    if(debug): _cuda_flags += ['-D__DEBUG_PRINT', '-g', '-G']
    return _cuda_flags


def test_tk_attn_h100_fwd(tk_lib):
    assert os.path.exists(tk_lib), \
        f"{tk_lib} should exist as the built cutlass kernel."
    torch.ops.load_library(tk_lib)    


def install_tk():
    # compile thunderkitten kernels
    output_dir = TRITONBENCH_TK_PATH.joinpath(".data")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["nvcc"]
    sources = _sources()
    cmd.extend(cuda_extension(debug=False, gpu_type="H100"))
    cmd.extend(sources)
    print(" ".join(cmd))
    print(str(output_dir.resolve()))
    subprocess.check_call(cmd, cwd=str(output_dir.resolve()))
    tk_lib = str(output_dir.joinpath(sources[-1]).resolve())
    test_tk_attn_h100_fwd(tk_lib)
