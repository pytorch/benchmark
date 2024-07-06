import os
from pathlib import Path
import subprocess
import torch
from utils import add_path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_HOME = "/usr/local/cuda" if not "CUDA_HOME" in os.environ else os.environ["CUDA_HOME"]
REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent.parent
TK_PATH = REPO_PATH.joinpath("submodules", "ThunderKittens")
TK_PYUTILS_PATH = TK_PATH.joinpath("src","common","pyutils")
TRITONBENCH_TK_PATH = REPO_PATH.joinpath("userbenchmark", "triton", "tk")
TORCH_BASE_PATH = Path(torch.__file__).parent


PKG_NAME_TO_SRC_NAME = {
    "tk_attn_h100_fwd": "h100_fwd",
}

def _sources(name):
    base_dir = TK_PATH.joinpath('examples', 'attn', 'h100')
    src_name = PKG_NAME_TO_SRC_NAME[name]
    frontend_cpp = str(base_dir.joinpath(f"{src_name}_frontend.cpp").resolve())
    cu = str(base_dir.joinpath(f"{src_name}.cu").resolve())
    return [frontend_cpp, cu]

import distutils.command.build

class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = str(TRITONBENCH_TK_PATH.joinpath(".data").resolve())

def cuda_extension(name, debug, gpu_type):
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
    return CUDAExtension(f'{name}',
                        sources=_sources(name),
                        extra_compile_args={'cxx' : ['-std=c++20'],
                                            'nvcc' : ['-O3'] + _cuda_flags},
                        libraries=['cuda'])

def test_tk_attn_h100_fwd():
    import tk_attn_h100_fwd
    tk_attn_h100_fwd.attention_forward


def install_tk():
    # compile thunderkitten kernels
    output_dir = TRITONBENCH_TK_PATH.joinpath(".data")
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "tk_attn_h100_fwd"
    gpu = "H100"
    debug = False
    cuda_ext = cuda_extension(name, debug, gpu)
    import sys
    backup_sys_argv = sys.argv.copy()
    sys.argv = ['install.py', 'install']
    setup(name=f"{name}",
          ext_modules=[cuda_ext],
          cmdclass={'build': BuildCommand, 'build_ext': BuildExtension})
    sys.argv = backup_sys_argv
    test_tk_attn_h100_fwd()
