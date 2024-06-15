import os
from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
TRITONBENCH_PATH = REPO_PATH.joinpath("userbenchmark", "triton")

def load_library(library_path: str):
    import torch
    prefix, _delimiter, so_file = library_path.partition("/")
    so_full_path = TRITONBENCH_PATH.joinpath(prefix, ".data", so_file).resolve()
    torch.ops.load_library(str(so_full_path))
