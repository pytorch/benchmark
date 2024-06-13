import os
from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
REPO_LIBRARY_PATH = REPO_PATH.joinpath("userbenchmark", "triton", ".data")

def load_library(library_path: str):
    import torch
    library_path = REPO_LIBRARY_PATH.joinpath(library_path).resolve()
    torch.ops.load_library(str(library_path))