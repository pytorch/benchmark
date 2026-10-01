import os
import sys
import subprocess

from torchbenchmark.util.framework.huggingface.patch_hf import (
    cache_model,
    patch_transformers,
)
from utils.python_utils import pip_install_requirements

if __name__ == "__main__":
    patch_transformers()
    pip_install_requirements()
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    cache_model(model_name)
