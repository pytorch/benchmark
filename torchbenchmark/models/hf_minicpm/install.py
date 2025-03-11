import os
import sys
import subprocess

from torchbenchmark.util.framework.huggingface.patch_hf import (
    cache_model,
    patch_transformers,
)

def setup_install():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./requirements.txt"])

if __name__ == "__main__":
    patch_transformers()
    setup_install()
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    cache_model(model_name)
