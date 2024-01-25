import subprocess
import sys
import os
from torchbenchmark.util.framework.huggingface.patch_hf import patch_transformers, cache_model

if __name__ == '__main__':
    patch_transformers()
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    cache_model(model_name)
