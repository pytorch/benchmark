import os
from utils.python_utils import pip_install_requirements
from torchbenchmark.util.framework.huggingface.patch_hf import patch_transformers, cache_model

if __name__ == '__main__':
    pip_install_requirements()
    patch_transformers()
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    cache_model(model_name)