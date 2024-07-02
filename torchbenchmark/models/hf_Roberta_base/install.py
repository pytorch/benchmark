
import subprocess
import sys
import os
from torchbenchmark.util.framework.huggingface.patch_hf import patch_transformers, cache_model

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    patch_transformers()
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    cache_model(model_name)
