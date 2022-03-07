
import subprocess
import sys
from torchbenchmark.util.framework.huggingface.patch_hf import patch_transformers

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    patch_transformers()
