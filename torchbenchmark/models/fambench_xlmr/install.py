import os
import sys
import torch
import subprocess
from torchbenchmark import REPO_PATH

def update_fambench_submodule():
    "Update FAMBench submodule of the benchmark repo"
    update_command = ["git", "submodule", "update", 
                      "--init", "--recursive", os.path.join("submodules","FAMBench")]
    subprocess.check_call(update_command, cwd=REPO_PATH)

def pip_install_requirements():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
        # pin fairseq version to 0.12.2
        # ignore deps specified in requirements.txt
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'fairseq==0.12.2'])
    except subprocess.CalledProcessError:
        # We ignore the ResolutionImpossible error because fairseq requires omegaconf < 2.1
        # but detectron2 requires omegaconf >= 2.1
        pass

if __name__ == "__main__":
    update_fambench_submodule()
    pip_install_requirements()
