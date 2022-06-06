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
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == "__main__":
    update_fambench_submodule()
    pip_install_requirements()
