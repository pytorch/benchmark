import subprocess
import os.path
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def install_pytorch_geometric():
    pip_install_requirements()

def pip_install_requirements():
    requirements_file = os.path.join(CURRENT_DIR, "requirements.txt")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements_file])
