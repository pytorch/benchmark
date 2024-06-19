import os.path
from utils.python_utils import pip_install_requirements

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def install_pytorch_geometric():
    pip_install_requirements(os.path.join(CURRENT_DIR, "requirements.txt"))
