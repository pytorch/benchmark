import os
from pathlib import Path
from utils.python_utils import pip_install_requirements

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

def install_diffusers():
    requirements_file = os.path.join(CURRENT_DIR, "requirements.txt")
    pip_install_requirements(requirements_txt=requirements_file)
