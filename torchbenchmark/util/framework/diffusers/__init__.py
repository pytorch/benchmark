import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


def pip_install_requirements():
    requirements_file = os.path.join(CURRENT_DIR, "requirements.txt")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_file]
    )


def install_diffusers():
    pip_install_requirements()
