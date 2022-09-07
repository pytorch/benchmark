import os
from pathlib import Path
import subprocess
import sys


def check_data_dir():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    tacotron2_data_dir = os.path.join(current_dir.parent.parent, "data", ".data", "tacotron2-minimal")
    assert os.path.exists(tacotron2_data_dir), "Couldn't find tacotron2 minimal data dir, please run install.py again."


def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
