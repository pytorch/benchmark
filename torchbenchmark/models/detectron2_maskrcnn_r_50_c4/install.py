import os
import sys
from pathlib import Path
import subprocess

def check_data_dir():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    coco2017_data_dir = os.path.join(current_dir.parent.parent, "data", ".data", "coco2017-minimal")
    assert os.path.exists(coco2017_data_dir), "Couldn't find coco2017 minimal data dir, please run install.py again."

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    check_data_dir()
    pip_install_requirements()
