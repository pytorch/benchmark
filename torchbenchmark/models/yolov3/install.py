import subprocess
import sys
import os
from pathlib import Path

def setup_data_dir():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    coco128_data_dir = os.path.join(current_dir.parent.parent, "data", ".data", "coco128")
    assert os.path.exists(coco128_data_dir), "Couldn't find coco128 data dir, please run install.py again."


def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    setup_data_dir()
