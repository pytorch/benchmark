import os
import subprocess
from pathlib import Path

def setup_data_dir():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    coco2017_data_dir = os.path.join(current_dir.parent.parent, "data", ".data", "coco2017-minimal")
    assert os.path.exists(coco2017_data_dir), "Couldn't find coco2017 minimal data dir, please run install.py again."

if __name__ == '__main__':
    setup_data_dir()
