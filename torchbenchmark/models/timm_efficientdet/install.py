import os
import sys
import patch
from pathlib import Path
import subprocess

def check_data_dir():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    coco2017_data_dir = os.path.join(current_dir.parent.parent, "data", ".data", "coco2017-minimal")
    assert os.path.exists(coco2017_data_dir), "Couldn't find coco2017 minimal data dir, please run install.py again."

def patch_effdet():
    import effdet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(current_dir, "effdet.patch")
    target_dir = os.path.dirname(effdet.__file__)
    p = patch.fromfile(patch_file)
    if not p.apply(strip=1, root=target_dir):
        print("Failed to patch effdet. Exit.")
        exit(1)

def patch_pycocotools():
    import pycocotools
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(current_dir, "pycocotools.patch")
    target_dir = os.path.dirname(os.path.abspath(pycocotools.__file__))
    p = patch.fromfile(patch_file)
    if not p.apply(strip=1, root=target_dir):
        print("Failed to patch pycocotools. Exit.")
        exit(1)

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    check_data_dir()
    pip_install_requirements()
    patch_effdet()
    patch_pycocotools()
