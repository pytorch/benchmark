import os
import patch
import subprocess
import sys

def patch_dalle2():
    import dalle2_pytorch
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dalle2_dir = os.path.dirname(dalle2_pytorch.__file__)
    #dalle2_target_file = os.path.join(dalle2_dir, "dalle2_pytorch", "dalle2_pytorch.py")
    #setup_target_file = os.path.join(dalle2_dir, "setup.py")
    dalle2_patch = patch.fromfile(os.path.join(current_dir, "dalle2_pytorch.patch"))
    setup_patch = patch.fromfile(os.path.join(current_dir, "setup.patch"))
    if not dalle2_patch.apply(strip=1, root=dalle2_dir):
        print("Failed to patch dalle2_pytorch/dalle2_pytorch.py. Exit.")
        exit(1)
    if not setup_patch.apply(strip=1, root=dalle2_dir):
        print("Failed to patch setup.py. Exit.")
        exit(1)

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    patch_dalle2()