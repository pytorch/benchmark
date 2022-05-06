import subprocess
import sys
import os
import patch

def patch_opacus():
    import opacus
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(current_dir, "opacus.patch")
    opacus_dir = os.path.dirname(opacus.__file__)
    target_file = os.path.join(opacus_dir, "embeddings")
    p = patch.fromfile(patch_file)
    if not p.apply(strip=0, root=opacus_dir):
        print("Failed to patch opacus. Exit.")
        exit(1)

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    patch_opacus()
