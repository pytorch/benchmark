import os
import patch
import subprocess
import sys

def patch_dalle2():
    import dalle2_pytorch
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dalle2_dir = os.path.dirname(dalle2_pytorch.__file__)
    dalle2_patch = patch.fromfile(os.path.join(current_dir, "dalle2_pytorch.patch"))
    if not dalle2_patch.apply(strip=1, root=dalle2_dir):
        print("Failed to patch dalle2_pytorch/dalle2_pytorch.py. Exit.")
        exit(1)

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
    # DALLE2_pytorch requires embedding-reader
    # https://github.com/lucidrains/DALLE2-pytorch/blob/00e07b7d61e21447d55e6d06d5c928cf8b67601d/setup.py#L34
    # embedding-reader requires an old version of pandas and pyarrow
    # https://github.com/rom1504/embedding-reader/blob/a4fd55830a502685600ed8ef07947cd1cb92b083/requirements.txt#L5
    # So we need to reinstall a newer version of pandas and pyarrow, to be compatible with other models
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'pandas', 'pyarrow'])

if __name__ == '__main__':
    pip_install_requirements()
    patch_dalle2()