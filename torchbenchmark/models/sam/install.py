import subprocess
import sys

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def download_checkpoint():
    subprocess.check_call(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'])


if __name__ == '__main__':
    pip_install_requirements()
    download_checkpoint()