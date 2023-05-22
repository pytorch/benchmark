import os
import subprocess
import sys

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def download_checkpoint():
    subprocess.check_call(['wget', '-P', '.data', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'])

def download_data():
    subprocess.check_call(['wget', '-P', '.data', 'https://github.com/facebookresearch/segment-anything/raw/main/notebooks/images/truck.jpg'])

if __name__ == '__main__':
    pip_install_requirements()

    # Create .data folder in the script's directory
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
    os.makedirs(data_folder, exist_ok=True)

    # Download checkpoint and data files to the .data folder
    download_checkpoint()
    download_data()
