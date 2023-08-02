import os
import subprocess
import sys

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def download_data(data_folder):
    # CC-0 image from wikipedia page on pizza so legal to use
    subprocess.check_call(['wget', '-O', os.path.join(data_folder, 'pizza.jpg'), 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Pizza-3007395.jpg/2880px-Pizza-3007395.jpg'])

if __name__ == '__main__':
    pip_install_requirements()

    # Create .data folder in the script's directory
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
    os.makedirs(data_folder, exist_ok=True)

    download_data(data_folder)
