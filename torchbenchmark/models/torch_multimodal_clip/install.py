import os
import subprocess
import sys
import requests

def download(output_filename, uri):
    # Download the file with streaming to handle large files
    response = requests.get(uri, stream=True)
    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):  # Define the chunk size to be used
                f.write(chunk)
    else:
        print(f'Failed to download file with status code {response.status_code}')

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def download_data(data_folder):
    # CC-0 image from wikipedia page on pizza so legal to use
    download(os.path.join(data_folder, 'pizza.jpg'), 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Pizza-3007395.jpg/2880px-Pizza-3007395.jpg')

if __name__ == '__main__':
    pip_install_requirements()

    # Create .data folder in the script's directory
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
    os.makedirs(data_folder, exist_ok=True)

    download_data(data_folder)
