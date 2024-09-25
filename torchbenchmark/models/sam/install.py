import os
import subprocess
import sys

import requests
from utils.python_utils import pip_install_requirements


def download(uri):
    directory = ".data"
    filename = os.path.basename(uri)  # Extracts the filename from the URI
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    response = requests.get(uri, stream=True)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(
                chunk_size=8192
            ):  # Define the chunk size to be used
                f.write(chunk)
    else:
        print(f"Failed to download file with status code {response.status_code}")


def download_checkpoint():
    download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")


def download_data():
    download(
        "https://github.com/facebookresearch/segment-anything/raw/main/notebooks/images/truck.jpg"
    )


if __name__ == "__main__":
    pip_install_requirements()

    # Create .data folder in the script's directory
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
    os.makedirs(data_folder, exist_ok=True)

    # Download checkpoint and data files to the .data folder
    download_checkpoint()
    download_data()
