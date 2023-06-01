import os
import subprocess
import sys
import tarfile
import tqdm

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def download_checkpoint():
    checkpoint_path = os.path.join('.data', 'sam_vit_h_4b8939.pth')

    # Check if the checkpoint file already exists
    if os.path.exists(checkpoint_path):
        print("Checkpoint file already exists. Skipping download.")
        return

    # Download the checkpoint
    subprocess.check_call(['wget', '-P', '.data', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'])
    print("Checkpoint downloaded successfully")


def download_data_point():
    data_point_path = os.path.join('.data', 'truck.jpg')

    # Check if the data point file already exists
    if os.path.exists(data_point_path):
        print("Data point file already exists. Skipping download.")
        return

    # Download the data point
    subprocess.check_call(['wget', '-P', '.data', 'https://github.com/facebookresearch/segment-anything/raw/main/notebooks/images/truck.jpg'])

    print("Data point downloaded successfully")


def download_dataset():
    dataset_path = os.path.join('.data', 'segment-anything.tar')
    dataset_folder = os.path.join('.data', 'segment-anything')

    # Check if the dataset folder already exists
    if os.path.exists(dataset_folder):
        print("Dataset folder already exists. Skipping download.")
        return

    # Check if the dataset file already exists
    if os.path.exists(dataset_path):
        print("Dataset file already exists. Skipping download.")
    else:
        # Download the dataset
        # This is a 10GB subset of the full data found on https://ai.facebook.com/datasets/segment-anything-downloads/
        subprocess.check_call(['wget', '-P', '.data', 'https://scontent.fsan1-1.fna.fbcdn.net/m1/v/t6/An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0.tar?ccb=10-5&oh=00_AfCxktL_1l96jDw_LS4N_AkRjcO1EzG6R2RFEjynsHDSqA&oe=64A0817E&_nc_sid=f5a210'])

        # Rename the downloaded file to 'segment-anything.tar'
        downloaded_file = os.path.join('.data', 'An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0.tar?ccb=10-5&oh=00_AfCxktL_1l96jDw_LS4N_AkRjcO1EzG6R2RFEjynsHDSqA&oe=64A0817E&_nc_sid=f5a210')
        os.rename(downloaded_file, dataset_path)

    # Untar the dataset file into the folder '.data/segment-anything'
    with tarfile.open(dataset_path, 'r') as tar:
        # Get the total number of files in the archive
        total_files = len(tar.getmembers())

        # Set up the progress bar
        progress_bar = tqdm(tar.getmembers(), total=total_files, desc='Extracting')

        # Extract each file while updating the progress bar
        for member in progress_bar:
            tar.extract(member, path=dataset_folder)

    print("SAM dataset downloaded and untarred successfully")


if __name__ == '__main__':
    pip_install_requirements()

    # Create .data folder in the script's directory
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')
    os.makedirs(data_folder, exist_ok=True)

    # Download checkpoint and data files to the .data folder
    download_checkpoint()
    download_data_point()
    download_dataset()
