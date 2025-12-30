import os
import time

import requests
from utils.python_utils import pip_install_requirements


def download(output_filename, uri):
    # Download the file with streaming to handle large files
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    max_retries = 10
    retry_delay = 3  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(uri, headers=headers, stream=True, timeout=30)
            if response.status_code == 200:
                with open(output_filename, "wb") as f:
                    for chunk in response.iter_content(
                        chunk_size=8192
                    ):  # Define the chunk size to be used
                        f.write(chunk)
                print(f"Successfully downloaded {output_filename}")
                return
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    print(f"Rate limited (429). Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to download file with status code {response.status_code} after {max_retries} attempts")
            else:
                print(f"Failed to download file with status code {response.status_code}")
                return
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Request failed: {e}. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to download file after {max_retries} attempts: {e}")


def download_data(data_folder):
    # CC-0 image from wikipedia page on pizza so legal to use
    download(
        os.path.join(data_folder, "pizza.jpg"),
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Pizza-3007395.jpg/2880px-Pizza-3007395.jpg",
    )


if __name__ == "__main__":
    # pip_install_requirements()

    # Create .data folder in the script's directory
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
    os.makedirs(data_folder, exist_ok=True)

    download_data(data_folder)
