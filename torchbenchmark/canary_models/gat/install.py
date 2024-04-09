
import subprocess
import sys
from utils import s3_utils


def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt', '-f', 'https://data.pyg.org/whl/torch-2.1.0+cpu.html'])


if __name__ == '__main__':
    s3_utils.checkout_s3_data("INPUT_TARBALLS", "Reddit_minimal.tar.gz", decompress=True)
    pip_install_requirements()
