import subprocess
import sys
from utils import s3_utils


def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    s3_utils.checkout_s3_data("INPUT_TARBALLS", "pytorch_CycleGAN_and_pix2pix_inputs.tar.gz", decompress=True)
    pip_install_requirements()
