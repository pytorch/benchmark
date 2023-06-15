import subprocess
import sys
from utils import s3_utils

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    s3_utils.checkout_s3_data("MODEL_PKLS", "drq/obs.pkl", decompress=False)
