from utils import s3_utils
from utils.python_utils import pip_install_requirements

if __name__ == '__main__':
    pip_install_requirements()
    s3_utils.checkout_s3_data("MODEL_PKLS", "drq/obs.pkl", decompress=False)
