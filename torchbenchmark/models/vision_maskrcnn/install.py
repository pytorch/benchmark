from utils import s3_utils
from utils.python_utils import pip_install_requirements

if __name__ == "__main__":
    s3_utils.checkout_s3_data(
        "INPUT_TARBALLS", "coco2017-minimal.tar.gz", decompress=True
    )
    pip_install_requirements()
