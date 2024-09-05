import os
from pathlib import Path

from utils import s3_utils
from utils.python_utils import pip_install_requirements


def setup_data_dir():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    coco128_data_dir = os.path.join(
        current_dir.parent.parent, "data", ".data", "coco128"
    )
    assert os.path.exists(
        coco128_data_dir
    ), "Couldn't find coco128 data dir, please run install.py again."


if __name__ == "__main__":
    s3_utils.checkout_s3_data("INPUT_TARBALLS", "coco128.tar.gz", decompress=True)
    pip_install_requirements()
