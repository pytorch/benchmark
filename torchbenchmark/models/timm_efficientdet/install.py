import os

import patch
from utils import s3_utils
from utils.python_utils import pip_install_requirements


def patch_effdet():
    import effdet

    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(current_dir, "effdet.patch")
    target_dir = os.path.dirname(effdet.__file__)
    p = patch.fromfile(patch_file)
    if not p.apply(strip=1, root=target_dir):
        print("Failed to patch effdet. Exit.")
        exit(1)


def patch_pycocotools():
    import pycocotools

    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(current_dir, "pycocotools.patch")
    target_dir = os.path.dirname(os.path.abspath(pycocotools.__file__))
    p = patch.fromfile(patch_file)
    if not p.apply(strip=1, root=target_dir):
        print("Failed to patch pycocotools. Exit.")
        exit(1)


if __name__ == "__main__":
    s3_utils.checkout_s3_data(
        "INPUT_TARBALLS", "coco2017-minimal.tar.gz", decompress=True
    )
    pip_install_requirements()
    patch_effdet()
    patch_pycocotools()
