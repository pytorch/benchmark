from utils import s3_utils
from utils.python_utils import pip_install_requirements

if __name__ == "__main__":
    s3_utils.checkout_s3_data(
        "INPUT_TARBALLS", "Reddit_minimal.tar.gz", decompress=True
    )
    pip_install_requirements(
        extra_args=["-f", "https://data.pyg.org/whl/torch-2.1.0+cpu.html"]
    )
