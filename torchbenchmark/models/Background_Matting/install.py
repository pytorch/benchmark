from utils import python_utils, s3_utils


def pip_install_requirements():
    python_utils.pip_install_requirements("requirements.txt")


if __name__ == "__main__":
    pip_install_requirements()
    s3_utils.checkout_s3_data(
        "INPUT_TARBALLS", "Background_Matting_inputs.tar.gz", decompress=True
    )
