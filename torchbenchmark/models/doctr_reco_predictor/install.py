import warnings
import subprocess
from utils.python_utils import pip_install_requirements


def pip_install_requirements_doctr():
    try:
        subprocess.check_call(
            [
                "conda",
                "install",
                "-y",
                "expecttest",
                "libglib",
                "pango",
                "-c",
                "conda-forge",
            ]
        )
    except:
        warnings.warn(
            "The doctr_reco_predictor model requires conda binary libaries to be installed. Missing conda packages might break this model."
        )
    pip_install_requirements()


if __name__ == "__main__":
    pip_install_requirements_doctr()
