import subprocess
import warnings
import json

from utils.python_utils import pip_install_requirements


def pip_install_requirements_doctr():
    required_pkgs = ["expecttest", "libglib", "pango"]
    try:
        installed_pkgs = [x["name"] for x in json.loads(subprocess.check_output(["conda", "list", "--json"], text=True))]
        if not all(pkg in installed_pkgs for pkg in required_pkgs):
            subprocess.check_call(
                [
                    "conda",
                    "install",
                    "-y",
                    *required_pkgs,
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
