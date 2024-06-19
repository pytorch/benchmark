import warnings
from pathlib import Path
import subprocess

DEFAULT_PYTHON_VERSION = "3.11"

PYTHON_VERSION_MAP = {
    "3.8": {
        "pytorch_url": "cp38",
    },
    "3.10": {
        "pytorch_url": "cp310",
    },
    "3.11": {
        "pytorch_url": "cp311",
    },
}
REPO_DIR = Path(__file__).parent.parent


def create_conda_env(pyver: str, name: str):
    command = ["conda", "create", "-n", name, "-y", f"python={pyver}"]
    subprocess.check_call(command)


def pip_install_requirements(requirements_txt="requirements.txt", continue_on_fail=False):
    import sys
    constraints_file = REPO_DIR.joinpath("build", "contraints.txt")
    if not constraints_file.exists():
        warnings.warn("contraints.txt could not be found, please rerun install.py.")
        constraints_parameters = []
    else:
        constraints_parameters = ["-c", str(constraints_file.resolve())]
    if not continue_on_fail:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements_txt] + constraints_parameters,
        )
        return True, None
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_txt] + constraints_parameters,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    except Exception as e:
        return (False, e)
    return True, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyver",
        type=str,
        default=DEFAULT_PYTHON_VERSION,
        help="Specify the Python version.",
    )
    parser.add_argument(
        "--create-conda-env",
        type=str,
        default=None,
        help="Create conda environment of the default Python version.",
    )
    args = parser.parse_args()
    if args.create_conda_env:
        create_conda_env(args.pyver, args.create_conda_env)
