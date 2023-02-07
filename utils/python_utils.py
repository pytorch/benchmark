import argparse
import subprocess

DEFAULT_PYTHON_VERSION = "3.8"

PYTHON_VERSION_MAP = {
    "3.8": {
        "pytorch_url": "cp38",
    },
    "3.10": {
        "pytorch_url": "cp310",
    },
}

def install_conda_env(pyver: str, name: str):
    command = [ "conda", "create", "-n", name, f"python={pyver}" ]
    subprocess.check_call(command)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyver", type=str, default=DEFAULT_PYTHON_VERSION, help="Specify the Python version.")
    parser.add_argument("--install-conda-env", type=str, default=None, help="Install conda environment of the default Python version.")
    args = parser.parse_args()
    if args.install_conda_env:
        install_conda_env(args.pyver, args.install_conda_env)
