import argparse
import subprocess

DEFAULT_PYTHON_VERSION = "3.10"

PYTHON_VERSION_MAP = {
    "3.8": {
        "pytorch_url": "cp38",
    },
    "3.10": {
        "pytorch_url": "cp310",
    },
}

def create_conda_env(pyver: str, name: str):
    command = [ "conda", "create", "-n", name, f"python={pyver}" ]
    subprocess.check_call(command)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyver", type=str, default=DEFAULT_PYTHON_VERSION, help="Specify the Python version.")
    parser.add_argument("--create-conda-env", type=str, default=None, help="Create conda environment of the default Python version.")
    args = parser.parse_args()
    if args.create_conda_env:
        create_conda_env(args.pyver, args.create_conda_env)
