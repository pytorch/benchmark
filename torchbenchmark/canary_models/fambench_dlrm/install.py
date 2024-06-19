import os
import sys
import subprocess
from torchbenchmark import REPO_PATH
from utils.python_utils import pip_install_requirements


def update_fambench_submodule():
    "Update FAMBench submodule of the benchmark repo"
    update_command = [
        "git",
        "submodule",
        "update",
        "--init",
        "--recursive",
        os.path.join("submodules", "FAMBench"),
    ]
    subprocess.check_call(update_command, cwd=REPO_PATH)


if __name__ == "__main__":
    update_fambench_submodule()
    pip_install_requirements()
