import subprocess
import sys

from utils.python_utils import get_pip_cmd


def pip_install_requirements():
    subprocess.check_call(
        get_pip_cmd() + ["install", "-q", "-r", "requirements.txt"]
    )


if __name__ == "__main__":
    pip_install_requirements()
