import subprocess

from utils.python_utils import get_pip_cmd


def setup_install():
    subprocess.check_call(get_pip_cmd() + ["install", "-e", "."])


if __name__ == "__main__":
    setup_install()
