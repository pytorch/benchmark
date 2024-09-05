import subprocess
import sys


def setup_install():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])


if __name__ == "__main__":
    setup_install()
