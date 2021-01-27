import subprocess
import sys


def setup_install():
    subprocess.check_call([sys.executable, 'setup.py', 'develop'])

if __name__ == '__main__':
    setup_install()
