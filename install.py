import subprocess
import sys
from torchbenchmark import setup

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m',
                           'pip', 'install', '-r', 'requirements.txt'])


if __name__ == '__main__':
    pip_install_requirements()
    setup()