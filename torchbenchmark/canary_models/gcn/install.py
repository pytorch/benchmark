
import subprocess
import sys


def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt', '-f', 'https://data.pyg.org/whl/torch-2.0.0+cpu.html'])


if __name__ == '__main__':
    pip_install_requirements()
