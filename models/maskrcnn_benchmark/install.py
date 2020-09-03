import os
import subprocess
import sys
import tempfile


def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])


def install_other_dependencies():
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f'Installing stuff in {tmpdir}')
        subprocess.check_call(['bash', 'install_dependencies.sh', tmpdir])


if __name__ == '__main__':
    pip_install_requirements()
    install_other_dependencies()
