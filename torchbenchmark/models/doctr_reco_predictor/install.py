import os
import subprocess
import sys

def pip_install_requirements():
    # install deps from conda-forge
    # model doctr_reco_predictor needs weasyprint, which needs libglib and pango
    subprocess.check_call(["conda", "install", "-y", "expecttest", "libglib", "pango", "-c", "conda-forge"])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
