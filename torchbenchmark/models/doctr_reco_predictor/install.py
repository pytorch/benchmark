import os
import warnings
import subprocess
import sys

def pip_install_requirements():
    try:
        subprocess.check_call(["conda", "install", "-y", "expecttest", "libglib", "pango", "-c", "conda-forge"])
    except:
        warnings.warn("The doctr_reco_predictor model requires conda binary libaries to be installed. Missing conda packages might break this model.")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
