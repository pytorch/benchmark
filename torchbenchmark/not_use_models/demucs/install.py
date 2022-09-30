import subprocess
import sys


def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def spacy_download(language):
    pass

def preprocess():
    pass

if __name__ == '__main__':
    pip_install_requirements()
    spacy_download('')
    preprocess()
