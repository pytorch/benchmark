import os
import sys
import subprocess
from pathlib import Path

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def spacy_download(language):
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', language])

def preprocess():
    root = os.path.join(str(Path(__file__).parent), ".data")
    subprocess.check_call([sys.executable, 'preprocess.py', '-lang_src', 'de', '-lang_trg', 'en', '-share_vocab', '-save_data', os.path.join(root, 'm30k_deen_shr.pkl')])

if __name__ == '__main__':
    pip_install_requirements()
    spacy_download('en')
    spacy_download('de')
    # Preprocessed pkl is larger than 100MB so we cannot skip preprocess
    preprocess()
