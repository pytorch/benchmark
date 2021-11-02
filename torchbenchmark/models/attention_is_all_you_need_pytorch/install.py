import subprocess
import sys
import git
import os
import yaml
import pathlib
from spacy_simulator import generate_dataset

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def spacy_generate_multi30k(langs):
    MULTI30K_DIR = '.data/multi30k/'
    pathlib.Path(MULTI30K_DIR).mkdir(parents=True, exist_ok=True)
    for lang in langs:
        config_file = f'multi30k-configs/multi30k-{lang}-config.yaml'
        with open(config_file, "r") as cf:
            config = yaml.safe_load(cf)
        generate_dataset(MULTI30K_DIR, config)

def preprocess():
    subprocess.check_call([sys.executable, 'attention/preprocess.py', '-lang_src', 'de', '-lang_trg', 'en', '-share_vocab', '-save_data', '.data/m30k_deen_shr.pkl'])

if __name__ == '__main__':
    pip_install_requirements()
    # Generate spacy-style datasets
    spacy_generate_multi30k(["en", "de"])
    # Preprocess the dataset and save the result as pkl file
    preprocess()

