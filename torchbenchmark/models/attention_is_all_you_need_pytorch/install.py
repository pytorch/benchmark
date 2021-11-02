import subprocess
import sys
import git


def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

def spacy_generate(language):
    pass

def preprocess():
    subprocess.check_call([sys.executable, 'attention_is_all_you_need_pytorch/preprocess.py', '-lang_src', 'de', '-lang_trg', 'en', '-share_vocab', '-save_data', 'm30k_deen_shr.pkl'])

if __name__ == '__main__':
    pip_install_requirements()
    # Checkout code
    # Generate spacy-style dataset
    spacy_generate('en')
    spacy_generate('de')
    # Preprocess the dataset and save the result as pkl file
    preprocess()

