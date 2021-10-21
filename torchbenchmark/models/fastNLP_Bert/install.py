import subprocess
import sys
import patch

def patch_fastnlp():
    import fastNLP
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(current_dir, "fastnlp.patch")
    fastNLP_dir = os.path.dirname(fastNLP.__file__)
    fastNLP_target_file = os.path.join(fastNLP_dir, "embeddings", "bert_embedding.py")
    p = patch.fromfile(patch_file)
    if not p.apply(strip=1, root=fastNLP_dir):
        print("Failed to patch fastNLP. Exit.")
        exit(1)

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    patch_fastnlp()
