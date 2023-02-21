import subprocess
import sys

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    #pip_install_requirements()
    print("skip pip install requirements by qinlingg")
