import subprocess
import sys
import pathlib

path_to_reqs = pathlib.Path(__file__).parent / 'requirements.txt'

def setup_install():
    print(path_to_reqs)
    subprocess.check_call(['pip', 'install', '-r', str(path_to_reqs)])

if __name__ == '__main__':
    setup_install()