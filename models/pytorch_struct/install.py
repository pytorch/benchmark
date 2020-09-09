import subprocess
import sys

def setup_install():
  subprocess.check_call('pip install dgl wandb'.split(' '))
  subprocess.check_call([sys.executable, 'setup.py', 'install'])

if __name__ == '__main__':
  setup_install()