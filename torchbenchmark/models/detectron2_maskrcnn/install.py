import os
import sys
import subprocess

# Decompress tarball to .data
def unpack_input_tarball(input_tarball='detectron2_maskrcnn_benchmark_data.tar.gz'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, ".data")
    os.makedirs(data_dir, exist_ok=True)
    subprocess.check_call(['tar', 'xzvf', input_tarball, '-C', data_dir])

def build_detectron2():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'git+https://github.com/facebookresearch/detectron2.git'])

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    build_detectron2()
    unpack_input_tarball()
