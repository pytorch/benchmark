import os
import subprocess

# Decompress tarball to .data
def unpack_input_tarball(input_tarball='detectron2_maskrcnn_benchmark_data.tar.gz'):
    pass

def build_detectron2():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/facebookresearch/detectron2.git'])

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    build_detectron2()
    unpack_input_tarball()
