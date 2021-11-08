import os
import subprocess

# Decompress tarball to .data
def unpack_input_tarball(input_tarball='detectron2_maskrcnn_benchmark_data.tar.gz'):
    pass

def install_or_build_detectron2(build_from_src=False):
    # First try install the prebuilt package, if failed, rebuild detectron2
    try:
        if not build_from_src:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'detectron2', '-f',
                                   'https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html'])
        else:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '', '-f',
    except SubProcessError:
        if not build_from_src:
            install_or_build_detectron2(build_from_src=True)
        else:
            print(f"Failed to install or build detectron2. Please check your environment.")

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

if __name__ == '__main__':
    pip_install_requirements()
    unpack_input_tarball()
