import subprocess
import os
import shutil
from pathlib import Path

BM_NAME = "torchao"

def install_torchao():
    # Install Torch-TensorRT with validation
    uninstall_torchtrt_cmd = ["pip", "uninstall", "-y", "torchao"]
    subprocess.check_call(uninstall_torchtrt_cmd)

    workdir = os.path.join(os.environ["PWD"], ".userbenchmark", BM_NAME)
    Path(workdir).mkdir(exist_ok=True, parents=True)
    ao_src_path = os.path.join(os.environ["PWD"], ".userbenchmark", BM_NAME, "ao")
    if os.path.exists(ao_src_path):
        shutil.rmtree(ao_src_path)

    clone_src = [
        "git",
        "clone",
        "https://github.com/pytorch-labs/ao",
    ]
    subprocess.check_call(clone_src, cwd=workdir)
    install_torchao_cmd = [
        "pip",
        "install",
        "-e",
        ".",
    ]
    subprocess.check_call(install_torchao_cmd, cwd=ao_src_path)
    validate_torchao_cmd = ["python", "-c", "'import torchao'"]
    subprocess.check_call(validate_torchao_cmd)

if __name__ == "__main__":
    install_torchao()
