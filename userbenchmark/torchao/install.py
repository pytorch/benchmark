import os
import subprocess

def install_torchao():
    # Set ARCH list so that we can build fp16 with SM75+, the logic is copied from
    # pytorch/builder
    # https://github.com/pytorch/ao/blob/main/packaging/env_var_script_linux.sh#L16C1-L19
    torchao_env = os.environ
    torchao_env["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    subprocess.check_call(["pip", "install", "--pre", "git+https://github.com/pytorch/ao.git"], env=torchao_env)

if __name__ == "__main__":
    install_torchao()