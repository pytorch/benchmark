import subprocess

import torch


def install_torch_tensorrt():
    # Install Torch-TensorRT with validation
    uninstall_torchtrt_cmd = ["pip", "uninstall", "-y", "torch_tensorrt"]
    subprocess.check_call(uninstall_torchtrt_cmd)

    if torch.version.cuda.startswith("12"):
        cuda_index_modifier = "cu121"
    elif torch.version.cuda.startswith("11"):
        cuda_index_modifier = "cu118"
    else:
        raise AssertionError(
            f"Detected Torch-TRT unsupported CUDA version {torch.version.cuda}"
        )

    pytorch_nightly_url = (
        f"https://download.pytorch.org/whl/nightly/{cuda_index_modifier}"
    )
    install_torchtrt_cmd = [
        "pip",
        "install",
        "--pre",
        "--no-cache-dir",
        "torch_tensorrt",
        "--extra-index-url",
        pytorch_nightly_url,
    ]
    validate_torchtrt_cmd = ["python", "-c", "'import torch_tensorrt'"]

    # Install and validate Torch-TensorRT
    try:
        subprocess.check_call(install_torchtrt_cmd)
        subprocess.check_call(validate_torchtrt_cmd)
    except subprocess.CalledProcessError:
        subprocess.check_call(uninstall_torchtrt_cmd)
        print("Failed to install torch-tensorrt, skipping install")


if __name__ == "__main__":
    install_torch_tensorrt()
