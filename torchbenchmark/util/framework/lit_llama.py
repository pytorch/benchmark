import os
import subprocess
import sys
import traceback
from pathlib import Path

from torchbenchmark import REPO_PATH

LIT_LLAMA_PATH = os.path.join(REPO_PATH, "submodules", "lit-llama")


def update_lit_llama_submodule():
    update_command = [
        "git",
        "submodule",
        "update",
        "--init",
        "--recursive",
        os.path.join("submodules", "lit-llama"),
    ]
    subprocess.check_call(update_command, cwd=REPO_PATH)


def pip_install_requirements():
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            os.path.join(LIT_LLAMA_PATH, "requirements.txt"),
        ]
    )


def openllama_download():
    if os.path.exists(
        os.path.join(LIT_LLAMA_PATH, "checkpoints/lit-llama/7B/lit-llama.pth")
    ):
        return
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(LIT_LLAMA_PATH, "scripts/download.py"),
            "--repo_id",
            "openlm-research/open_llama_7b_700bt_preview",
            "--local_dir",
            os.path.join(LIT_LLAMA_PATH, "checkpoints/open-llama/7B"),
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(LIT_LLAMA_PATH, "scripts/convert_hf_checkpoint.py"),
            "--checkpoint_dir",
            os.path.join(LIT_LLAMA_PATH, "checkpoints/open-llama/7B"),
            "--model_size",
            "7B",
        ],
        cwd=LIT_LLAMA_PATH,
    )


def install_lit_llama():
    import torch

    update_lit_llama_submodule()
    pip_install_requirements()
    try:
        from pynvml import nvmlDeviceGetMemoryInfo

        info = nvmlDeviceGetMemoryInfo(torch.cuda._get_pynvml_handler())
        if info.total < 40 * 1024**3:
            print(
                "not enough GPU memory for 7B parameters, skipping llama (avail: {info.total / 1024 ** 3}GB)"
            )
            return
    except Exception as e:
        print("failed to test GPU memory, skipping llama weights")
        traceback.print_exc()
        return
    openllama_download()
