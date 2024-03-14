"""
Patch the transformer source code to enable optimizations.
"""
import os
import subprocess
import sys
from .basic_configs import download_model

PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")

def cache_model(name: str):
    download_model(name)

def patch_transformers():
    import transformers
    transformers_dir = os.path.dirname(transformers.__file__)
    if not os.path.exists(PATCH_DIR):
        return
    for patch_file in os.listdir(PATCH_DIR):
        patch_file_fullpatch = os.path.join(PATCH_DIR, patch_file)
        if not patch_file_fullpatch.endswith(".patch"):
            continue
        try:
            subprocess.check_output(["patch", "-p1", "--forward", "-i", patch_file_fullpatch, "-r", "/tmp/rej"], cwd=transformers_dir)
        except subprocess.SubprocessError as e:
            output_str = str(e.output)
            if "previously applied" in output_str:
                return
            else:
                print(str(output_str))
                sys.exit(1)
