"""
Patch the transformer source code to enable optimizations.
"""
import os
import subprocess
import sys

PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")

def patch_transformers():
    import transformers
    import patch
    patch_file = os.path.join(PATCH_DIR, "0001-transformers-enable-fx.patch")
    transformers_dir = os.path.dirname(transformers.__file__)
    p = patch.fromfile(patch_file)
    try:
        subprocess.check_output(["patch", "-p1", "--forward", "-i", patch_file, "-r", "/tmp/rej"], cwd=transformers_dir)
    except subprocess.SubprocessError as e:
        output_str = str(e.output)
        if "previously applied" in output_str:
            return
        else:
            print(str(output_str))
            sys.exit(1)
