"""
Patch the transformer source code to enable optimizations.
"""
import os
import sys
import patch

PATCH_DIR = os.path.join(os.path.dirname(__file__), "patches")

def patch_transformer():
    import transformers
    patch_file = os.path.join(PATCH_DIR, "0001-transformers-enable-fx.patch")
    transformers_dir = os.path.dirname(transformers.__file__)
    p = patch.fromfile(patch_file)
    if not p.apply(strip=1, root=transformers_dir):
        print("Failed to patch fastNLP. Exit.")
        sys.exit(1)