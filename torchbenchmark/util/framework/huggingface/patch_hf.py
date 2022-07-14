"""
Patch the transformer source code to enable optimizations.
"""
import os
import subprocess
import sys
from .model_factory import class_models
from transformers import AutoConfig, ReformerConfig, BigBirdConfig, BertConfig

PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")

def cache_model(name: str):
    import transformers
    model_config = eval(class_models[name][2])
    model_ctor = getattr(transformers, class_models[name][3])
    model_ctor.from_config(model_config)

def patch_transformers():
    pass
    # import transformers
    # transformers_dir = os.path.dirname(transformers.__file__)
    # for patch_file in os.listdir(PATCH_DIR):
    #     patch_file_fullpatch = os.path.join(PATCH_DIR, patch_file)
    #     try:
    #         subprocess.check_output(["patch", "-p1", "--forward", "-i", patch_file_fullpatch, "-r", "/tmp/rej"], cwd=transformers_dir)
    #     except subprocess.SubprocessError as e:
    #         output_str = str(e.output)
    #         if "previously applied" in output_str:
    #             return
    #         else:
    #             print(str(output_str))
    #             sys.exit(1)
