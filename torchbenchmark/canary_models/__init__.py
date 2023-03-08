from typing import List
import os

from torchbenchmark import _list_model_paths

CANARY_MODEL_DIR = os.path.realpath(os.path.dirname(__file__))

def install_models(models: List[str], continue_on_fail=False):
    models = list(map(lambda p: p.lower(), models))
    model_paths = filter(lambda p: True if not models else os.path.basename(p).lower() in models, \
                         _list_model_paths(model_dir=CANARY_MODEL_DIR))
    for model_path in model_paths:
        print(f"running setup for {model_path}...", end="", flush=True)
        success, errmsg, stdout_stderr = _install_deps(model_path, verbose=False)
        if success and errmsg and "No install.py is found" in errmsg:
            print("SKIP - No install.py is found")

def is_canary_model(model: str):
    pass

def load_canary_model(model: str):
    pass