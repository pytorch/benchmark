import os
from pathlib import Path
import subprocess
import sys 
import torch
from urllib import request
import importlib

proxy_suggestion = "Unable to verify https connectivity, " \
                   "required for setup.\n" \
                   "Do you need to use a proxy?"

this_dir = Path(__file__).parent.absolute()
model_dir = 'models'
install_file = 'install.py'


def _test_https(test_url='https://github.com', timeout=0.5):
    try:
        request.urlopen(test_url, timeout=timeout)
    except OSError:
        return False
    return True


def _install_deps(model_path):
    try:
        if os.path.exists(os.path.join(model_path, install_file)):
            subprocess.run([sys.executable, install_file],
                           cwd=model_path, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            return (False, f"No install.py is found in {model_path}.")
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    except Exception as e:
        return (False, e)

    return (True,  None)
    

def _list_model_paths():
    p = Path(__file__).parent.joinpath(model_dir)
    return sorted(str(child.absolute()) for child in p.iterdir() if child.is_dir())


def setup(verbose=False):
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)

    failures = {}
    for model_path in _list_model_paths():
        print(f"running setup for {model_path}...", end="", flush=True)
        success, errmsg = _install_deps(model_path)
        if success:
            print("OK")
        else:
            print("FAIL")
            failures[model_path] = errmsg
    if verbose and len(failures):
        for model_path in failures:
            print(f"Error for {model_path}:")
            print("---------------------------------------------------------------------------")
            print(failures[model_path])
            print("---------------------------------------------------------------------------")
            print()


def list_models():
    models = []
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        module = importlib.import_module(f'.models.{model_name}', package=__name__)
        Model = getattr(module, 'Model')
        if not hasattr(Model, 'name'):
            Model.name = model_name
        models.append(Model)
    return models
