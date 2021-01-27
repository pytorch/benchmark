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


def _install_deps(model_path, verbose=True):
    run_args = [
        [sys.executable, install_file],
    ]
    run_kwargs = {
        'cwd': model_path,
        'check': True,
    }
    try:
        if os.path.exists(os.path.join(model_path, install_file)):
            if not verbose:
                run_kwargs['stderr'] = subprocess.STDOUT
                run_kwargs['stdout'] = subprocess.PIPE
            subprocess.run(*run_args, **run_kwargs)
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


def setup(verbose=True, continue_on_fail=False):
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)

    failures = {}
    for model_path in _list_model_paths():
        print(f"running setup for {model_path}...", end="", flush=True)
        success, errmsg = _install_deps(model_path, verbose=verbose)
        if success:
            print("OK")
        else:
            print("FAIL")
            try:
                errmsg = errmsg.decode()
            except:
                pass
            failures[model_path] = errmsg
            if not continue_on_fail:
                break
    if verbose and len(failures):
        for model_path in failures:
            print(f"Error for {model_path}:")
            print("---------------------------------------------------------------------------")
            print(failures[model_path])
            print("---------------------------------------------------------------------------")
            print()

    return len(failures) == 0

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
