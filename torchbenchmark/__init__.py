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
    if os.path.exists(os.path.join(model_path, install_file)):
        subprocess.check_call([sys.executable, install_file], cwd=model_path)
    else:
        print('No install.py is found in {}.'.format(model_path))
        sys.exit(-1)

def _model_folder():
    return Path(__file__).parent / model_dir

def model_names():
    return sorted(child.name for child in _model_folder().iterdir() if child.is_dir())

def setup():
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)

    for model_name in model_names():
        _install_deps(str((_model_folder() / model_name).absolute()))


def list_models():
    return (load_model(model_name) for model_name in model_names())

def load_model(model_name):
    module = importlib.import_module(f'.models.{model_name}', package=__name__)
    Model = getattr(module, 'Model')
    if not hasattr(Model, 'name'):
        Model.name = model_name
    return Model
