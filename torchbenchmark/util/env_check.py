import importlib
from typing import List, Dict

def validate_args(**kwargs):
    print(kwargs.keys())
    assert "device" in kwargs, "Model must specify `device`."
    assert "test" in kwargs, "Model must specify `test`, test can be either 'train' or 'eval'."
    assert kwargs["test"] == "train" or kwargs["test"] == "eval", "Model test can only be train or eval."
    assert "extra_args" in kwargs, "Model must specify `extra_args`"

def get_pkg_versions(packages: List[str]) -> Dict[str, str]:
    versions = {}
    for module in packages:
        module = importlib.import_module(module)
        versions[module] = module.__version__
    return versions

def has_native_amp() -> bool:
    import torch
    try:
        if getattr(torch.cuda.amp, 'autocast') is not None:
            return True
    except AttributeError:
        pass
    return False
