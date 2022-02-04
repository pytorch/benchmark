import importlib
from typing import List, Dict
from torchbenchmark.util.extra_args import parse_args, apply_args

# Apply post initialization features
def post_processing(model: 'torchbenchmark.util.model.BenchmarkModel'):
    # sanity checks of the options
    assert model.test == "train" or model.test == "eval", f"Test must be 'train' or 'eval', but provided {model.test}."
    model.extra_args = parse_args(model, model.extra_args)
    apply_args(model, model.extra_args)

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
