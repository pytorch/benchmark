import importlib
from typing import List, Dict

# Apply post initialization features
def post_processing(model: 'BenchmarkModel'):
    # run the checks
    pass

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
