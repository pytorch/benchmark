import importlib
import torch
from typing import List, Dict, Tuple

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

def correctness_check(eager_output: Tuple[torch.Tensor], output: Tuple[torch.Tensor]) -> float:
    # sanity checks
    assert len(eager_output) == len(output), "Correctness check requires two inputs have the same length"
    result = 1.0
    for i in range(len(eager_output)):
        t1 = eager_output[i]
        t2 = output[i]
        cos = torch.nn.CosineSimilarity()
        result *= cos(t1, t2)
    return result