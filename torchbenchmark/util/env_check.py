"""
PyTorch benchmark env check utils.
This file may be loaded without torch packages installed, e.g., in OnDemand CI.
"""
import importlib
from typing import List, Dict, Tuple

MAIN_RANDOM_SEED = 1337

def set_random_seed():
    import torch
    import random
    import numpy
    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)

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

def correctness_check(eager_output: Tuple['torch.Tensor'], output: Tuple['torch.Tensor']) -> float:
    import torch
    # sanity checks
    assert len(eager_output) == len(output), "Correctness check requires two inputs have the same length"
    result = 1.0
    for i in range(len(eager_output)):
        t1 = eager_output[i]
        t2 = output[i]
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-4)
        # need to call float() because fp16 tensor may overflow when calculating cosine similarity
        result *= cos(t1.flatten().float(), t2.flatten().float())
    assert list(result.size())==[], "The result of cosine similarity must be a scalar."
    return float(result)