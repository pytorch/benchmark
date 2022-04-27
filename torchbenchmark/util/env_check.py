"""
PyTorch benchmark env check utils.
This file may be loaded without torch packages installed, e.g., in OnDemand CI.
"""
import importlib
from typing import List, Dict, Tuple, Optional

MAIN_RANDOM_SEED = 1337
# run 10 rounds for stableness and correctness tests
CORRECTNESS_CHECK_ROUNDS: int = 10
CORRECTNESS_THRESHOLD: float = 0.99

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

def stableness_check(model: 'torchbenchmark.util.model.BenchmarkModel') -> Optional[Tuple['torch.Tensor']]:
    """Get the eager output. Run eager mode a couple of times to guarantee stableness.
       If the result is not stable, return None. """
    assert model.test=="eval", "We only support stableness check for inference."
    previous_result = None
    for _i in range(CORRECTNESS_CHECK_ROUNDS):
        if not previous_result:
            previous_result = model.invoke()
        else:
            cos_sim = cos_similarity(model.invoke(), previous_result)
            if cos_sim < CORRECTNESS_THRESHOLD:
                return None
    return previous_result

def correctness_check(model: 'torchbenchmark.util.model.BenchmarkModel') -> str:
    assert model.test=="eval", "We only support correctness check for inference."
    assert hasattr(model, 'eager_output'), "Need stableness result to check correctness."
    if not model.eager_output:
        return "Unstable"
    for _i in range(CORRECTNESS_CHECK_ROUNDS):
        cos_sim = cos_similarity(model.eager_output, model.invoke())
        if cos_sim < CORRECTNESS_THRESHOLD:
            return f"Incorrect (cos_sim: {cos_sim})"
    return "Correct"

def cos_similarity(eager_output: Tuple['torch.Tensor'], output: Tuple['torch.Tensor']) -> float:
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