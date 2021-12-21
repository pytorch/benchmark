import importlib
from typing import List, Dict

<<<<<<< HEAD
=======
def parse_extraargs(extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # by default, enable half precision for inference
    parser.add_argument("--eval-fp16", action='store_false', help="enable eval fp16")
    parser.add_argument("--flops", action='store_true', help="enable flops counting")
    parser.add_argument("--fx2trt", action='store_true', help="enable fx2trt")
    return parser.parse_args(extra_args)

>>>>>>> 9f7ca38 (Enable flops counting.)
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
