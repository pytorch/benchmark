import importlib
import sys
from typing import List, Dict

TORCH_DEPS = ["torch", "torchvision", "torchaudio"]


class add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


def get_pkg_versions(packages: List[str], reload: bool = False) -> Dict[str, str]:
    versions = {}
    for module in packages:
        module = importlib.import_module(module)
        if reload:
            module = importlib.reload(module)
        versions[module.__name__] = module.__version__
    return versions
