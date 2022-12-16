import importlib
import sys
from urllib import request
from typing import List, Dict

TORCH_DEPS = ['torch', 'torchvision', 'torchtext', 'torchaudio']

proxy_suggestion = "Unable to verify https connectivity, " \
                   "required for setup.\n" \
                   "Do you need to use a proxy?"

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

def get_pkg_versions(packages: List[str], reload: bool=False) -> Dict[str, str]:
    versions = {}
    for module in packages:
        module = importlib.import_module(module)
        if reload:
            module = importlib.reload(module)
        versions[module.__name__] = module.__version__
    return versions

def _test_https(test_url: str = 'https://github.com', timeout: float = 0.5) -> bool:
    try:
        request.urlopen(test_url, timeout=timeout)
    except OSError:
        return False
    return True
