import importlib
from urllib import request
from typing import List, Dict

TORCH_DEPS = ['torch', 'torchvision', 'torchtext']
proxy_suggestion = "Unable to verify https connectivity, " \
                   "required for setup.\n" \
                   "Do you need to use a proxy?"

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
