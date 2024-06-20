import sys
import subprocess
from typing import Dict, List
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
TORCH_DEPS = ["numpy", "torch", "torchvision", "torchaudio"]


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

def get_pkg_versions(packages: List[str]) -> Dict[str, str]:
    versions = {}
    for module in packages:
        cmd = [sys.executable, "-c", f'import {module}; print({module}.__version__)']
        version = subprocess.check_output(cmd).decode().strip()
        versions[module] = version
    return versions

def generate_pkg_constraints(package_versions: Dict[str, str]):
    """
    Generate package versions dict and save them to {REPO_ROOT}/build/constraints.txt
    """
    output_dir = REPO_DIR.joinpath("build")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir.joinpath("constraints.txt"), "w") as fp:
        for k, v in package_versions.items():
            fp.write(f"{k}=={v}\n")
