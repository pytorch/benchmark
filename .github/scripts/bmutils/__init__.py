import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
REPO_ROOT = str(CURRENT_DIR.parent.parent.parent)

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