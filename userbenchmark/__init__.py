from pathlib import Path

CURRENT_DIR = Path(__file__).parent

def list_userbenchmarks():
    ub_dirs = [x for x in CURRENT_DIR.iterdir() if x.is_dir() and x.joinpath('__init__.py').exists() ]
    ub_names = list(map(lambda x: x.name, ub_dirs))
    return ub_names