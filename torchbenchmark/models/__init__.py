import os
import sys
import importlib
import importlib.util
from pathlib import Path

def _list_models_without_import():
    def _is_non_empty(dirpath):
        init_file_path = dirpath.joinpath("__init__.py")
        return init_file_path.exists() and init_file_path.stat().st_size > 0
    current_dir = Path(__file__).parent
    subdirs = [entry for entry in current_dir.iterdir() if entry.is_dir()]
    non_empty_subdirs = list(map(lambda x: x.name, filter(_is_non_empty, subdirs)))
    return non_empty_subdirs


class LazyImport:
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, attr: str):
        if self._module is None:
            self._module = importlib.import_module(self.module_name, package=__name__)
        return getattr(self._module, attr)


for _model_name in _list_models_without_import():
    globals()[_model_name] = LazyImport(f".{_model_name}")

