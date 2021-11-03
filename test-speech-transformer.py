import contextlib
from typing import Union, Iterator
import pathlib
import os
import sys
import importlib

@contextlib.contextmanager
def add_sys_path(path: Union[str, os.PathLike]) -> Iterator[None]:
    """Temporarily add the given path to `sys.path`."""
    path = os.fspath(path)
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)


# p = pathlib.Path(__file__).parent.resolve().joinpath('torchbenchmark/models').joinpath('speech_transformer')
# with add_sys_path(p):
#    print(sys.path)
module = importlib.import_module(f'.models.speech_transformer', package="torchbenchmark")
