import os
from pathlib import Path

PATH = os.path.dirname(os.path.realpath(__file__))


root = str(Path(__file__).parent.parent.parent)
DATA_PATH = f'{root}/data/.data'

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
