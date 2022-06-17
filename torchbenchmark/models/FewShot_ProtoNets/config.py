import os
from pathlib import Path

PATH = os.path.dirname(os.path.realpath(__file__))


root = str(Path(__file__).parent.parent)
DATA_PATH = os.path.join(root, "data", ".data", "omiglot_minimal")

EPSILON = 1e-8

if DATA_PATH is None or not os.path.exists(DATA_PATH):
    raise Exception(f'Configure your input data folder location in {DATA_PATH} before continuing!')
