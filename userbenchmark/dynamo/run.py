import logging
import warnings

from .torchbench import setup_torchbench_cwd, TorchBenchmarkRunner
try:
    from .common import main
except ImportError:
    from common import main

from typing import List

def run(args: List[str]):
    original_dir = setup_torchbench_cwd()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), original_dir, args=args)
