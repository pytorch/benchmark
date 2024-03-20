import logging
import warnings

from torchbenchmark import add_path
from . import DYNAMOBENCH_PATH

try:
    # OSS Import
    with add_path(str(DYNAMOBENCH_PATH)):
        from torchbench import setup_torchbench_cwd, TorchBenchmarkRunner
        from common import main
except ImportError:
    # Meta Internal Import
    from caffe2.benchmarks.dynamo.torchbench import setup_torchbench_cwd, TorchBenchmarkRunner
    from caffe2.benchmarks.dynamo.common import main

from typing import List

def run(args: List[str]):
    original_dir = setup_torchbench_cwd()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), original_dir, args)
