import logging
import warnings
import sys

from torchbenchmark import add_path, REPO_PATH

DYNAMOBENCH_PATH = REPO_PATH.joinpath("userbenchmark", "dynamo", "dynamobench")

try:
    # OSS Import
    with add_path(str(DYNAMOBENCH_PATH)):
        from torchbench import setup_torchbench_cwd, TorchBenchmarkRunner
        from common import main
except ImportError:
    # Meta Internal Import
    from caffe2.benchmarks.dynamo.torchbench import setup_torchbench_cwd, TorchBenchmarkRunner
    from caffe2.benchmarks.dynamo.common import main

from typing import List, Optional

def run(args: Optional[List[str]]=None):
    if args is None:
        args = sys.argv[1:]
    original_dir = setup_torchbench_cwd()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), original_dir, args)
