import logging
import warnings
import sys

from torchbenchmark import add_path, REPO_PATH
from torchbenchmark.util.framework.huggingface.extended_configs import (
    list_extended_huggingface_models,
)
from torchbenchmark.util.framework.timm.extended_configs import (
    list_extended_timm_models,
)

DYNAMOBENCH_PATH = REPO_PATH.joinpath("userbenchmark", "dynamo", "dynamobench")

from typing import List, Optional

def _get_model_set_by_model_name(args: List[str]) -> str:
    if "--huggingface" in args:
        args.remove("--huggingface")
        return "huggingface"
    if "--timm" in args:
        args.remove("--timm")
        return "timm"
    if "--torchbench" in args:
        args.remove("--torchbench")
        return "torchbench"
    if "--only" in args:
        only_index = args.index("--only")
        model_name = args[only_index + 1]
        if model_name in list_extended_huggingface_models():
            return "huggingface"
        if model_name in list_extended_timm_models():
            return "timm"
    return "torchbench"

def _run_huggingface(args: List[str]) -> None:
    try:
        # OSS Import
        with add_path(str(DYNAMOBENCH_PATH)):
            from huggingface import HuggingfaceRunner
            from common import main
    except ImportError:
        # Meta Internal Import
        from caffe2.benchmarks.dynamo.huggingface import HuggingfaceRunner
        from caffe2.benchmarks.dynamo.common import main
    main(runner=HuggingfaceRunner(), args=args)


def _run_timm(args: List[str]) -> None:
    try:
        # OSS Import
        with add_path(str(DYNAMOBENCH_PATH)):
            from timm_models import TimmRunner
            from common import main
    except ImportError:
        # Meta Internal Import
        from caffe2.benchmarks.dynamo.timm_models import TimmRunner
        from caffe2.benchmarks.dynamo.common import main
    main(runner=TimmRunner(), args=args)


def _run_torchbench(args: List[str]) -> None:
    try:
        # OSS Import
        with add_path(str(DYNAMOBENCH_PATH)):
            from torchbench import setup_torchbench_cwd, TorchBenchmarkRunner
            from common import main
    except ImportError:
        # Meta Internal Import
        from caffe2.benchmarks.dynamo.torchbench import setup_torchbench_cwd, TorchBenchmarkRunner
        from caffe2.benchmarks.dynamo.common import main
    original_dir = setup_torchbench_cwd()
    main(TorchBenchmarkRunner(), original_dir, args)


def run(args: Optional[List[str]]=None):
    if args is None:
        args = sys.argv[1:]
    model_set = _get_model_set_by_model_name(args)
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    if model_set == "huggingface":
        _run_huggingface(args)
    elif model_set == "timm":
        _run_timm(args)
    else:
        _run_torchbench(args)
