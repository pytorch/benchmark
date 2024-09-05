import argparse
import itertools
from typing import List

from userbenchmark.utils import get_output_dir

from . import BM_NAME
from .upload import post_ci_process

OUTPUT_DIR = get_output_dir(BM_NAME)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def _get_ci_args(
    backend: str, modelset: str, dtype, mode: str, device: str, experiment: str
) -> List[List[str]]:
    if modelset == "timm":
        modelset_full_name = "timm_models"
    else:
        modelset_full_name = modelset
    output_file_name = f"torchao_{backend}_{modelset_full_name}_{dtype}_{mode}_{device}_{experiment}.csv"
    ci_args = [
        "--progress",
        f"--{modelset}",
        "--quantization",
        f"{backend}",
        f"--{mode}",
        f"--{dtype}",
        f"--{experiment}",
        "--output",
        f"{str(OUTPUT_DIR.joinpath(output_file_name).resolve())}",
    ]
    return ci_args


def _get_full_ci_args(modelset: str) -> List[List[str]]:
    backends = ["autoquant", "int8dynamic", "int8weightonly", "noquant"]
    modelset = [modelset]
    dtype = ["bfloat16"]
    mode = ["inference"]
    device = ["cuda"]
    experiment = ["performance", "accuracy"]
    cfgs = itertools.product(*[backends, modelset, dtype, mode, device, experiment])
    return [_get_ci_args(*cfg) for cfg in cfgs]


def _get_output(pt2_args):
    if "--output" in pt2_args:
        output_index = pt2_args.index("--output")
        return pt2_args[output_index + 1]
    return "not_available"


def _run_pt2_args(pt2_args: List[str]) -> str:
    from userbenchmark.dynamo.run import run as run_pt2_benchmark

    print(
        f"=================== [TORCHAO] Running PT2 Benchmark Runner with Args: {pt2_args} ==================="
    )
    run_pt2_benchmark(pt2_args)
    return _get_output(pt2_args)


def run(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Run the CI workflow")
    parser.add_argument("--timm", action="store_true", help="Run the TIMM CI workflow")
    parser.add_argument(
        "--huggingface", action="store_true", help="Run the Huggingface CI workflow"
    )
    parser.add_argument(
        "--torchbench", action="store_true", help="Run the Torchbench CI workflow"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Update the output files to prepare the S3 upload and dashboard.",
    )
    args, pt2_args = parser.parse_known_args(args)

    if args.ci:
        if args.timm:
            benchmark_args = _get_full_ci_args(modelset="timm")
        elif args.huggingface:
            benchmark_args = _get_full_ci_args(modelset="huggingface")
        elif args.torchbench:
            benchmark_args = _get_full_ci_args(modelset="torchbench")
        else:
            raise RuntimeError(
                "CI mode must run with --timm, --huggingface, or --torchbench"
            )
    else:
        benchmark_args = [pt2_args]

    output_files = [_run_pt2_args(args) for args in benchmark_args]
    # Post-processing
    if args.dashboard:
        post_ci_process(output_files)
    print("\n".join(output_files))
