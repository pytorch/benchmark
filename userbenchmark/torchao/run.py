import argparse

from . import BM_NAME
from userbenchmark.utils import get_output_dir
from typing import List

OUTPUT_DIR = get_output_dir(BM_NAME)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CI_ARGS = [
    # TIMM
    ["--progress", "--timm", "--performance", "--inference", "--bfloat16", "--quantization", "noquant", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_noquant_timm_bfloat16_inference_cuda_performance.csv').resolve())}"],
    # ["--progress", "--timm", "--performance", "--inference", "--bfloat16", "--quantization", "int8dynamic", "--output", ".userbenchmark/torchao/torchao_int8dynamic_timm_bfloat16_inference_cuda_performance.csv"],
    # ["--progress", "--timm", "--performance", "--inference", "--bfloat16", "--quantization", "int8weightonly", "--output", ".userbenchmark/torchao/torchao_int8weightonly_timm_bfloat16_inference_cuda_performance.csv"],
    # ["--progress", "--timm", "--performance", "--inference", "--bfloat16", "--quantization", "autoquant", "--output", ".userbenchmark/torchao/torchao_autoquant_timm_bfloat16_inference_cuda_performance.csv"],
]


def _get_output(pt2_args):
    if "--output" in pt2_args:
        output_index = pt2_args.index("--output")
        return pt2_args[output_index + 1]
    return "not_available"


def _run_pt2_args(pt2_args: List[str]) -> str:
    from userbenchmark.dynamo.run import run as run_pt2_benchmark
    run_pt2_benchmark(pt2_args)
    return _get_output(pt2_args)

def run(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Run the CI workflow")
    args, pt2_args = parser.parse_known_args(args)

    if args.ci:
        group_pt2_args = CI_ARGS
    else:
        group_pt2_args = [pt2_args]
    
    output_files = [_run_pt2_args(pt2_args) for pt2_args in group_pt2_args]
    print("\n".join(output_files))
