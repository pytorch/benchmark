import argparse

from userbenchmark.utils import get_output_dir
from typing import List

from . import BM_NAME
from .upload import post_ci_process
OUTPUT_DIR = get_output_dir(BM_NAME)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CI_ARGS = [
    # TIMM
    ["--progress", "--timm", "--performance", "--inference", "--bfloat16", "--quantization", "noquant", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_noquant_timm_models_bfloat16_inference_cuda_performance.csv').resolve())}"],
    ["--progress", "--timm", "--accuracy", "--inference", "--bfloat16", "--quantization", "noquant", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_noquant_timm_models_bfloat16_inference_cuda_accuracy.csv').resolve())}"],
    ["--progress", "--timm", "--performance", "--inference", "--bfloat16", "--quantization", "int8dynamic", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_int8dynamic_timm_models_bfloat16_inference_cuda_performance.csv').resolve())}"],
    ["--progress", "--timm", "--accuracy", "--inference", "--bfloat16", "--quantization", "int8dynamic", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_int8dynamic_timm_models_bfloat16_inference_cuda_accuracy.csv').resolve())}"],
    ["--progress", "--timm", "--performance", "--inference", "--bfloat16", "--quantization", "int8weightonly", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_int8weightonly_timm_models_bfloat16_inference_cuda_performance.csv').resolve())}"],
    ["--progress", "--timm", "--accuracy", "--inference", "--bfloat16", "--quantization", "int8weightonly", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_int8weightonly_timm_models_bfloat16_inference_cuda_accuracy.csv').resolve())}"],
    ["--progress", "--timm", "--performance", "--inference", "--bfloat16", "--quantization", "autoquant", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_autoquant_timm_models_bfloat16_inference_cuda_performance.csv').resolve())}"],
    ["--progress", "--timm", "--accuracy", "--inference", "--bfloat16", "--quantization", "autoquant", "--output", f"{str(OUTPUT_DIR.joinpath('torchao_autoquant_timm_models_bfloat16_inference_cuda_accuracy.csv').resolve())}"],
]


def _get_output(pt2_args):
    if "--output" in pt2_args:
        output_index = pt2_args.index("--output")
        return pt2_args[output_index + 1]
    return "not_available"



def _run_pt2_args(pt2_args: List[str]) -> str:
    from userbenchmark.dynamo.run import run as run_pt2_benchmark
    print(f"=================== [TORCHAO] Running PT2 Benchmark Runner with Args: {pt2_args} ===================")
    run_pt2_benchmark(pt2_args)
    return _get_output(pt2_args)

def run(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Run the CI workflow")
    parser.add_argument("--dashboard", action="store_true", help="Update the output files to prepare the S3 upload and dashboard.")
    args, pt2_args = parser.parse_known_args(args)

    if args.ci:
        group_pt2_args = CI_ARGS
    else:
        group_pt2_args = [pt2_args]
    
    output_files = [_run_pt2_args(pt2_args) for pt2_args in group_pt2_args]
    # Post-processing
    if args.dashboard:
        post_ci_process(output_files)
    print("\n".join(output_files))
