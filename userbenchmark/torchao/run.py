import argparse

from typing import List

CI_ARGS = [
    # TIMM
    ["--timm", "--inference", "--bfloat16", "--quantization", "noquant", "--output", ".userbenchmark/torchao/torchao_noquant_timm_bfloat16_inference_cuda_performance.csv"],
    # ["--timm", "--inference", "--bfloat16", "--quantization", "int8dynamic", "--output", ".userbenchmark/torchao/torchao_int8dynamic_timm_bfloat16_inference_cuda_performance.csv"],
    # ["--timm", "--inference", "--bfloat16", "--quantization", "int8weightonly", "--output", ".userbenchmark/torchao/torchao_int8weightonly_timm_bfloat16_inference_cuda_performance.csv"],
    # ["--timm", "--inference", "--bfloat16", "--quantization", "autoquant", "--output", ".userbenchmark/torchao/torchao_autoquant_timm_bfloat16_inference_cuda_performance.csv"],
]


def _run_pt2_args(pt2_args: List[str]) -> str:
    from userbenchmark.dynamo.run import run as run_pt2_benchmark
    run_pt2_benchmark(pt2_args)


def run(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", actions="store_true", help="Run the CI workflow")
    args, pt2_args = parser.parse_known_args(args)

    if args.ci:
        group_pt2_args = CI_ARGS
    else:
        group_pt2_args = [pt2_args]
    
    output_files = [_run_pt2_args(pt2_args) for pt2_args in group_pt2_args]
    print("\n".join(output_files))
