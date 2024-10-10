import json
import logging
from typing import Any, Dict, List

import triton  # @manual=fbcode//triton:triton
from torchbenchmark.util.triton_op import BenchmarkOperatorResult
from userbenchmark.utils import get_default_output_json_path, get_output_json

from . import BM_NAME
from .run import _run, get_parser

CI_TESTS: Dict[str, List[str]] = {
    # baseline: sdpa
    "flash_attention": [
        "--op",
        "flash_attention",
        "--d-head",
        "128",
        "--only",
        "sdpa,triton_tutorial_flash_v2",
    ],
    "softmax": ["--op", "softmax", "--num-inputs", "10"],
    "fp8_gemm": [
        "--op",
        "fp8_gemm",
        "--llama",
        "--only",
        "torch_fp8_gemm, triton_fp8_gemm",
    ],
    "fp8_gemm_blockwise": [
        "--op",
        "fp8_gemm_blockwise",
        "--llama",
        "--only",
        "_cutlass, _triton",
    ],
    "fp8_gemm_rowwise": [
        "--op",
        "fp8_gemm_rowwise",
        "--llama",
        "--only",
        "_cutlass,_triton,_cublass",
    ],
    "gemm": [
        "--op",
        "gemm",
        "--only",
        "aten_matmul,pt2_cutlass_matmul,triton_tutorial_matmul",
        "--precision",
        "bf16",
        "--llama",
    ],
}


def ci_result_to_userbenchmark_json(
    ci_metrics: List[BenchmarkOperatorResult],
) -> Dict[str, Any]:
    result = {}
    for metric in ci_metrics:
        result.update(metric.userbenchmark_dict)
    return result


def run_ci():
    ci_result = []
    for op, test_opts in CI_TESTS.items():
        logging.info(f"Running the test opts: {op}")
        test_args, test_extra_args = get_parser(test_opts).parse_known_args(test_opts)
        metrics = _run(test_args, test_extra_args)
        ci_result.append(metrics)
    result = ci_result_to_userbenchmark_json(ci_result)
    result_with_environ = get_output_json(BM_NAME, result)
    result_with_environ["environ"]["triton_version"] = triton.__version__
    output_file = get_default_output_json_path(BM_NAME)
    json_str = json.dumps(result_with_environ, indent=4)

    print(json_str)
    with open(output_file, "w") as f:
        f.write(json_str)

    logging.info(f"Benchmark result saved to {output_file}.")
