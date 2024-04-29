import logging
import json
import triton
from typing import List, Any, Dict

from . import BM_NAME
from .run import parse_args, _run
from userbenchmark.utils import get_default_output_json_path, get_output_json
from torchbenchmark.util.triton_op import BenchmarkOperatorResult

CI_TESTS = [
    ["--op", "softmax", "--num-inputs", "10"],
]

def ci_result_to_userbenchmark_json(ci_metrics: List[BenchmarkOperatorResult]) -> Dict[str, Any]:
    result = {}
    for metric in ci_metrics:
        result.update(metric.userbenchmark_dict)
    return result

def run_ci():
    ci_result = []
    for test_opts in CI_TESTS:
        logging.info(f"Running the test opts: {test_opts}")
        test_args, test_extra_args = parse_args(test_opts)
        metrics = _run(test_args, test_extra_args)
        ci_result.append(metrics)
    result = ci_result_to_userbenchmark_json(ci_result)
    result_with_environ = get_output_json(BM_NAME, result)
    result_with_environ["environ"]["triton_version"] = triton.__version__
    output_file = get_default_output_json_path(BM_NAME)
    json_str = json.dumps(result_with_environ, indent=4)

    print(json_str)
    with open(output_file, 'w') as f:
        f.write(json_str)

    logging.info(f"Benchmark result saved to {output_file}.")
