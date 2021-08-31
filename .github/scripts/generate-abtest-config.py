"""
This script reads from a PyTorch benchmarking directory and generates a yaml file
that drives the bisector to run tests on the specified PyTorch commits.
This only works on V1 benchmark, V0 is not supported.
"""
import os
import json
import yaml
import argparse
import dataclasses
from pathlib import Path

# We will generate bisection config for tests with performance change > 7%
PERF_CHANGE_THRESHOLD = 7
# Timeout of the bisection job in hours
PERF_TEST_TIMEOUT_THRESHOLD = 120

@dataclasses.dataclass
class PyTorchVer:
    version: str
    commit: str

def exist_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def find_latest_nonempty_json(path):
    json_files = list(filter(lambda x: x.endswith(".json"), os.listdir(path)))
    json_files.sort(reverse=True)
    for f in json_files:
        # Return the first non-empty json file
        json_path = os.path.join(path, f)
        if os.path.exists(json_path) and os.stat(json_path).st_size:
            return json_path
    print(f"Can't find non-empty json files in path: {path}")
    return None

def get_pytorch_version(json_path):
    with open(json_path, "r") as json_obj:
        bm_result = json.load(json_obj)
    pytorch_ver = PyTorchVer(version=bm_result["machine_info"]["pytorch_version"],
                             commit=bm_result["machine_info"]["pytorch_git_version"])
    return pytorch_ver

# Compare the tests and generate a list of tests whose perf change larger than threshold
def generate_bisection_tests(base, tip):
    def get_test_stats(bm):
        ret = {}
        for benchmark in bm["benchmarks"]:
            name = benchmark["name"]
            ret[name] = benchmark["stats"]["mean"]
        return ret
    base_tests = get_test_stats(base)
    tip_tests = get_test_stats(tip)
    result = []
    for benchmark, tip_latency in tip_tests.items():
        base_latency = base_tests.get(benchmark, None)
        if base_latency is None:
            # This benchmark is new or was failing, so there is no prior point
            # of reference against which to compare.
            continue

        if abs(tip_latency - base_latency) / min(base_latency, tip_latency) >= PERF_CHANGE_THRESHOLD:
            result.append(benchmark)
    return result

def generate_bisection_config(base_file, tip_file):
    result = {}
    with open(base_file, "r") as bf:
        base = json.load(bf)
    with open(tip_file, "r") as tf:
        tip = json.load(tf)
    result["start_version"] = base["machine_info"]["pytorch_version"]
    result["start"] = base["machine_info"]["pytorch_git_version"]
    result["end_version"] = base["machine_info"]["pytorch_version"]
    result["end"] = tip["machine_info"]["pytorch_git_version"]
    result["threshold"] = PERF_CHANGE_THRESHOLD
    result["direction"] = "both"
    result["timeout"] = PERF_TEST_TIMEOUT_THRESHOLD
    result["tests"] = generate_bisection_tests(base, tip)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir",
                        required=True,
                        help="PyTorch benchmark result directory",
                        type=exist_dir_path)
    parser.add_argument("--out",
                        required=True,
                        help="Result output file")
    args = parser.parse_args()
    # input directory
    input_dir = Path(args.benchmark_dir)
    tip_json_file = find_latest_nonempty_json(input_dir)
    assert tip_json_file, "The input benchmark directory must contains non-empty json file"
    tip_version = get_pytorch_version(tip_json_file)
    parent_dir = input_dir.parent
    all_benchmark_dirs = [ os.path.join(parent_dir, name) for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name)) ]
    all_benchmark_dirs.sort(reverse=True)
    result = {}
    # Search from the latest to the earliest
    # Use the latest benchmark result with a different version than tip
    for bm in all_benchmark_dirs:
        json_file = find_latest_nonempty_json(bm)
        if json_file:
            base_version = get_pytorch_version(json_file)
            if base_version.commit != tip_version.commit:
                result = generate_bisection_config(json_file, tip_json_file)
                break
    with open(args.out, "w") as fo:
        yaml.dump(result, fo)
