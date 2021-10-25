"""
This script reads from a PyTorch benchmarking directory and generates a yaml file
that drives the bisector to run tests on the specified PyTorch commits.
This only works on V1 and later benchmark, V0 is not supported.
"""
import os
import re
import git
import json
import yaml
import argparse
import dataclasses
from pathlib import Path

# We will generate bisection config for tests with performance change > 7%
PERF_CHANGE_THRESHOLD = 7
# Assume the nightly branch commit message is in the following format
# Hash in the parentheses links to the commit on the master branch
NIGHTLY_COMMIT_MSG = "nightly release \((.*)\)"
# Timeout of the bisection job in hours
PERF_TEST_TIMEOUT_THRESHOLD = 120
GITHUB_ISSUE_TEMPLATE = """
TorchBench CI has detected a performance signal.

Base PyTorch version: {start_version}

Base PyTorch commit: {start}

Affected PyTorch version: {end_version}

Affected PyTorch commit: {end}

Affected Tests:
{test_details}

cc @xuzhao9
"""

@dataclasses.dataclass
class PyTorchVer:
    version: str
    commit: str

def exist_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_pytorch_main_commit(pytorch_repo, nightly_commit):
    repo = git.Repo(pytorch_repo)
    msg = repo.commit(nightly_commit).message
    nightly_commit_regex = re.compile(NIGHTLY_COMMIT_MSG)
    return nightly_commit_regex.search(msg).group(1)

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

def get_pytorch_version(pytorch_src_path, json_path):
    with open(json_path, "r") as json_obj:
        bm_result = json.load(json_obj)
    nightly_git_version = bm_result["machine_info"]["pytorch_git_version"]
    # get main git commit by git version
    main_commit = get_pytorch_main_commit(pytorch_src_path, nightly_git_version)
    pytorch_ver = PyTorchVer(version=bm_result["machine_info"]["pytorch_version"],
                             commit=main_commit)
    return pytorch_ver

def get_workflow_id(workflow_dir):
    prefix = "gh"
    prefix_loc = workflow_dir.find(prefix)
    return int(workflow_dir[prefix_loc + len(prefix):])

# Compare the tests and generate a list of tests whose perf change larger than threshold
def generate_bisection_tests(base, tip, flaky_tests):
    def get_test_stats(bm):
        ret = {}
        for benchmark in bm["benchmarks"]:
            name = benchmark["name"]
            ret[name] = benchmark["stats"]["mean"]
        return ret
    base_tests = get_test_stats(base)
    tip_tests = get_test_stats(tip)
    signals = []
    signal_details = {}
    for benchmark, tip_latency in tip_tests.items():
        base_latency = base_tests.get(benchmark, None)
        # Skip the flaky tests
        if benchmark in flaky_tests:
            continue
        if base_latency is None:
            # This benchmark is new or was failing, so there is no prior point
            # of reference against which to compare.
            continue
        delta_percent = (tip_latency - base_latency) / base_latency * 100
        if abs(delta_percent) >= PERF_CHANGE_THRESHOLD:
            signals.append(benchmark)
            signal_details[benchmark] = delta_percent
    return (signals, signal_details)

def generate_bisection_config(base_file, tip_file, flaky_tests):
    result = {}
    with open(base_file, "r") as bf:
        base = json.load(bf)
    with open(tip_file, "r") as tf:
        tip = json.load(tf)
    result["start_version"] = base["machine_info"]["pytorch_version"]
    result["start"] = base["machine_info"]["pytorch_git_version"]
    result["end_version"] = tip["machine_info"]["pytorch_version"]
    result["end"] = tip["machine_info"]["pytorch_git_version"]
    result["threshold"] = PERF_CHANGE_THRESHOLD
    result["direction"] = "both"
    result["timeout"] = PERF_TEST_TIMEOUT_THRESHOLD
    (result["tests"], result["details"]) = generate_bisection_tests(base, tip, flaky_tests)
    return result

def generate_gh_issue(ghi_fpath, result):
    ghi_config = result
    ghi_config["test_details"] = ""
    for test, delta in result["details"].items():
        sign = "+" if delta > 0 else ""
        ghi_config["test_details"] += f"- {test}: {sign}{delta:.5f}%\n"
    ghi_body = GITHUB_ISSUE_TEMPLATE.format(**ghi_config)
    with open(ghi_fpath, "w") as f:
        f.write(ghi_body)

# Setup TORCHBENCH_PERF_SIGNAL to enable follow-up steps
def setup_gh_env(affected_pytorch_version):
    fname = os.environ["GITHUB_ENV"]
    content = f"TORCHBENCH_PERF_SIGNAL='{affected_pytorch_version}'\n"
    with open(fname, 'a') as fo:
        fo.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-dir",
                        required=True,
                        help="PyTorch source directory",
                        type=exist_dir_path)
    parser.add_argument("--benchmark-dir",
                        required=True,
                        help="PyTorch benchmark result directory",
                        type=exist_dir_path)
    parser.add_argument("--out",
                        required=True,
                        help="Result output file")
    parser.add_argument("--github-issue",
                        help="Setup environment variables and GitHub Issue file for GitHub Actions")
    parser.add_argument("--flaky-test-list",
                        default="v1-flaky-tests.txt",
                        help="Setup environment variables and GitHub Issue file for GitHub Actions")

    args = parser.parse_args()
    # input directory
    input_dir = Path(args.benchmark_dir)
    tip_json_file = find_latest_nonempty_json(input_dir)
    assert tip_json_file, "The input benchmark directory must contain a non-empty json file!"
    tip_version = get_pytorch_version(args.pytorch_dir, tip_json_file)
    parent_dir = input_dir.parent
    base_benchmark_dirs = list(filter(lambda x: get_workflow_id(x) < get_workflow_id(os.path.basename(input_dir)),
                                         os.listdir(parent_dir)))

    # Search from the latest to the earliest
    base_benchmark_dirs.sort(reverse=True)
    base_benchmark_paths = [ os.path.join(parent_dir, name) for name in base_benchmark_dirs if os.path.isdir(os.path.join(parent_dir, name)) ]
    result = {}

    # Read the flaky test list
    flaky_tests = []
    current_dir = os.path.dirname(os.path.realpath(__file__))
    flaky_test_file_path = os.path.join(current_dir, args.flaky_test_list)
    with open(flaky_test_file_path, "r") as flaky_test_fp:
        flaky_tests = flaky_test_fp.read().splitlines()

    # Use the latest benchmark result with a different version than tip
    for bm in base_benchmark_paths:
        json_file = find_latest_nonempty_json(bm)
        if json_file:
            base_version = get_pytorch_version(args.pytorch_dir, json_file)
            if base_version.commit != tip_version.commit:
                result = generate_bisection_config(json_file, tip_json_file, flaky_tests)
                break
    with open(args.out, "w") as fo:
        yaml.dump(result, fo)
    # If there is at least one regressing test, setup the Bisection GitHub Action workflow
    if args.github_issue and "tests" in result:
        setup_gh_env(result["end_version"])
        generate_gh_issue(args.github_issue, result)
