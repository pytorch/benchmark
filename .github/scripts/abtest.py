"""
This script runs userbenchmarks abtest upon two PyTorch versions.
"""
import argparse
import os
import subprocess
import shutil
import sys
import json
from pathlib import Path
from bmutils import REPO_ROOT, add_path
from typing import Dict, Optional

with add_path(REPO_ROOT):
    import torchbenchmark.util.gitutils
    from userbenchmark import list_userbenchmarks

USERBENCHMARK_OUTPUT_PATH = os.path.join(REPO_ROOT, ".userbenchmark")

def run_commit(repo_path: str, commit: str, bm_name: str, skip_build: bool=False) -> Path:
    "Run the userbenchmark on the commit. Return the metrics output file path."
    # build the pytorch commit if required
    if not skip_build:
        build_pytorch_commit(repo_path, commit)
    # run_benchmark
    return run_benchmark(bm_name)

def validate_benchmark_output(bm_output: Path, bm_name: str):
    with open(bm_output, "r") as bmobj:
        output = json.load(bmobj)
    assert output["name"] == bm_name, f"Expected benchmark name {bm_name}, getting {output['name']}."

def run_benchmark(bm_name: str) -> Path:
    def find_latest_output(p) -> Optional[Path]:
        pass
    command = ["python", "run_benchmark.py", bm_name]
    subprocess.check_call(command, cwd=REPO_ROOT, shell=True)
    output_path = os.path.join(USERBENCHMARK_OUTPUT_PATH, bm_name)
    output_file = find_latest_output(output_path)
    if not output_file:
        print(f"Benchmark {bm_name} didn't print any output.")
        sys.exit(1)
    validate_benchmark_output(output_file)
    return output_file

def setup_build_env(self, env) -> Dict[str, str]:
    env["USE_CUDA"] = "1"
    env["BUILD_CAFFE2_OPS"] = "0"
    # Do not build the test
    env["BUILD_TEST"] = "0"
    env["USE_MKLDNN"] = "1"
    env["USE_MKL"] = "1"
    env["USE_CUDNN"] = "1"
    env["CMAKE_PREFIX_PATH"] = env["CONDA_PREFIX"]
    return env

def build_pytorch_commit(repo_path: str, commit: str):
    # checkout pytorch commit
    print(f"Checking out pytorch commit {commit} ...", end="", flush=True)
    gitutils.checkout_git_commit(repo_path, commit)
    print("done.")
    # build pytorch
    print(f"Building pytorch commit {commit.sha} ...", end="", flush=True)
    # Check if version.py exists, if it does, remove it.
    # This is to force pytorch update the version.py file upon incremental compilation
    version_py_path = os.path.join(repo_path, "torch/version.py")
    if os.path.exists(version_py_path):
        os.remove(version_py_path)
    try:
        command = ["python", "setup.py", "install"]
        # setup environment variables
        build_env = setup_build_env(os.environ.copy())
        subprocess.check_call(command, cwd=repo_path, env=build_env, shell=True)
        command_testbuild = ["python" "-c" "'import torch'"]
        subprocess.check_call(command_testbuild, cwd=os.environ["HOME"], env=build_env, shell=True)
    except subprocess.CalledProcessError:
        # If failed, remove the build directory, then try again
        build_path = os.path.join(repo_path, "build")
        if os.path.exists(build_path):
            shutil.rmtree(build_path)
        subprocess.check_call(command, cwd=repo_path, env=build_env, shell=True)
    print("done")

def process_test_result(result_a: Path, result_b: Path, output_dir: str) -> str:
    # check two results are different files
    # validate result
    pass

if __name__ == "__main__":
    print(list_userbenchmarks())
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-repo", required=True, type=str, help="PyTorch repo path")
    parser.add_argument("--base", required=True, type=str, help="PyTorch base commit")
    parser.add_argument("--head", required=True, type=str, help="PyTorch head commit")
    parser.add_argument("--userbenchmark", required=True, type=str, help="Name of the userbenchmark to run")
    parser.add_argument("--output-dir", required=True, type=str, help="Output dir path")
    parser.add_argument("--skip-build", action="store_true", help="Skip PyTorch build")
    args = parser.parse_args()
    # sanity checks
    assert args.userbenchmark in list_userbenchmarks(), f"Available userbenchmark list: {list_userbenchmarks()}, " \
                                                        f"but you specified {args.userbenchmark}."
    if not args.skip_build:
        assert Path(args.pytorch_repo).is_dir(), f"Specified PyTorch repo dir {args.pytorch_repo} doesn't exist."
        commits = gitutils.get_git_commits(args.pytorch_repo, args.base, args.head)
        assert commits, f"Can't find git commit {args.base} or {args.head} in repo {args.pytorch_repo}"
    result_a = run_commit(args.pytorch_repo, args.base, args.userbenchmark)
    result_b = run_commit(args.pytorch_repo, args.head, args.userbenchmark)
    compare_result = process_test_result(result_a, result_b, args.output_dir)
    print(compare_result)
