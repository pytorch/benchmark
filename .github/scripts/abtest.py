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
    import torchbenchmark.util.gitutils as gitutils
    from userbenchmark import list_userbenchmarks

USERBENCHMARK_OUTPUT_PATH = os.path.join(REPO_ROOT, ".userbenchmark")
# only preserve the first 10 chars of the git hash
GIT_HASH_LEN = 10

def cleanup():
    packages = ["torch"]
    print("Cleaning up torch packages...", end="", flush=True)
    CLEANUP_ROUND = 5
    # Clean up multiple times to make sure the packages are all uninstalled
    for _ in range(CLEANUP_ROUND):
        command = "pip uninstall -y " + " ".join(packages) + " || true"
        subprocess.check_call(command, shell=False)
    print("done")

def run_commit(repo_path: str, commit: str, bm_name: str, skip_build: bool=False) -> Path:
    "Run the userbenchmark on the commit. Return the metrics output file path."
    # build the pytorch commit if required
    if not skip_build:
        cleanup()
        build_pytorch_commit(repo_path, commit)
    # run_benchmark
    return run_benchmark(bm_name)

def validate_benchmark_output(bm_output: Path, bm_name: str):
    with open(bm_output, "r") as bmobj:
        output = json.load(bmobj)
    assert output["name"] == bm_name, f"Expected benchmark name {bm_name}, getting {output['name']}."
    assert "environ" in output and "pytorch_git_version" in output["environ"], \
        f"Missing pytorch git version in {bm_output}."
    assert "metrics" in output, f"Missing definition of metrics in {bm_output}."

def run_benchmark(bm_name: str) -> Path:
    def find_latest_output(p: str) -> Optional[Path]:
        if not os.path.exists(p) or not os.path.isdir(p):
            return None
        json_files = [ os.path.join(p, jf) for jf in sorted(os.listdir(p)) if jf.endswith(".json") ]
        if len(json_files) == 0:
            return None
        return json_files[-1]
    command = [sys.executable, "run_benchmark.py", bm_name]
    try:
        subprocess.check_call(command, cwd=REPO_ROOT, shell=False)
    except subprocess.CalledProcessError as e:
        print(f"Failed to call userbenchmark {command}. Error: {e}")
        sys.exit(1)
    output_path = os.path.join(USERBENCHMARK_OUTPUT_PATH, bm_name)
    output_file = find_latest_output(output_path)
    if not output_file:
        print(f"Benchmark {bm_name} didn't print any output. Exit.")
        sys.exit(1)
    validate_benchmark_output(output_file, bm_name)
    return output_file

def setup_build_env(env) -> Dict[str, str]:
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
    print(f"Building pytorch commit {commit} ...", end="", flush=True)
    # Check if version.py exists, if it does, remove it.
    # This is to force pytorch update the version.py file upon incremental compilation
    version_py_path = os.path.join(repo_path, "torch/version.py")
    if os.path.exists(version_py_path):
        os.remove(version_py_path)
    try:
        # some packages are not included in the wheel, so use `develop`, not `install`
        command = ["python", "setup.py", "develop"]
        # setup environment variables
        build_env = setup_build_env(os.environ.copy())
        subprocess.check_call(command, cwd=repo_path, env=build_env, shell=False)
        command_testbuild = ["python", "-c", "'import torch'"]
        subprocess.check_call(command_testbuild, cwd=os.environ["HOME"], env=build_env, shell=False)
    except subprocess.CalledProcessError:
        # If failed, remove the build directory, then try again
        build_path = os.path.join(repo_path, "build")
        if os.path.exists(build_path):
            shutil.rmtree(build_path)
        subprocess.check_call(command, cwd=repo_path, env=build_env, shell=False)
    print("done")

def process_test_result(result_a: Path, result_b: Path, output_dir: str) -> str:
    def validate_results(a, b) -> bool:
        metrics = a["metrics"].keys()
        return sorted(metrics) == sorted(b["metrics"])
    # check two results are different files
    assert not result_a == result_b, f"Path {result_a} and {result_b} are the same. Exit."
    # validate results
    with open(result_a, "r") as fa:
        a = json.load(fa)
    with open(result_b, "r") as fb:
        b = json.load(fb)
    assert validate_results(a, b), f"Result validation failed for {result_a} and {result_b}."
    # print result in csv format
    header = ["Metric", a["environ"]["pytorch_git_version"][:GIT_HASH_LEN], b["environ"]["pytorch_git_version"][:GIT_HASH_LEN]]
    out = [header]
    metrics = a["metrics"].keys()
    for m in sorted(metrics):
        val = [m]
        val.append(a["metrics"][m])
        val.append(b["metrics"][m])
        out.append(val)
    out = "\n".join([";".join(map(lambda y: str(y), x)) for x in out])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "control.json"), "w") as fcontrol:
        json.dump(a, fcontrol, indent=4)
    with open(os.path.join(output_dir, "treatment.json"), "w") as ftreatment:
        json.dump(b, ftreatment, indent=4)
    with open(os.path.join(output_dir, "result.csv"), "w") as fout:
        fout.write(out + "\n")
    return out

if __name__ == "__main__":
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
    result_a = run_commit(args.pytorch_repo, args.base, args.userbenchmark, args.skip_build)
    result_b = run_commit(args.pytorch_repo, args.head, args.userbenchmark, args.skip_build)
    compare_result = process_test_result(result_a, result_b, args.output_dir)
    print(compare_result)
