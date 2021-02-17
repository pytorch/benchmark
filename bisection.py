"""bisection.py
Runs bisection to determine PRs that cause performance regression.
Performance regression is defined by TorchBench score drop greater than the threshold.
By default, the torchvision and torchtext package version will be fixed to the latest version in the conda defaults channel.

Usage:
  python bisection.py --pytorch-src <PYTORCH_SRC_DIR> \
    --torchbench-src <TORCHBENCH_SRC_DIR> \
    --start <SHA> --end <SHA> --threshold <SCORE_THRESHOLD> \
    --timeout <TIMEOUT_IN_MINS> --output <OUTPUT_FILE_PATH>
"""

import os
import json
import yaml
import argparse
import typing
import re
import subprocess
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

from torchbenchmark.score.compute_score import compute_score
from torchbenchmark.util import gitutils
from torchbenchmark.util import torch_nightly

# Bisection Algorithm: for the bisection range [start, end]
# Step 1: Fetch commit list: [start, ..., mid, ..., end]
# Step 2: Put pair (start, end) into queue bisectq
# Step 3: Get the first pair (start, end) in bisectq.
# Step 4: Run benchmark for start and end. If abs(start.score-end.score)>threshold ...
#         ... and start/end are adjacent, add pair(start, end) to result
# Step 4: If abs(start.score-end.score)>threshold, but start/end are not adjacent ...
#         ... test benchmark on commit mid, and:
#               if abs(end.score - mid.score) > threshold: insert (mid, end) into the bisectq
#               if abs(start.score - mid.score) > threshold: insert (start, mid) into the bisectq
# Step 5: goto step 2 until bisectq is empty

# Workdir Organization:
# WORKDIR/
#    commit-shas.txt
#    <COMMIT-SHA1>.json
#    <COMMIT-SHA2>.json
#    ...

## Helper functions
def exist_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

TORCH_GITREPO="https://github.com/pytorch/pytorch.git"
TORCHBENCH_GITREPO="https://github.com/pytorch/benchmark.git"
TORCHBENCH_SCORE_CONFIG="torchbenchmark/score/configs/v0/config-v0.yaml"

## Class definitions
class Commit:
    sha: str
    ctime: str
    score: Optional[float]
    def __init__(self, sha, ctime, score):
        self.sha = sha
        self.ctime = ctime
        self.score = score

class TorchSource:
    srcpath: str
    commits: List[Commit]
    # Map from commit SHA to index in commits
    commit_dict: Dict[str, int]
    def __init__(self, srcpath: str):
        self.srcpath = srcpath
        self.commits = []
        self.commit_dict = dict()

    def prep(self) -> bool:
        repo_origin_url = gitutils.get_git_origin(self.srcpath)
        if not repo_origin_url == TORCH_GITREPO:
            print(f"Unmatched repo origin url: {repo_origin_url} with standard {TORCH_GITREPO}")
            return False
        # TODO: check conda packages dependencies are installed
        return True
    
    # Get all commits between start and end, save them in commits
    def init_commits(self, start: str, end: str) -> bool:
        commits = gitutils.get_git_commits(self.srcpath, start, end)
        if not commits or len(commits) < 2:
            print(f"Failed to retrieve commits from {start} to {end} in {self.srcpath}.")
            return False
        for count, commit in enumerate(commits):
            ctime = gitutils.get_git_commit_date(self.srcpath, commit)
            self.commits.append(Commit(sha=commit, ctime=ctime, score=None))
            self.commit_dict[commit] = count
        return True
    
    def get_mid_commit(self, left: Commit, right: Commit) -> Optional[Commit]:
        left_index = commit_dict[left.sha]
        right_index = commit_dict[right.sha]
        if right_index == left_index + 1:
            return None
        else:
            return commits[(left_index + right_index) / 2]

    def setup_build_env(self, env):
        env["USE_CUDA"] = "1"
        env["BUILD_CAFFE2_OPS"] = "0"
        env["USE_XNNPACK"] = "0"
        env["USE_MKLDNN"] = "1"
        env["USE_MKL"] = "1"
        env["USE_CUDNN"] = "1"
        env["CMAKE_PREFIX_PATH"] = env["CONDA_PREFIX"]
        return env

    def build(self, commit: Commit):
        # checkout pytorch commit
        gitutils.checkout_git_commit(self.srcpath, commit.sha)
        # setup environment variables
        build_env = self.setup_build_env(os.environ.copy())
        # build pytorch
        print(f"Building pytorch commit {commit.sha}...", end="", flush=True)
        command = "python setup.py install &> /dev/null"
        subprocess.check_call(command, cwd=self.srcpath, env=build_env, shell=True)
        print("done")

class TorchBench:
    srcpath: str # path to pytorch/benchmark source code
    branch: str
    timeout_limit: int # timeout limit in minutes
    workdir: str
    torch_src: TorchSource

    def __init__(self, srcpath: str,
                 torch_src: TorchSource,
                 workdir: str,
                 branch: str = "0.1"):
        self.srcpath = srcpath
        self.torch_src = torch_src
        self.branch = branch
        self.workdir = workdir

    def prep(self) -> bool:
        # Verify the code in srcpath is pytorch/benchmark
        repo_origin_url = gitutils.get_git_origin(self.srcpath)
        if not repo_origin_url == TORCHBENCH_GITREPO:
            return False
        # Checkout branch
        if not gitutils.checkout_git_branch(self.srcpath, self.branch):
            return False
        return True
 
    # Install dependencies such as torchtext and torchvision
    def install_deps(self, commit: Commit):
        # Find the matching torchtext/torchvision version
        # Sometimes the nightly wheel is unavailable, increase version until the first available date
        datetime_obj = datetime.strptime(commit.ctime.split(" ")[0], "%Y-%m-%d")
        present = datetime.now()
        packages = ["torchtext", "torchvision", "torchaudio"]
        nightly_wheel_urls = torch_nightly.get_nightly_wheel_urls(packages, datetime_obj)
        assert nightly_wheel_urls, f"Failed to get dependency wheels version: {commit.ctime} from nightly html"
        # Install the wheels
        wheels = [nightly_wheel_urls[pkg]["wheel"] for pkg in packages]
        command = "pip install --no-deps " + " ".join(wheels) + " &> /dev/null"
        print("Installing deps: " + command)
        subprocess.check_call(command, cwd=self.srcpath, shell=True)
    
    def run_benchmark(self, commit: Commit) -> str:
        # Benchmark output dir: self
        # Return the result json file
        output_dir = os.path.join(self.workdir, commit.sha)
        # If the directory exists, delete its contents
        if os.path.exists(output_dir):
            assert os.path.isdir(output_dir), "Must specify output directory: {output_dir}"
            filelist = [ f for f in os.listdir(output_dir) ]
            for f in filelist:
                os.remove(os.path.join(output_dir, f))
        else:
            os.mkdir(output_dir)
        command = f"bash .github/scripts/run-nodocker.sh {output_dir} &> {output_dir}/benchmark.log"
        subprocess.check_call(command, cwd=self.srcpath, shell=True)
        return output_dir

    def compute_score(self, result_dir: str) -> float:
        filelist = [ f for f in os.listdir(result_dir) if f.endswith(".json") ]
        assert len(filelist) > 0, f"Can't compute score in an empty directory {result_dir}."
        # benchmark data file
        data_file = os.path.join(result_dir, filelist[0])
        # configuration
        config_file = os.path.join(self.workdir, TORCHBENCH_SCORE_CONFIG)
        config = yaml.full_load(config_file)
        data = json.load(data_file)
        return compute_score(config, data)
    
    def get_score(self, commit: Commit) -> float:
        # Score is cached
        if commit.score is not None:
            return commit.score
        # Build pytorch
        # self.torch_src.build(commit)
        # Build benchmark and install deps
        self.install_deps(commit)
        # Run benchmark
        print(f"Running TorchBench for commit: {commit.sha} ...", end="", flush=True)
        result_dir = self.run_benchmark(commit)
        commit.score = self.compute_score(result_dir)
        print(f" score: {commit.score}")
        return commit.score
        
class TorchBenchBisection:
    start: str
    end: str
    workdir: str
    threshold: int
    output: str
    bisectq: List[Tuple[Commit, Commit]]
    result: List[Tuple[Commit, Commit]]
    torch_src: TorchSource
    bench: TorchBench
    output_json: str

    def __init__(self,
                 workdir: str,
                 torch_src: str,
                 bench_src: str,
                 start: str,
                 end: str,
                 threshold: int,
                 timeout: int,
                 output_json: str):
        self.workdir = workdir
        self.start = start
        self.end = end
        self.threshold = threshold
        self.timeout = timeout
        self.output = output_json
        self.bisectq = list()
        self.torch_src = TorchSource(srcpath = torch_src)
        self.bench = TorchBench(srcpath = bench_src,
                                torch_src = self.torch_src,
                                workdir = self.workdir)
        self.output_json = output_json

    # Left: older commit; right: newer commit
    def regression(self, left: Commit, right: Commit) -> bool:
        assert left.score is not None
        assert right.score is not None
        return left.score - right.score >= threshold

    def prep(self) -> bool:
        if not self.torch_src.prep() or not self.bench.prep():
            return False
        if not self.torch_src.init_commits(self.start, self.end):
            return False
        left_commit = self.torch_src.commits[0]
        right_commit = self.torch_src.commits[-1]
        self.bisectq.append((left_commit, right_commit))
        return True
        
    def run(self):
        while not len(self.bisectq) == 0:
            (left, right) = self.bisectq[0]
            left.score = self.bench.get_score(left)
            right.score = self.bench.get_score(right)
            if self.regression(left, right):
                mid = self.torch_src.get_mid_commit(left, right)
                if mid == None:
                    result.append((left, right))
                else:
                    mid.score = self.bench.get_score(mid_commit)
                    if self.regression(left, mid):
                        self.bisectq.append(left, mid)
                    if self.regression(mid, right):
                        self.bisectq.append(right, mid)
 
    def output(self):
        json_obj = dict()
        json_obj["start"] = self.start
        json_obj["end"] = self.end
        json_obj["threshold"] = self.threshold
        json_obj["timeout"] = self.timeout
        json_obj["torchbench_branch"] = self.bench.branch
        json_obj["result"] = self.result
        with open(self.output_json, 'w') as outfile:
            json.dump(json_obj, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",
                        help="bisection working directory",
                        type=exist_dir_path,
                        required=True)
    parser.add_argument("--pytorch-src",
                        help="the directory of pytorch source code git repository",
                        type=exist_dir_path,
                        required=True)
    parser.add_argument("--torchbench-src",
                        help="the directory of torchbench source code git repository",
                        type=exist_dir_path,
                        required=True)
    parser.add_argument("--start",
                        help="7-digit SHA hash of the start commit to bisect",
                        required=True)
    parser.add_argument("--end",
                        help="7-digit SHA hash of the end commit to bisect",
                        required=True)
    parser.add_argument("--threshold",
                        help="the torchbench score threshold to report a regression",
                        type=float,
                        required=True)
    parser.add_argument("--timeout",
                        type=int,
                        help="the maximum time to run the benchmark in minutes",
                        required=True)
    parser.add_argument("--output",
                        help="the output json file",
                        required=True)
    args = parser.parse_args()
    bisection = TorchBenchBisection(workdir=args.work_dir,
                                    torch_src=args.pytorch_src,
                                    bench_src=args.torchbench_src,
                                    start=args.start,
                                    end=args.end,
                                    threshold=args.threshold,
                                    timeout=args.timeout,
                                    output_json=args.output)
    assert bisection.prep(), "The working condition of bisection is not satisfied."
    print("Preparation steps ok.")
    bisection.run()
    bisection.output()
