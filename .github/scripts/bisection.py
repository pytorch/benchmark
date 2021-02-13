"""bisection.py
Runs bisection to determine PRs that cause performance regression.
Performance regression is defined by TorchBench score drop greater than the threshold.
By default, the torchvision and torchtext package version will be fixed to the latest nightly.

Usage:
  python bisection.py --pytorch-src <PYTORCH_SRC_DIR> \
    --torchbench-src <TORCHBENCH_SRC_DIR> \
    --start <SHA> --end <SHA> --threshold <SCORE_THRESHOLD> \
    --timeout <TIMEOUT_IN_MINS> --env-name <CONDA_ENV_NAME>
"""

import os
import json
import argparse
import typing
import re
import subprocess
from . import gitutils

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

## Class definitions
class Commit:
    sha: str
    score: Option[float]

class TorchSource:
    srcpath: str
    commits: List[Commit]
    # Map from commit SHA to index in commits
    commit_dict: Dict[str, int]
    def __init__(self, srcpath: str):
        self.srcpath = srcpath

    def prep() -> bool:
        # Verify the code in srcpath is pytorch/pytorch
        return

    # Get all commits between start and end, save them in commits
    def init_commits(self, start: str, end: str):
        pass
    
    def get_mid_commit(self, left: Commit, right: Commit) -> Option[Commit]:
        left_index = commit_dict[left.sha]
        right_index = commit_dict[right.sha]
        if right_index == left_index + 1:
            return None
        else:
            return commits[(left_index + right_index) / 2]

    # TODO: optimize building speed with ccache
    # TODO: check conda build environment
    def checkout_build(self, commit: Commit):
        pass

class TorchBench:
    srcpath: str # path to pytorch/benchmark source code
    branch: str
    timeout_limit: int # timeout limit in minutes
    torch_src: TorchSource

    def __init__(self, srcpath: str,
                 torch_src: TorchSource,
                 branch: str = "0.1"):
        self.srcpath = srcpath
        self.branch = branch
        self.torch_src = torch_src

    def prep(self) -> bool:
        # Verify the code in srcpath is pytorch/benchmark
        # Update the code
        # Checkout branch and test success
        pass
        
    def build_benchmark(self):
        pass

    def run_benchmark(self):
        pass

    def get_score(self, commit: Commit) -> float:
        if commit.score is not None:
            return commit.score
        # compile
        
class TorchBenchBisection:
    start: str
    end: str
    workdir: str
    threshold: int
    output: str
    bisectq: List[Tuple[str, str]]
    result: List[Tuple[Commit, Commit]]
    torch_src: TorchSource
    bench: TorchBench
    conda_env: str
    output_json: str

    def __init__(self,
                 workdir: str,
                 torch_src: str,
                 bench_src: str,
                 start: str,
                 end: str,
                 threshold: int,
                 timeout: int,
                 output_json: str,
                 conda_env: str):
        self.start = start
        self.end = end
        self.threshold = threshold
        self.timeout = timeout
        self.output = output
        self.bisectq = list()
        self.torch_src = TorchSource(srcpath = torch_src)
        self.bench = TorchBench(srcpath = bench_src,
                                torch_src = torch_src)
        self.output_json = output_json
        self.conda_env = conda_env

    def regression(self, left: Commit, right: Commit) -> bool:
        assert left.score is not None
        assert right.score is not None
        return abs(left.score - right.score) >= threshold

    def prep(self) -> bool:
        if not torch_src.prep() or not bench.prep():
            return False
        if not torch_src.init_commits(start, end):
            return False
        # Activate the conda environment
        if not subprocess.call(". activate " + conda_env) == 0:
            return False
        return True
        
    def run(self):
        while not commit_ranges.empty():
            (left, right) = commit_ranges[0]
            left.score = tbench.get_score(left)
            right.score = tbench.get_score(right)
            if self.regression(left, right):
                mid = torch_src.get_mid_commit(left, right)
                if mid == None:
                    result.append((left, right))
                else:
                    mid.score = tbench.get_score(mid_commit)
                    if self.regression(left, mid):
                        commit_ranges.append(left, mid)
                    if self.regression(mid, right):
                        commit_ranges.append(right, mid)
            
    def dump_result(self):
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
    parser.add_argument("--conda-env",
                        help="name of the conda environment that contains build dependencies",
                        required=True)
    args = parser.parse_args()
    bisection = TorchBenchBisection(workdir=args.work_dir,
                                    pytorch_src=args.pytorch_src,
                                    bench_src=args.torchbench_src,
                                    start=args.start,
                                    end=args.end,
                                    threshold=args.threshold,
                                    timeout=args.timeout,
                                    output_json=args.output,
                                    conda_env=args.conda_env)
    assert bisection.prep(), "The working condition of bisection is not satisfied."
    # bisection.run()
    # bisection.dump_result()
