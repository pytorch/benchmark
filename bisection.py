"""bisection.py
Runs bisection to determine PRs that cause performance regression.
Performance regression is defined by TorchBench score drop greater than the threshold.
By default, the torchvision, torchaudio, and torchtext package version will be fixed to the latest git version.

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
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from torchbenchmark.score.compute_score import compute_score
from torchbenchmark.util import gitutils
from torchbenchmark.util import torch_nightly

## Helper functions
def exist_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

TORCH_GITREPO="https://github.com/pytorch/pytorch.git"
TORCHBENCH_GITREPO="https://github.com/pytorch/benchmark.git"
TORCHBENCH_SCORE_CONFIG="torchbenchmark/score/configs/v0/config-v0.yaml"
TORCHBENCH_DEPS = {
    "torchtext": os.path.expandvars("${HOME}/text"),
    "torchvision": os.path.expandvars("${HOME}/vision"),
    "torchaudio": os.path.expandvars("${HOME}/audio"),
}

## Class definitions
class Commit:
    sha: str
    ctime: str
    score: Optional[float]
    def __init__(self, sha, ctime, score):
        self.sha = sha
        self.ctime = ctime
        self.score = score
    def __str__(self):
        return self.sha

class TorchSource:
    srcpath: str
    commit_date: datetime
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
        # Setup commit date
        last_commit_date = self.commits[-1].ctime.split()[0]
        self.commit_date = datetime.strptime(last_commit_date, "%Y-%m-%d")
        return True
    
    def get_mid_commit(self, left: Commit, right: Commit) -> Optional[Commit]:
        left_index = self.commit_dict[left.sha]
        right_index = self.commit_dict[right.sha]
        if right_index == left_index + 1:
            return None
        else:
            return self.commits[(left_index + right_index) / 2]

    def setup_build_env(self, env):
        env["USE_CUDA"] = "1"
        env["BUILD_CAFFE2_OPS"] = "0"
        env["USE_XNNPACK"] = "0"
        env["USE_MKLDNN"] = "1"
        env["USE_MKL"] = "1"
        env["USE_CUDNN"] = "1"
        env["CMAKE_PREFIX_PATH"] = env["CONDA_PREFIX"]
        return env

    # Install dependencies such as torchaudio, torchtext and torchvision
    def install_deps(self, build_env):
        # Build torchvision
        print(f"Building torchvision ...", end="", flush=True)
        command = "python setup.py install &> /dev/null"
        subprocess.check_call(command, cwd=TORCHBENCH_DEPS["torchvision"], env=build_env, shell=True)
        print("done")
        # Build torchaudio
        print(f"Building torchaudio ...", end="", flush=True)
        command = "BUILD_SOX=1 python setup.py install &> /dev/null"
        subprocess.check_call(command, cwd=TORCHBENCH_DEPS["torchaudio"], env=build_env, shell=True)
        print("done")
        # Build torchtext
        print(f"Building torchtext ...", end="", flush=True)
        command = "python setup.py clean install &> /dev/null"
        subprocess.check_call(command, cwd=TORCHBENCH_DEPS["torchtext"], env=build_env, shell=True)
        print("done")
 
    def build(self, commit: Commit):
        # checkout pytorch commit
        gitutils.checkout_git_commit(self.srcpath, commit.sha)
        # setup environment variables
        build_env = self.setup_build_env(os.environ.copy())
        # build pytorch
        print(f"Building pytorch commit {commit.sha} ...", end="", flush=True)
        command = "python tools/generate_torch_version.py --is_debug on"
        subprocess.check_call(command, cwd=self.srcpath, env=build_env, shell=True)
        command = "python setup.py install &> /dev/null"
        subprocess.check_call(command, cwd=self.srcpath, env=build_env, shell=True)
        print("done")
        self.install_deps(build_env)

    def cleanup(self, commit: Commit):
        print(f"Cleaning up packages from commit {commit.sha} ...", end="", flush=True)
        packages = ["torch", "torchtext", "torchvision", "torchaudio"]
        command = "pip uninstall -y " + " ".join(packages)
        subprocess.check_call(command, shell=True)
        print("done")

class TorchBench:
    srcpath: str # path to pytorch/benchmark source code
    branch: str
    timelimit: int # timeout limit in minutes
    bmfilter: str
    workdir: str
    torch_src: TorchSource

    def __init__(self, srcpath: str,
                 torch_src: TorchSource,
                 timelimit: int,
                 bmfilter: str,
                 workdir: str,
                 branch: str = "0.1"):
        self.srcpath = srcpath
        self.torch_src = torch_src
        self.timelimit = timelimit
        self.bmfilter = bmfilter
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
        # Checkout dependency commit
        print(f"Checking out dependency last commit date: {self.torch_src.commit_date}")
        for pkg in TORCHBENCH_DEPS:
            last_commit = gitutils.get_git_commit_on_date(TORCHBENCH_DEPS[pkg], self.torch_src.commit_date)
            if not last_commit:
                return False
            if not gitutils.checkout_git_commit(TORCHBENCH_DEPS[pkg], last_commit):
                return False
        return True
 
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
        command = str()
        if self.bmfilter:
            command = f"bash .github/scripts/run-nodocker.sh {output_dir} \"{self.bmfilter}\" &> {output_dir}/benchmark.log"
        else:
            command = f"bash .github/scripts/run-nodocker.sh {output_dir} &> {output_dir}/benchmark.log"
        try:
            subprocess.check_call(command, cwd=self.srcpath, shell=True, timeout=self.timelimit * 60)
        except subprocess.TimeoutExpired:
            print(f"Benchmark timeout for {commit.sha}. Returning zero value.")
            return output_dir
        return output_dir

    def compute(self, result_dir: str) -> float:
        filelist = [ f for f in os.listdir(result_dir) if f.endswith(".json") ]
        if len(filelist) == 0:
            print(f"Empty directory or json file in {result_dir}. Return zero score.")
            return 0.0
        # benchmark data file
        data_file = os.path.join(result_dir, filelist[0])
        if os.stat(data_file).st_size == 0:
            print(f"Empty json file {filelist[0]} in {result_dir}. Return zero score.")
            return 0.0
        # configuration
        config_file = os.path.join(self.srcpath, TORCHBENCH_SCORE_CONFIG)
        with open(config_file) as cfg_file:
            config = yaml.full_load(cfg_file)
        with open(data_file) as dfile:
            data = json.load(dfile)
        return compute_score(config, data)
    
    def get_score(self, commit: Commit) -> float:
        # Score is cached
        if commit.score is not None:
            return commit.score
        # Build pytorch and its dependencies
        self.torch_src.build(commit)
        # Run benchmark
        print(f"Running TorchBench for commit: {commit.sha} ...", end="", flush=True)
        result_dir = self.run_benchmark(commit)
        commit.score = self.compute(result_dir)
        print(f" score: {commit.score}")
        self.torch_src.cleanup(commit)
        return commit.score
        
class TorchBenchBisection:
    start: str
    end: str
    workdir: str
    threshold: int
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
                 bmfilter: str,
                 timeout: int,
                 output_json: str):
        self.workdir = workdir
        self.start = start
        self.end = end
        self.threshold = threshold
        self.bisectq = list()
        self.torch_src = TorchSource(srcpath = torch_src)
        self.bench = TorchBench(srcpath = bench_src,
                                torch_src = self.torch_src,
                                timelimit = timeout,
                                bmfilter = bmfilter,
                                workdir = self.workdir)
        self.output_json = output_json

    # Left: older commit; right: newer commit
    def regression(self, left: Commit, right: Commit) -> bool:
        assert left.score is not None
        assert right.score is not None
        return left.score - right.score >= self.threshold

    def prep(self) -> bool:
        if not self.torch_src.prep():
            return False
        if not self.torch_src.init_commits(self.start, self.end):
            return False
        if not self.bench.prep():
            return False
        left_commit = self.torch_src.commits[0]
        right_commit = self.torch_src.commits[-1]
        self.bisectq.append((left_commit, right_commit))
        return True
        
    def run(self):
        while not len(self.bisectq) == 0:
            (left, right) = self.bisectq.pop(0)
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
    parser.add_argument("--bmfilter",
                        help="the benchmark filter to run")
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
                                    bmfilter=args.bmfilter,
                                    timeout=args.timeout,
                                    output_json=args.output)
    assert bisection.prep(), "The working condition of bisection is not satisfied."
    print("Preparation steps ok. Commit list: " + " ".join([str(x) for x in bisection.torch_src.commits]))
    bisection.run()
    bisection.output()
