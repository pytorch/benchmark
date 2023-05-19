"""bisection.py
Runs bisection to determine PRs that trigger performance signals.
It assumes that the pytorch, torchbench, torchtext, torchvision, and torchaudio repositories provided are all clean with the latest code.
By default, the torchaudio, torchvision and torchtext packages will be fixed to the latest commit on the same pytorch commit date.

Usage:
  python bisection.py --work-dir <WORK-DIR> \
    --pytorch-src <PYTORCH_SRC_DIR> \
    --torchbench-src <TORCHBENCH_SRC_DIR> \
    --config <BISECT_CONFIG> --output <OUTPUT_FILE_PATH>
"""

import os
import sys
import json
import shutil
import yaml
import argparse
from pathlib import Path
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

from utils.gitutils import *
from utils.build_utils import (
    setup_bisection_build_env,
    build_repo,
    TorchRepo,
)
from utils.cuda_utils import prepare_cuda_env, DEFAULT_CUDA_VERSION

TORCHBENCH_BISECTION_TARGETS = {
    "pytorch": {
        "name": "pytorch",
        "url": "https://github.com/pytorch/pytorch.git",
        "build_command": [sys.executable, "setup.py", "install"],
    },
    "torchdata": {
        "name": "data",
        "url": "https://github.com/pytorch/data.git",
        "build_command": [sys.executable, "setup.py", "install"],
    },
    "torchvision": {
        "name": "vision",
        "url": "https://github.com/pytorch/vision.git",
        "build_command": [sys.executable, "setup.py", "install"],
    },
    "torchtext": {
        "name": "text",
        "url": "https://github.com/pytorch/text.git",
        "build_command": [sys.executable, "setup.py", "clean", "install"],
    },
    "torchaudio": {
        "name": "audio",
        "url": "https://github.com/pytorch/audio.git",
        "build_command": [sys.executable, "setup.py", "clean", "develop"],
    },
    "torchbench": {
        "name": "benchmark",
        "url": "https://github.com/pytorch/benchmark.git",
        "build_command": [sys.executable, "install.py"],
    },
}

def exist_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_updated_torch_repos(pytorch_repos_path: str, torchbench_repo_path: str) -> Dict[str, TorchRepo]:
    all_repos = {}
    for repo_name in TORCHBENCH_BISECTION_TARGETS.keys():
        repo_path = Path(pytorch_repos_path).joinpath("repo_name")
        assert repo_path.exists() and repo_path.is_dir(), f"{str(repo_path)} is not an existing directory."
        main_branch = "main" if not "main_branch" in TORCHBENCH_BISECTION_TARGETS[repo_name] else \
                      TORCHBENCH_BISECTION_TARGETS[repo_name]["main_branch"]
        update_git_repo(repo_path.absolute(), main_branch)
        cur_commit = get_current_commit(repo_path.absolute())
        all_repos[repo_name] = TorchRepo(name=repo_name, src_path=repo_path, cur_commit=cur_commit, \
                                         main_branch=main_branch, build_command=TORCHBENCH_BISECTION_TARGETS[repo_name]["build_command"])
    return all_repos


class Commit:
    sha: str
    ctime: str
    digest: Optional[Any]
    def __init__(self, sha, ctime):
        self.sha = sha
        self.ctime = ctime
        self.digest = None
    def __str__(self):
        return self.sha

class TorchSource:
    srcpath: str
    commits: List[Commit]
    build_env: os._Environ
    # Map from commit SHA to index in commits
    commit_dict: Dict[str, int]
    def __init__(self, srcpath: str):
        self.srcpath = srcpath
        self.commits = []
        self.commit_dict = dict()

    def prep(self, build_env: os._Environ) -> bool:
        repo_origin_url = get_git_origin(self.srcpath)
        if not repo_origin_url == TORCH_GITREPO:
            print(f"WARNING: Unmatched repo origin url: {repo_origin_url} with standard {TORCH_GITREPO}")
        self.update_repos()
        # Clean up the existing packages
        self.cleanup()
        self.build_env = build_env
        return True

    # Update pytorch, torchtext, torchvision, and torchaudio repo
    def update_repos(self):
        repos = [(self.srcpath, "main")]
        repos.extend(TORCHBENCH_DEPS.values())
        for (repo, branch) in repos:
            gitutils.clean_git_repo(repo)
            assert gitutils.update_git_repo(repo, branch), f"Failed to update {branch} branch of repository {repo}."

    # Get all commits between start and end, save them in self.commits
    def init_commits(self, start: str, end: str, abtest: bool) -> bool:
        if not abtest:
            commits = gitutils.get_git_commits(self.srcpath, start, end)
        else:
            commits = [start, end]
        if not commits or len(commits) < 2:
            print(f"Failed to retrieve commits from {start} to {end} in {self.srcpath}.")
            return False
        for count, commit in enumerate(commits):
            ctime = gitutils.get_git_commit_date(self.srcpath, commit)
            self.commits.append(Commit(sha=commit, ctime=ctime))
            self.commit_dict[commit] = count
        return True
    
    def get_mid_commit(self, left: Commit, right: Commit) -> Optional[Commit]:
        left_index = self.commit_dict[left.sha]
        right_index = self.commit_dict[right.sha]
        if right_index == left_index + 1:
            return None
        else:
            return self.commits[int((left_index + right_index) / 2)]

    # Checkout the last commit of dependencies on date
    def checkout_deps(self, cdate: datetime):
        for pkg in TORCHBENCH_DEPS:
            pkg_path, branch = TORCHBENCH_DEPS[pkg]
            gitutils.checkout_git_branch(pkg_path, branch)
            dep_commit = gitutils.get_git_commit_on_date(pkg_path, cdate)
            print(f"Checking out {pkg} commit {dep_commit} ...", end="", flush=True)
            assert dep_commit, "Failed to find the commit on {cdate} of {pkg}"
            assert gitutils.checkout_git_commit(pkg_path, dep_commit), f"Failed to checkout commit {dep_commit} of {pkg}"
            print("done.")
    
    # Install dependencies such as torchtext and torchvision
    def build_install_deps(self, build_env):
        build_repo(self.repos["torchdata"], build_env)
        build_repo(self.repos["torchvision"], build_env)
        build_repo(self.repos["torchtext"], build_env)
        build_repo(self.repos["torchaudio"], build_env)

    def build(self, commit: Commit):
        # checkout pytorch commit
        print(f"Checking out pytorch commit {commit.sha} ...", end="", flush=True)
        gitutils.checkout_git_commit(self.srcpath, commit.sha)
        print("done.")
        # checkout pytorch deps commit
        ctime = datetime.strptime(commit.ctime.split(" ")[0], "%Y-%m-%d")
        self.checkout_deps(ctime)
        # setup environment variables
        build_env = setup_bisection_build_env(self.build_env)
        # build pytorch
        print(f"Building pytorch commit {commit.sha} ...", end="", flush=True)
        # Check if version.py exists, if it does, remove it.
        # This is to force pytorch update the version.py file upon incremental compilation
        version_py_path = os.path.join(self.srcpath, "torch/version.py")
        if os.path.exists(version_py_path):
            os.remove(version_py_path)
        try:
            command = "python setup.py install"
            subprocess.check_call(command, cwd=self.srcpath, env=build_env, shell=True)
            command_testbuild = "python -c 'import torch'"
            subprocess.check_call(command_testbuild, cwd=os.environ["HOME"], env=build_env, shell=True)
        except subprocess.CalledProcessError:
            # Remove the build directory, then try build it again
            build_path = os.path.join(self.srcpath, "build")
            if os.path.exists(build_path):
                shutil.rmtree(build_path)
            subprocess.check_call(command, cwd=self.srcpath, env=build_env, shell=True)
        print("done")
        self.build_install_deps(build_env)

    def cleanup(self):
        packages = ["torch"] + list(TORCHBENCH_DEPS.keys())
        CLEANUP_ROUND = 5
        # Clean up multiple times to make sure the packages are all uninstalled
        for _ in range(CLEANUP_ROUND):
            command = "pip uninstall -y " + " ".join(packages) + " || true"
            subprocess.check_call(command, shell=True)
        print("done")

class TorchBench:
    srcpath: str # path to pytorch/benchmark source code
    branch: str
    timelimit: int # timeout limit in minutes
    workdir: str
    models: List[str]
    first_time: bool
    torch_src: TorchSource
    bench_env: os._Environ

    def __init__(self, srcpath: str,
                 torch_src: TorchSource,
                 timelimit: int,
                 workdir: str):
        self.srcpath = srcpath
        self.torch_src = torch_src
        self.timelimit = timelimit
        self.workdir = workdir
        self.first_time = True
        self.models = list()

    def prep(self, bench_env) -> bool:
        self.bench_env = bench_env
        # Verify the code in srcpath is pytorch/benchmark
        repo_origin_url = gitutils.get_git_origin(self.srcpath)
        if not repo_origin_url == TORCHBENCH_GITREPO:
            print(f"WARNING: Unmatched repo origin url: {repo_origin_url} with standard {TORCHBENCH_GITREPO}")
        # get the name of current branch
        self.branch = gitutils.get_current_branch(self.srcpath)
        # get list of models
        self.models = [ model for model in os.listdir(os.path.join(self.srcpath, "torchbenchmark", "models"))
                        if os.path.isdir(os.path.join(self.srcpath, "torchbenchmark", "models", model)) ]
        return True

    def _install_benchmark(self):
        "Install and build TorchBench dependencies"
        command = ["python", "install.py"]
        subprocess.check_call(command, cwd=self.srcpath, env=self.bench_env, shell=False)

    def run_benchmark(self, commit: Commit, targets: List[str]) -> str:
        # Return the result json file path
        output_dir = os.path.join(self.workdir, commit.sha)
        # If the directory already exists, clear its contents
        if os.path.exists(output_dir):
            assert os.path.isdir(output_dir), "Must specify output directory: {output_dir}"
            filelist = [ f for f in os.listdir(output_dir) ]
            for f in filelist:
                os.remove(os.path.join(output_dir, f))
        else:
            os.mkdir(output_dir)
        bmfilter = targets_to_bmfilter(targets, self.models)
        # If the first time to run benchmark, install the dependencies first
        if self.first_time:
            self._install_benchmark()
            self.first_time = False
        print(f"Running TorchBench for commit: {commit.sha}, filter {bmfilter} ...", end="", flush=True)
        command = f"""bash .github/scripts/run.sh "{output_dir}" "{bmfilter}" 2>&1 | tee {output_dir}/benchmark.log"""
        try:
            subprocess.check_call(command, cwd=self.srcpath, env=self.bench_env, shell=True, timeout=self.timelimit * 60)
        except subprocess.TimeoutExpired:
            print(f"Benchmark timeout for {commit.sha}. Result will be None.")
            return output_dir
        print("done.")
        return output_dir

    def gen_digest(self, result_dir: str, targets: List[str]) -> Dict[str, float]:
        filelist = [ f for f in os.listdir(result_dir) if f.endswith(".json") ]
        out = dict()
        if not len(filelist):
            print(f"Empty directory or json file in {result_dir}. Return empty digest.")
            return out
        # Use the first json as the benchmark data file
        data_file = os.path.join(result_dir, filelist[0])
        if not os.stat(data_file).st_size:
            print(f"Empty json file {filelist[0]} in {result_dir}. Return empty digest.")
            return out
        with open(data_file, "r") as df:
            data = json.load(df)
        # Fill in targets if it is None
        if targets == None:
            targets = list()
            for each in data["benchmarks"]:
                targets.append(each["name"])
        old_targets = targets.copy()
        for t in filter(lambda x: x in self.models, old_targets):
            targets.remove(t)
            names =  filter(lambda y: t in y, map(lambda x: x["name"], data["benchmarks"]))
            targets.extend(list(names))
        for each in data["benchmarks"]:
            if each["name"] in targets:
                out[each["name"]] = each["stats"]["mean"]
        # Make sure all target tests are available
        for target in targets:
            assert out[target], f"Don't find benchmark result of {target} in {filelist[0]}."
        return out

    def get_digest(self, commit: Commit, targets: List[str], debug: bool) -> Dict[str, float]:
        # digest is cached
        if commit.digest is not None:
            return commit.digest
        # if debug mode, skip the build and benchmark run
        if debug:
            result_dir = os.path.join(self.workdir, commit.sha)
            if os.path.isdir(result_dir):
                filelist = [ f for f in os.listdir(result_dir) if f.endswith(".json") ]
                if len(filelist):
                    data_file = os.path.join(result_dir, filelist[0])
                    if os.stat(data_file).st_size:
                        commit.digest = self.gen_digest(result_dir, targets)
                        return commit.digest
        # Build pytorch and its dependencies
        self.torch_src.build(commit)
        # Run benchmark
        result_dir = self.run_benchmark(commit, targets)
        commit.digest = self.gen_digest(result_dir, targets)
        print(f"Cleaning up packages from commit {commit.sha} ...", end="", flush=True)
        self.torch_src.cleanup()
        return commit.digest
        
class TorchBenchBisection:
    workdir: str
    start: str
    end: str
    threshold: float
    direction: str
    targets: List[str]
    # left commit, right commit, targets to test
    bisectq: List[Tuple[Commit, Commit, List[str]]]
    result: List[Tuple[Commit, Commit]]
    torch_src: TorchSource
    bench: TorchBench
    output_json: str
    debug: bool
    abtest: bool

    def __init__(self,
                 workdir: str,
                 torch_src: str,
                 bench_src: str,
                 start: str,
                 end: str,
                 threshold: float,
                 direction: str,
                 timeout: int,
                 targets: List[str],
                 output_json: str,
                 debug: bool = False):
        self.workdir = workdir
        self.start = start
        self.end = end
        self.threshold = threshold
        self.direction = direction
        self.targets = targets
        self.bisectq = list()
        self.result = list()
        self.torch_src = TorchSource(srcpath = torch_src)
        self.bench = TorchBench(srcpath = bench_src,
                                torch_src = self.torch_src,
                                timelimit = timeout,
                                workdir = self.workdir)
        self.output_json = output_json
        self.debug = debug
        # Special treatment for abtest
        self.abtest = False
        if self.threshold == 100.0 and self.direction == "decrease":
            self.abtest = True

    # Left: older commit; right: newer commit
    # Return: List of targets that satisfy the regression rule: <threshold, direction>
    def regression(self, left: Commit, right: Commit, targets: List[str]) -> List[str]:
        # If uncalculated, commit.digest will be None
        assert left.digest, "Commit {left.sha} must have a digest"
        assert right.digest, "Commit {right.sha} must have a digest"
        out = []
        for target in targets:
            # digest could be empty if benchmark timeout
            left_mean = left.digest[target] if len(left.digest) else 0
            right_mean = right.digest[target] if len(right.digest) else 0
            # If either left or right timeout, diff is 100. Otherwise use the min mean value to calculate diff.
            diff = abs(left_mean - right_mean) / min(left_mean, right_mean) * 100 if min(left_mean, right_mean) else 100
            # If both timeout, diff is zero percent
            diff = 0 if not max(left_mean, right_mean) else diff
            print(f"Target {target}: left commit {left.sha} mean {left_mean} vs. right commit {right.sha} mean {right_mean}. Diff: {diff}.")
            if diff >= self.threshold:
                if self.direction == "increase" and left_mean < right_mean:
                    # Time increase == performance regression
                    out.append(target)
                elif self.direction == "decrease" and left_mean > right_mean:
                    # Time decrease == performance optimization
                    out.append(target)
                elif self.direction == "both":
                    out.append(target)
        return out

    def prep(self) -> bool:
        base_build_env = prepare_cuda_env(cuda_version=DEFAULT_CUDA_VERSION)
        if not self.torch_src.prep(base_build_env):
            return False
        if not self.torch_src.init_commits(self.start, self.end, self.abtest):
            return False
        if not self.bench.prep(base_build_env):
            return False
        left_commit = self.torch_src.commits[0]
        right_commit = self.torch_src.commits[-1]
        self.bisectq.append((left_commit, right_commit, self.targets))
        return True
        
    def run(self):
        while len(self.bisectq):
            (left, right, targets) = self.bisectq.pop(0)
            self.bench.get_digest(left, targets, self.debug)
            self.bench.get_digest(right, targets, self.debug)
            if targets == None and len(left.digest):
                targets = left.digest.keys()
            if targets == None and len(right.digest):
                targets = right.digest.keys()
            updated_targets = self.regression(left, right, targets)
            if len(updated_targets):
                mid = self.torch_src.get_mid_commit(left, right)
                if mid == None:
                    self.result.append((left, right))
                else:
                    self.bisectq.append((left, mid, updated_targets))
                    self.bisectq.append((mid, right, updated_targets))
 
    def output(self):
        json_obj = dict()
        json_obj["start"] = self.start
        json_obj["end"] = self.end
        json_obj["threshold"] = self.threshold
        json_obj["timeout"] = self.bench.timelimit
        json_obj["torchbench_branch"] = self.bench.branch
        json_obj["result"] = []
        for res in self.result:
            r = dict()
            r["commit1"] = res[0].sha
            r["commit1_time"] = res[0].ctime
            r["commit1_digest"] = res[0].digest if len(res[0].digest) else "timeout"
            r["commit2"] = res[1].sha
            r["commit2_time"] = res[1].ctime
            r["commit2_digest"] = res[1].digest if len(res[1].digest) else "timeout"
            json_obj["result"].append(r)
        with open(self.output_json, 'w') as outfile:
            json.dump(json_obj, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",
                        help="bisection working directory for logs and results",
                        type=exist_dir_path)
    parser.add_argument("--pytorch-repos-path",
                        help="the directory of pytorch/* source code repositories",
                        type=exist_dir_path)
    parser.add_argument("--torchbench-repo-path",
                        help="the directory of torchbench source code git repository",
                        type=exist_dir_path)
    parser.add_argument("--target-repo",
                        help="the target repo for bisection, default to pytorch. It should match the hash in the bisection config.",
                        default="pytorch",
                        choices=TORCHBENCH_BISECTION_TARGETS.keys())
    parser.add_argument("--config",
                        help="the bisection configuration in YAML format")
    parser.add_argument("--output",
                        help="the output json file")
    # by default, debug mode is disabled
    parser.add_argument("--debug",
                        help="run in debug mode, if the result json exists, use it directly",
                        action='store_true')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        bisect_config = yaml.full_load(f)

    # sanity checks
    valid_directions = ["increase", "decrease", "both"]
    assert("start" in bisect_config), "Illegal bisection config, must specify start commit SHA."
    assert("end" in bisect_config), "Illegal bisection config, must specify end commit SHA."
    assert("threshold" in bisect_config), "Illegal bisection config, must specify threshold."
    assert("direction" in bisect_config), "Illegal bisection config, must specify direction."
    assert(bisect_config["direction"] in valid_directions), "We only support increase, decrease, or both directions"
    assert("timeout" in bisect_config), "Illegal bisection config, must specify timeout."
    targets = None
    if "tests" in bisect_config:
        targets = bisect_config["tests"]
    # read the repo directory, if necessary, update it
    torch_repos: Dict[str, TorchRepo] = get_updated_torch_repos(args.pytorch_repos_path, args.torchbench_repo_path)

    bisection = TorchBenchBisection(workdir=args.work_dir,
                                    torch_repos=torch_repos,
                                    target_repo=args.target_repo,
                                    start=bisect_config["start"],
                                    end=bisect_config["end"],
                                    threshold=bisect_config["threshold"],
                                    direction=bisect_config["direction"],
                                    timeout=bisect_config["timeout"],
                                    targets=targets,
                                    output_json=args.output,
                                    debug=args.debug)
    assert bisection.prep(), "The working condition of bisection is not satisfied."
    print("Preparation steps ok. Commit to bisect: " + " ".join([str(x) for x in bisection.torch_src.commits]))
    bisection.run()
    bisection.output()
