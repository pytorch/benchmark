"""bisection.py
Runs bisection to determine PRs that trigger performance signals.
It assumes that the pytorch, torchbench, torchtext, torchvision, and torchaudio repositories provided are all clean with the latest code.
By default, the torchaudio, torchvision and torchtext packages will be fixed to the latest commit on the same pytorch commit date.

Usage:
  python bisection.py --work-dir <WORK_DIR> \
    --torch-repos-path <PYTORCH_REPOS_PATH> \
    --torchbench-repo-path <TORCHBENCH_SRC_DIR> \
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

from userbenchmark.utils import parse_abtest_result_from_regression_dict, TorchBenchABTestResult
from regression_detector import generate_regression_dict
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

def get_updated_clean_torch_repos(pytorch_repos_path: str, torchbench_repo_path: Optional[str]=None) -> Dict[str, TorchRepo]:
    all_repos = {}
    def _gen_torch_repo(repo_name: str, repo_path: str):
        assert repo_path.exists() and repo_path.is_dir(), f"{str(repo_path)} is not an existing directory."
        main_branch = "main" if not "main_branch" in TORCHBENCH_BISECTION_TARGETS[repo_name] else \
                      TORCHBENCH_BISECTION_TARGETS[repo_name]["main_branch"]
        update_git_repo(repo_path.absolute(), main_branch)
        assert clean_git_repo(repo_path.absolute())
        cur_commit = get_current_commit(repo_path.absolute())
        return TorchRepo(name=repo_name, 
                         origin_url=TORCHBENCH_BISECTION_TARGETS[repo_name]["url"],
                         main_branch=main_branch,
                         src_path=repo_path,
                         cur_commit=cur_commit,
                         build_command=TORCHBENCH_BISECTION_TARGETS[repo_name]["build_command"])
    for repo_name in TORCHBENCH_BISECTION_TARGETS.keys():
        repo_subdir_name = TORCHBENCH_BISECTION_TARGETS[repo_name]["name"]
        repo_path = Path(pytorch_repos_path).joinpath(repo_subdir_name) if not (torchbench_repo_path and repo_name == "torchbench") \
                        else Path(torchbench_repo_path)
        all_repos[repo_name] = _gen_torch_repo(repo_name, repo_path)
    return all_repos

class Commit:
    sha: str
    ctime: str
    digest: Optional[Dict[str, Any]]
    def __init__(self, sha, ctime):
        self.sha = sha
        self.ctime = ctime
        self.digest = None
    def __str__(self):
        return self.sha

class BisectionTargetRepo:
    repo: TorchRepo
    start: str
    end: str
    bisection_env: os._Environ
    commits: List[Commit]
    # Map from commit SHA to its index in commits
    commit_dict: Dict[str, int]
    def __init__(self, repo: TorchRepo, start: str, end: str, bisection_env: os._Environ):
        self.repo = repo
        self.start = start
        self.end = end
        self.bisection_env = bisection_env
        self.commits = []
        self.commit_dict = dict()

    def prep(self) -> bool:
        commits = get_git_commits(self.repo.src_path, self.start, self.end)
        if not commits or len(commits) < 2:
            print(f"Failed to retrieve commits from {self.start} to {self.end} in {self.repo.src_path}.")
            return False
        for count, commit in enumerate(commits):
            ctime = get_git_commit_date(self.repo.src_path, commit)
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
        checkout_git_commit(self.srcpath, commit.sha)
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

class TorchBenchRepo:
    repo: TorchRepo
    timelimit: int # timeout limit in minutes
    workdir: str
    first_time: bool
    target_repo: BisectionTargetRepo
    bench_env: os._Environ

    def __init__(self, srcpath: str,
                 torch_src: BisectionTargetRepo,
                 workdir: str):
        self.srcpath = srcpath
        self.torch_src = torch_src
        self.workdir = workdir
        self.first_time = True

    def prep(self, bench_env) -> bool:
        self.bench_env = bench_env
        # get the name of current branch
        self.branch = get_current_branch(self.srcpath)
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
    workdir: Path
    bisection_env: os._Environ
    torch_repos: List[TorchRepo]
    target_repo: BisectionTargetRepo
    torchbench: TorchBenchRepo
    bisect_config: TorchBenchABTestResult
    output_json: str
    debug: bool
    # left commit, right commit, TorchBenchABTestResult to test
    bisectq: List[Tuple[Commit, Commit, TorchBenchABTestResult]]
    result: List[Tuple[Commit, Commit]]


    def __init__(self,
                 workdir: str,
                 torch_repos: List[TorchRepo],
                 target_repo: TorchRepo,
                 start: str,
                 end: str,
                 bisect_config: TorchBenchABTestResult,
                 output_json: str,
                 debug: bool = False):
        self.workdir = Path(workdir)
        self.torch_repos = torch_repos
        self.target_repo = BisectionTargetRepo(repo=target_repo, start=start, end=end)
        self.torchbench = TorchBenchRepo(repo=torch_repos["torchbench"],
                                         workdir=self.workdir)
        self.bisect_config = bisect_config
        self.bisectq = list()
        self.result = list()
        self.output_json = output_json
        self.debug = debug

    def prep(self) -> bool:
        base_build_env = prepare_cuda_env(cuda_version=DEFAULT_CUDA_VERSION)
        self.bisection_env = setup_bisection_build_env(base_build_env)
        if not self.target_repo.prep(self.bisection_env):
            return False
        if not self.torchbench.prep(self.bisection_env):
            return False
        left_commit = self.target_repo_src.commits[0]
        right_commit = self.target_repo_src.commits[-1]
        self.bisectq.append((left_commit, right_commit, self.bisect_config))
        return True

    # Left: older commit, right: newer commit, target: TorchBenchABTestResult
    # Return: List of [left, right, TorchBenchABTestResult] that satisfy the regression rule
    def regression_detection(self, left: Commit, right: Commit) -> TorchBenchABTestResult:
        # If uncalculated, commit.digest will be None
        assert left.digest, "Commit {left.sha} must have a digest"
        assert right.digest, "Commit {right.sha} must have a digest"
        regression_dict = generate_regression_dict(left.digest, right.digest)
        abtest_result = parse_abtest_result_from_regression_dict(regression_dict)
        return abtest_result
        
    def run(self):
        while len(self.bisectq):
            (left, right, abtest_result) = self.bisectq.pop(0)
            self.torchbench.run_commit(left, abtest_result, self.debug)
            self.torchbench.run_commit(right, abtest_result, self.debug)
            updated_abtest_result = self.regression_detection(left, right)
            if len(updated_abtest_result.details):
                mid = self.torch_src.get_mid_commit(left, right)
                if mid == None:
                    self.result.append((left, right))
                else:
                    self.bisectq.append((left, mid, updated_abtest_result))
                    self.bisectq.append((mid, right, updated_abtest_result))
 
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
    parser.add_argument("--torch-repos-path",
                        help="the directory of pytorch/* source code repositories",
                        type=exist_dir_path)
    parser.add_argument("--torchbench-repo-path",
                        default=None,
                        help="the directory of torchbench source code git repository, if None, use `args.torch_repo_path/benchmark`.",
                        type=exist_dir_path)
    parser.add_argument("--config",
                        help="the regression dict output of regression_detector.py in YAML")
    parser.add_argument("--output",
                        help="the output json file")
    # by default, debug mode is disabled
    parser.add_argument("--debug",
                        help="run in debug mode, if the result json exists, use it directly",
                        action='store_true')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        bisect_config = parse_abtest_result_from_regression_dict(yaml.full_load(f))

    # sanity checks
    assert bisect_config.control_env["git_commit_hash"], "Invalid bisection config, must specify control group commit hash."
    assert bisect_config.treatment_env["git_commit_hash"], "Invalid bisection config, must specify treatment group commit hash."
    assert bisect_config.bisection in TORCHBENCH_BISECTION_TARGETS.keys(), f"Invalid bisection config, " \
                                                                            f"get bisection target repo {bisect_config.bisection}, " \
                                                                            f"available target repos: {TORCHBENCH_BISECTION_TARGETS.keys()}"
    assert bisect_config.bisection_mode == "bisect", "Abtest mode is not supported yet."

    # load, update, and clean the repo directories
    torch_repos: Dict[str, TorchRepo] = get_updated_clean_torch_repos(args.torch_repos_path, args.torchbench_repo_path)

    bisection = TorchBenchBisection(workdir=args.work_dir,
                                    torch_repos=torch_repos,
                                    target_repo=torch_repos[bisect_config.bisection],
                                    start=bisect_config.control_env["git_commit_hash"],
                                    end=bisect_config.treatment_env["git_commit_hash"],
                                    bisect_config=bisect_config,
                                    output_json=args.output,
                                    debug=args.debug)
    assert bisection.prep(), "The working condition of bisection is not satisfied."
    print("Preparation steps ok. Commit to bisect: " + " ".join([str(x) for x in bisection.target_repo.commits]))
    # bisection.run()
    # bisection.output()
