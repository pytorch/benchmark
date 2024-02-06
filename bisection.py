"""bisection.py
Runs bisection to determine PRs that trigger performance signals.
It assumes that the pytorch, torchbench, torchvision, and torchaudio repositories provided are all clean with the latest code.
By default, the torchaudio and torchvision packages will be fixed to the latest commit on the same pytorch commit date.

Usage:
  python bisection.py --work-dir <WORK_DIR> \
    --torch-repos-path <PYTORCH_REPOS_PATH> \
    --torchbench-repo-path <TORCHBENCH_SRC_DIR> \
    --config <BISECT_CONFIG> --output <OUTPUT_FILE_PATH>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from userbenchmark.utils import (
    parse_abtest_result_from_regression_file_for_bisect,
    TorchBenchABTestResult,
)

TORCHBENCH_BISECTION_TARGETS = {
    "pytorch": {
        "name": "pytorch",
        "url": "https://github.com/pytorch/pytorch.git",
        "build_command": [sys.executable, "setup.py", "install"],
    },
    "torchvision": {
        "name": "vision",
        "url": "https://github.com/pytorch/vision.git",
        "build_command": [sys.executable, "setup.py", "install"],
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

try:
    # OSS utils
    from regression_detector import generate_regression_result
    from utils import gitutils
    from utils.build_utils import (
        build_repo,
        cleanup_torch_packages,
        setup_bisection_build_env,
        TorchRepo,
    )
    from utils.cuda_utils import DEFAULT_CUDA_VERSION, prepare_cuda_env
    from utils.github import process_bisection_into_gh_issue
    IS_FBCODE = False
except (ImportError, ModuleNotFoundError):
    # Meta-Internal imports
    from .regression_detector import generate_regression_result
    from .utils.build_utils import setup_bisection_build_env, TorchRepo
    from .utils.cuda_utils import DEFAULT_CUDA_VERSION, prepare_cuda_env
    from .utils.fb import hgutils
    from .utils.fb.build_utils import (
        build_fbcode_repo,
        FBCODE_BUILD_COMMAND,
        FBCODE_URL,
        get_fbcode_build_artifact_dir,
        setup_fbcode_repo,
    )

    IS_FBCODE = True
    TORCHBENCH_BISECTION_TARGETS["fbcode"] = {
        "name": "fbcode",
        "url": FBCODE_URL,
        "build_command": FBCODE_BUILD_COMMAND,
    }

SKIP_INSTALL_TORCHBENCH = False


def exist_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def exist_file_path(string):
    if not os.path.exists(string):
        raise FileNotFoundError(string)
    elif os.path.isdir(string):
        return IsADirectoryError(string)
    else:
        return string


def get_latest_non_empty_file(directory: str, cond: Callable) -> Optional[str]:
    if os.path.isdir(directory):
        filelist = [os.path.join(directory, f) for f in os.listdir(directory)]
        non_empty_filelist = [f for f in filelist if os.path.getsize(f) and cond(f)]
        if len(non_empty_filelist):
            return max(non_empty_filelist, key=os.path.getctime)
    return None


def get_updated_clean_torch_repos(
    pytorch_repos_path: str,
    torchbench_repo_path: Optional[str] = None,
    skip_update_repos: Optional[List[str]] = None,
) -> Dict[str, TorchRepo]:
    if IS_FBCODE:
        return setup_fbcode_repo(pytorch_repos_path)
    all_repos = {}

    def _gen_torch_repo(repo_name: str, repo_path: str):
        assert (
            repo_path.exists() and repo_path.is_dir()
        ), f"{str(repo_path)} is not an existing directory."
        main_branch = (
            "main"
            if not "main_branch" in TORCHBENCH_BISECTION_TARGETS[repo_name]
            else TORCHBENCH_BISECTION_TARGETS[repo_name]["main_branch"]
        )
        if not skip_update_repos or not repo_name in skip_update_repos:
            gitutils.cleanup_local_changes(repo_path.absolute())
            assert gitutils.update_git_repo(repo_path.absolute(), main_branch)
            assert gitutils.clean_git_repo(repo_path.absolute())
        cur_commit = gitutils.get_current_commit(repo_path.absolute())
        return TorchRepo(
            name=repo_name,
            origin_url=TORCHBENCH_BISECTION_TARGETS[repo_name]["url"],
            main_branch=main_branch,
            src_path=repo_path,
            cur_commit=cur_commit,
            build_command=TORCHBENCH_BISECTION_TARGETS[repo_name]["build_command"],
        )

    for repo_name in TORCHBENCH_BISECTION_TARGETS.keys():
        repo_subdir_name = TORCHBENCH_BISECTION_TARGETS[repo_name]["name"]
        repo_path = (
            Path(pytorch_repos_path).joinpath(repo_subdir_name)
            if not (torchbench_repo_path and repo_name == "torchbench")
            else Path(torchbench_repo_path)
        )
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
    # Start and end git hash
    start: str
    end: str
    # Start and end version
    start_version: str
    end_version: str
    non_target_repos: List[TorchRepo]
    # generated in prep()
    bisection_env: os._Environ
    commits: List[Commit]
    # Map from commit SHA to its index in commits
    commit_dict: Dict[str, int]
    def __init__(self,
                 repo: TorchRepo,
                 start: str,
                 end: str,
                 start_version: str,
                 end_version: str,
                 non_target_repos: List[TorchRepo]):
        self.repo = repo
        self.start = start
        self.end = end
        self.start_version = start_version
        self.end_version = end_version
        self.non_target_repos = non_target_repos
        self.commits = []
        self.commit_dict = dict()

    # Checkout the last commit of non-target repos on date
    def _checkout_non_target_repos(self, cdate: datetime):
        for repo in self.non_target_repos:
            gitutils.checkout_git_branch(repo.src_path.absolute(), repo.main_branch)
            dep_commit = gitutils.get_git_commit_on_date(
                repo.src_path.absolute(), cdate
            )
            assert dep_commit, f"Failed to find the commit on {cdate} of {repo.name}"
            print(
                f"Checking out {repo.name} commit {dep_commit} ...", end="", flush=True
            )
            assert gitutils.checkout_git_commit(
                repo.src_path.absolute(), dep_commit
            ), f"Failed to checkout commit {dep_commit} of {repo.name}"
            print("done.")

    def prep(self, interactive: bool = False) -> bool:
        if not IS_FBCODE:
            if not interactive:
                base_build_env = prepare_cuda_env(cuda_version=DEFAULT_CUDA_VERSION)
                self.bisection_env = setup_bisection_build_env(base_build_env)
            else:
                self.bisection_env = os.environ.copy()
            commits = gitutils.get_git_commits(self.repo.src_path, self.start, self.end)
        else:
            self.bisection_env = os.environ.copy()
            commits = hgutils.get_hg_commits_between(
                self.repo.src_path, self.start, self.end
            )
        if not commits or len(commits) < 2:
            print(
                f"Failed to retrieve commits from {self.start} to {self.end} in {self.repo.src_path}."
            )
            return False
        for count, commit in enumerate(commits):
            ctime = (
                hgutils.get_commit_date(self.repo.src_path, commit)
                if IS_FBCODE
                else gitutils.get_git_commit_date(self.repo.src_path, commit)
            )
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

    def build(self, commit: Commit):
        # checkout target repo commit
        print(
            f"====================== [TORCHBENCH] Checking out and building target repo {self.repo.name} commit {commit.sha} "
            "=======================",
            flush=True,
        )
        if not IS_FBCODE:
            assert gitutils.checkout_git_commit(
                self.repo.src_path.absolute(), commit.sha
            )
            ctime = datetime.strptime(commit.ctime.split(" ")[0], "%Y-%m-%d")
            self._checkout_non_target_repos(ctime)
            # build target repo
            build_repo(self.repo, self.bisection_env)
            # build non target repos
            for repo in self.non_target_repos:
                build_repo(repo, self.bisection_env)
        else:
            assert hgutils.checkout_commit(self.repo.src_path.absolute(), commit.sha)
            build_fbcode_repo(self.repo, self.bisection_env, commit.sha)


class TorchBenchRepo:
    repo: TorchRepo
    target_repo: BisectionTargetRepo
    workdir: Path
    bisection_env: os._Environ
    timelimit: int  # timeout limit in minutes
    first_time: bool

    def __init__(
        self, repo: TorchRepo, target_repo: BisectionTargetRepo, workdir: Path
    ):
        self.repo = repo
        self.target_repo = target_repo
        self.workdir = workdir
        self.first_time = True

    def prep(self, bisection_env: os._Environ) -> bool:
        self.bisection_env = bisection_env
        return True

    def _install_benchmark(self):
        "Install and build TorchBench dependencies"
        command = [sys.executable, "install.py"]
        subprocess.check_call(
            command, cwd=self.repo.src_path.absolute(), env=self.bisection_env
        )

    def _run_benchmark_for_commit(
        self, commit: Commit, bisect_config: TorchBenchABTestResult
    ) -> str:
        # Return the result json file path
        output_dir = os.path.join(self.workdir.absolute(), commit.sha)
        # If the directory already exists, clear its contents
        if os.path.exists(output_dir):
            assert os.path.isdir(
                output_dir
            ), "Must specify output directory: {output_dir}"
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        # If the first time to run benchmark, install the dependencies first
        if self.first_time and not SKIP_INSTALL_TORCHBENCH and not IS_FBCODE:
            self._install_benchmark()
            self.first_time = False
        bm_name = bisect_config.name
        output_file = "metrics-{}.json".format(
            datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
        )
        output_file_path = os.path.join(output_dir, output_file)
        print(
            f"===================== [TORCHBENCH] Running TorchBench for commit: {commit.sha} START =====================",
            flush=True,
        )
        if IS_FBCODE:
            self.bisection_env["CURRENT_HG_COMMIT_HASH"] = commit.sha
        command = (
            [
                sys.executable,
                "run_benchmark.py",
                bm_name,
                "--run-bisect",
                bisect_config.bisection_config_file_path,
                "--output",
                output_file_path,
            ]
            if not IS_FBCODE
            else [
                f"./run_benchmark.par",
                bm_name,
                "--run-bisect",
                bisect_config.bisection_config_file_path,
                "--output",
                output_file_path,
            ]
        )
        work_directory = (
            self.repo.src_path
            if not IS_FBCODE
            else get_fbcode_build_artifact_dir(self.repo, commit.sha)
        )
        subprocess.check_call(command, cwd=work_directory, env=self.bisection_env)
        print(
            f"===================== [TORCHBENCH] Running TorchBench for commit: {commit.sha} END. OUTPUT: {output_file_path} =====================",
            flush=True,
        )
        return output_file_path

    def _gen_digest(self, result_json: str) -> Dict[str, float]:
        out = {}
        if not os.path.getsize(result_json):
            print(f"Empty json file {result_json}. Return empty digest.")
            return out
        with open(result_json, "r") as df:
            data = json.load(df)
        return data

    def get_digest_for_commit(
        self, commit: Commit, abtest_result: Dict[str, Any], debug: bool
    ) -> Dict[str, float]:
        # digest is cached before
        if commit.digest:
            return commit.digest
        # if in debug mode, load from the benchmark file if it exists
        if debug:
            result_dir = os.path.join(self.workdir, commit.sha)
            result_json = get_latest_non_empty_file(
                result_dir, lambda x: x.endswith(".json")
            )
            if result_json:
                commit.digest = self._gen_digest(result_json)
                return commit.digest
        # Build all torch packages
        self.target_repo.build(commit)
        # Run benchmark, return the output json file
        result_json = self._run_benchmark_for_commit(commit, abtest_result)
        commit.digest = self._gen_digest(result_json)
        if not IS_FBCODE:
            print(
                f"================== [TORCHBENCH] Cleaning up packages for commit {commit.sha} ==================",
                flush=True,
            )
            cleanup_torch_packages()
        return commit.digest


class TorchBenchBisection:
    workdir: Path
    torch_repos: Dict[str, TorchRepo]
    target_repo: BisectionTargetRepo
    torchbench: TorchBenchRepo
    bisect_config: TorchBenchABTestResult
    output_json: str
    # Run in debug mode.
    # If true, will try to resume from the previous failed run.
    debug: bool
    # Run in interactive mode.
    # If true, skip running and manually specify good and bad commits.
    interactive: bool
    # left commit, right commit, TorchBenchABTestResult to test.
    bisectq: List[Tuple[Commit, Commit, TorchBenchABTestResult]]
    result: List[Tuple[Commit, Commit]]

    def __init__(
        self,
        workdir: str,
        torch_repos: List[TorchRepo],
        target_repo: TorchRepo,
        start: str,
        end: str,
        start_version: str,
        end_version: str,
        bisect_config: TorchBenchABTestResult,
        output_json: str,
        debug: bool = False,
        interactive: bool = False,
    ):
        self.workdir = Path(workdir)
        self.torch_repos = torch_repos
        non_target_repos = list(
            filter(
                lambda x: not x.name == target_repo.name and not x.name == "torchbench",
                torch_repos.values(),
            )
        )
        torchbench_repo_key = "torchbench" if not IS_FBCODE else "fbcode"
        self.target_repo = BisectionTargetRepo(
            repo=target_repo,
            start=start,
            end=end,
            start_version=start_version,
            end_version=end_version,
            non_target_repos=non_target_repos,
        )
        self.torchbench = TorchBenchRepo(
            repo=torch_repos[torchbench_repo_key],
            target_repo=self.target_repo,
            workdir=self.workdir,
        )
        self.bisect_config = bisect_config
        self.bisectq = list()
        self.result = list()
        self.output_json = output_json
        self.debug = debug
        self.interactive = interactive

    def prep(self) -> bool:
        if not IS_FBCODE:
            cleanup_torch_packages()
        if not self.target_repo.prep(self.interactive):
            return False
        if not self.torchbench.prep(self.target_repo.bisection_env):
            return False
        left_commit = self.target_repo.commits[0]
        right_commit = self.target_repo.commits[-1]
        self.bisectq.append((left_commit, right_commit, self.bisect_config))
        return True

    # Left: older commit, right: newer commit, target: TorchBenchABTestResult
    # Return: List of [left, right, TorchBenchABTestResult] that satisfy the regression rule
    def regression_detection(
        self, left: Commit, right: Commit
    ) -> TorchBenchABTestResult:
        # If uncalculated, commit.digest will be None
        assert left.digest, "Commit {left.sha} must have a digest"
        assert right.digest, "Commit {right.sha} must have a digest"
        regression_result = generate_regression_result(left.digest.copy(), right.digest.copy())
        regression_file = f"regression-{left.sha}-{right.sha}.yaml"
        regression_file_full_path = os.path.join(
            self.workdir.absolute(), regression_file
        )
        with open(regression_file_full_path, "w") as rf:
            rf.write(yaml.safe_dump(asdict(regression_result)))
        regression_result.bisection_config_file_path = regression_file_full_path
        return regression_result

    def interactive_signal(self, commit: Commit) -> bool:
        """Prompt for good or bad commit from user input."""
        val = input(f"Commit {commit.sha} good or bad (G/B)? ").strip()
        if val == "G":
            return True
        elif val == "B":
            return False
        else:
            assert False, "We only accept G or B as user input."

    def run(self):
        while len(self.bisectq):
            (left, right, abtest_result) = self.bisectq.pop(0)
            if self.interactive:
                left_signal = self.interactive_signal(left)
                right_signal = self.interactive_signal(right)
                signal = not (left_signal == right_signal)
                updated_abtest_result = None
            else:
                self.torchbench.get_digest_for_commit(left, abtest_result, self.debug)
                self.torchbench.get_digest_for_commit(right, abtest_result, self.debug)
                updated_abtest_result = self.regression_detection(left, right)
                signal = bool(len(updated_abtest_result.details))
            print(f"Left commit: {left.sha}, right commit: {right.sha}, signal: {signal}")
            if signal:
                mid = self.target_repo.get_mid_commit(left, right)
                if mid == None:
                    self.result.append((left, right))
                else:
                    self.bisectq.append((left, mid, updated_abtest_result))
                    self.bisectq.append((mid, right, updated_abtest_result))

    def output(self):
        json_obj = dict()
        json_obj["target_repo"] = self.target_repo.repo.name
        json_obj["start"] = self.target_repo.start
        json_obj["start_version"] = self.target_repo.start_version
        json_obj["end"] = self.target_repo.end
        json_obj["end_version"] = self.target_repo.end_version
        json_obj["result"] = []
        for res in self.result:
            r = dict()
            r["commit1"] = res[0].sha
            r["commit1_time"] = res[0].ctime
            r["commit1_digest"] = res[0].digest
            r["commit2"] = res[1].sha
            r["commit2_time"] = res[1].ctime
            r["commit2_digest"] = res[1].digest
            json_obj["result"].append(r)
        with open(self.output_json, "w") as outfile:
            json.dump(json_obj, outfile, indent=2)
        print(f"Bisection successful. Result saved to {self.output_json}.")
        print(json_obj)


def main() -> None:
    global SKIP_INSTALL_TORCHBENCH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        required=True,
        help="bisection working directory for logs and results",
        type=exist_dir_path,
    )
    parser.add_argument(
        "--torch-repos-path",
        required=True,
        help="the directory of pytorch/* source code repositories, or fbcode repo if running internally",
        type=exist_dir_path,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="the regression dict output of regression_detector.py in YAML",
        type=exist_file_path,
    )
    parser.add_argument(
        "--torchbench-repo-path",
        default=None,
        help="the directory of torchbench source code git repository, if None, use `args.torch_repo_path/benchmark`.",
        type=exist_dir_path,
    )
    parser.add_argument(
        "--skip-install-torchbench",
        action="store_true",
        help="Skip installing torchbench",
    )
    parser.add_argument("--output", required=True, help="the output json file")
    parser.add_argument(
        "--skip-update",
        type=str,
        default="torchbench",
        help="Repositories to skip update.",
    )
    parser.add_argument(
        "--gh-issue-path",
        default="gh-issue.md",
        help="Output path to print the issue body"
    )
    # by default, debug mode is disabled
    parser.add_argument(
        "--debug",
        help="run in debug mode, if the result json exists, use it directly",
        action="store_true",
    )
    parser.add_argument(
        "--interactive",
        help="run in interactive mode, manually specify good and bad commits",
        action="store_true",
    )
    args = parser.parse_args()

    bisect_config = parse_abtest_result_from_regression_file_for_bisect(args.config)
    # sanity checks
    assert (
        bisect_config.name
    ), "Invalid bisection config, must specify userbenchmark name."
    assert bisect_config.control_env[
        "git_commit_hash"
    ], "Invalid bisection config, must specify control group commit hash."
    assert bisect_config.treatment_env[
        "git_commit_hash"
    ], "Invalid bisection config, must specify treatment group commit hash."
    assert bisect_config.bisection in TORCHBENCH_BISECTION_TARGETS.keys(), (
        f"Invalid bisection config, "
        f"get bisection target repo {bisect_config.bisection}, "
        f"available target repos: {TORCHBENCH_BISECTION_TARGETS.keys()}"
    )
    assert bisect_config.bisection_mode == "bisect", "Abtest mode is not supported yet."
    assert len(bisect_config.details), "The bisection target metrics must not be empty."

    if args.skip_update:
        skip_update_repos = list(map(lambda x: x.strip(), args.skip_update.split(",")))
        for repo in skip_update_repos:
            assert repo in list(
                TORCHBENCH_BISECTION_TARGETS.keys()
            ), f"User specified skip update repo {repo} not in list: {TORCHBENCH_BISECTION_TARGETS.keys()}"
    else:
        skip_update_repos = None
    if args.skip_install_torchbench:
        SKIP_INSTALL_TORCHBENCH = True

    # load, update, and clean the repo directories
    torch_repos: Dict[str, TorchRepo] = get_updated_clean_torch_repos(
        args.torch_repos_path, args.torchbench_repo_path, skip_update_repos
    )
    target_repo = torch_repos[bisect_config.bisection]
    start_hash = (
        gitutils.get_torch_main_commit(
            target_repo.src_path.absolute(),
            bisect_config.control_env["git_commit_hash"],
        )
        if not IS_FBCODE
        else bisect_config.control_env["git_commit_hash"]
    )
    end_hash = (
        gitutils.get_torch_main_commit(
            target_repo.src_path.absolute(),
            bisect_config.treatment_env["git_commit_hash"],
        )
        if not IS_FBCODE
        else bisect_config.treatment_env["git_commit_hash"]
    )

    bisection = TorchBenchBisection(
        workdir=args.work_dir,
        torch_repos=torch_repos,
        target_repo=target_repo,
        start=start_hash,
        end=end_hash,
        start_version=bisect_config.control_env.get("pytorch_version", "N/A"),
        end_version=bisect_config.treatment_env.get("pytorch_version", "N/A"),
        bisect_config=bisect_config,
        output_json=args.output,
        debug=args.debug,
        interactive=args.interactive,
    )
    if start_hash == end_hash:
        print(f"Start and end hash are the same: {start_hash}. Skip bisection")
        bisection.output()
        exit(0)
    assert bisection.prep(), "The working condition of bisection is not satisfied."
    print(
        "Preparation steps ok. Commit to bisect: "
        + " ".join([str(x) for x in bisection.target_repo.commits])
    )
    bisection.run()
    bisection.output()
    # Format the output into a github issue if the bisector finds the root cause commit
    if bisection.result:
        process_bisection_into_gh_issue(bisection.output_json, args.gh_issue_path)

if __name__ == "__main__":
    main()  # pragma: no cover
