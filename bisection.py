"""bisection.py
Runs bisection to determine PRs that cause performance change.
It assumes that the pytorch, torchbench, torchtext and torchvision repositories provided are all clean with the latest code.
By default, the torchvision and torchtext package version will be fixed to the latest commit on the pytorch commit date.

Usage:
  python bisection.py --work-dir <WORK-DIR> \
    --pytorch-src <PYTORCH_SRC_DIR> \
    --torchbench-src <TORCHBENCH_SRC_DIR> \
    --config <BISECT_CONFIG> --output <OUTPUT_FILE_PATH>
"""

import os
import json
import yaml
import argparse
import typing
from tabulate import tabulate
import re
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from torchbenchmark.util import gitutils

TORCH_GITREPO="https://github.com/pytorch/pytorch.git"
TORCHBENCH_GITREPO="https://github.com/pytorch/benchmark.git"
TORCHBENCH_DEPS = {
    "torchtext": os.path.expandvars("${HOME}/text"),
    "torchvision": os.path.expandvars("${HOME}/vision"),
}

def exist_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# Translates test name to filter
# For example, ["test_eval[yolov3-cpu-eager]", "test_train[yolov3-gpu-eager]"]
#     -> "((eval and yolov3 and cpu and eager) or (train and yolov3 and gpu and eager))"
# If targets is None, run everything except slomo
def targets_to_bmfilter(targets: List[str], models: List[str]) -> str:
    bmfilter_names = []
    if targets == None or len(targets) == 0:
        return "(not slomo)"
    for test in targets:
        regex = re.compile("test_(train|eval)\[([a-zA-Z0-9_]+)-([a-z]+)-([a-z]+)\]")
        m = regex.match(test)
        if not m:
            if test in models:
                partial_name = test
            else:
                print(f"Cannot recognize the TorchBench filter: {test}. Exit.")
                exit(1)
        else:
            partial_name = " and ".join(m.groups())
        bmfilter_names.append(f"({partial_name})")
    return "(" + " or ".join(bmfilter_names) + ")"

# Find the latest non-empty json file in the directory
def find_latest_json_file(result_dir: str):
    json_files = list(filter(lambda x: x.endswith(".json"), os.listdir(result_dir)))
    json_files.sort(reverse=True)
    for f in json_files:
        # Return the first non-empty json file
        json_path = os.path.join(result_dir, f)
        if os.path.exists(json_path) and os.stat(json_path).st_size:
            return json_path
    print(f"Can't find non-empty json files in path: {result_dir}")
    return str()

def get_delta_str(reference: float, current: float) -> str:
    delta_num = ((current - reference) / current * 100)
    delta_str = "{:+3f}".format(delta_num) + "%"
    if (abs(delta_num) >= 5):
        delta_str = delta_str + "*"
    return delta_str

def get_means(data):
    rc = dict()
    for param in data["benchmarks"]:
        name = param["name"]
        mean = param["stats"]["mean"]
        rc[name] = mean
    return rc

def analyze_abtest_result_dir(result_dir: str):
    dirs = [ os.path.join(result_dir, name) for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name)) ]
    delta = False
    json_files = list(filter(len, map(find_latest_json_file, dirs)))
    out = [['Benchmark']]
    assert(len(json_files), f"Don't find benchmark result files in {result_dir}.")
    # If there are only two json files, we believe it is an abtest, so print delta of the mean
    if len(json_files) == 2:
        delta = True
    with open(json_files[0], "r") as fp:
        cur_result = json.load(fp)
        means = get_means(cur_result)
    for key in means:
        out.append([])
        out[-1].append(key)
    for index, json_file in enumerate(json_files):
        with open(json_file, "r") as fp:
            jsonobj = json.load(fp)
        header = f"Run {os.path.basename(os.path.dirname(json_file))}"
        out[0].append(header)
        means = get_means(jsonobj)
        if delta and index == 0:
            reference = means
        for key_index, key in enumerate(means):
            out[key_index+1].append(means[key])
            if delta and index == 1:
                out[0].append("Delta")
                out[key_index+1].append(get_delta_str(reference[key], means[key]))
    out_str = tabulate(out, headers='firstrow')
    return out_str

class Commit:
    sha: str
    ctime: str
    digest: Dict[str, float]
    def __init__(self, sha, ctime):
        self.sha = sha
        self.ctime = ctime
        self.digest = None
    def __str__(self):
        return self.sha

class TorchSource:
    srcpath: str
    build_lazy: bool
    commits: List[Commit]
    # Map from commit SHA to index in commits
    commit_dict: Dict[str, int]
    def __init__(self, srcpath: str, build_lazy: bool):
        self.srcpath = srcpath
        self.build_lazy = build_lazy
        self.commits = []
        self.commit_dict = dict()

    def prep(self) -> bool:
        repo_origin_url = gitutils.get_git_origin(self.srcpath)
        if not repo_origin_url == TORCH_GITREPO:
            print(f"WARNING: Unmatched repo origin url: {repo_origin_url} with standard {TORCH_GITREPO}")
        self.update_repos()
        return True

    # Update pytorch, torchtext, and torchvision repo
    def update_repos(self):
        repos = [(self.srcpath, "master")]
        for value in TORCHBENCH_DEPS.values():
            repos.append((value, "main"))
        for (repo, branch) in repos:
            gitutils.clean_git_repo(repo)
            assert gitutils.update_git_repo(repo, branch), f"Failed to update {branch} branch of {repo}."

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

    def setup_build_env(self, env) -> Dict[str, str]:
        env["USE_CUDA"] = "1"
        env["BUILD_CAFFE2_OPS"] = "0"
        env["USE_XNNPACK"] = "0"
        env["USE_MKLDNN"] = "1"
        env["USE_MKL"] = "1"
        env["USE_CUDNN"] = "1"
        env["CMAKE_PREFIX_PATH"] = env["CONDA_PREFIX"]
        return env

    # Checkout the last commit of dependencies on date
    def checkout_deps(self, cdate: datetime):
        for pkg in TORCHBENCH_DEPS:
            dep_commit = gitutils.get_git_commit_on_date(TORCHBENCH_DEPS[pkg], cdate)
            print(f"Checking out {pkg} commit {dep_commit} ...", end="", flush=True)
            assert dep_commit, "Failed to find the commit on {cdate} of {pkg}"
            assert gitutils.checkout_git_commit(TORCHBENCH_DEPS[pkg], dep_commit), "Failed to checkout commit {commit} of {pkg}"
            print("done.")

    # Install dependencies such as torchtext and torchvision
    def build_install_deps(self, build_env):
        # Build torchvision
        print(f"Building torchvision ...", end="", flush=True)
        command = "python setup.py install"
        subprocess.check_call(command, cwd=TORCHBENCH_DEPS["torchvision"], env=build_env, shell=True)
        print("done")
        # Build torchtext
        print(f"Building torchtext ...", end="", flush=True)
        command = "python setup.py clean install"
        subprocess.check_call(command, cwd=TORCHBENCH_DEPS["torchtext"], env=build_env, shell=True)
        print("done")

    def _build_lazy_tensor(self, commit: Commit, build_env: Dict[str, str]):
        if self.build_lazy:
            print(f"Building pytorch lazy tensor on {commit.sha} ...", end="", flush=True)
            lazy_tensor_path = os.path.join(self.srcpath, "lazy_tensor_core")
            command = "./scripts/apply_patches.sh"
            subprocess.check_call(command, cwd=lazy_tensor_path, env=build_env, shell=True)
            command = "python setup.py install"
            subprocess.check_call(command, cwd=lazy_tensor_path, env=build_env, shell=True)
            print("done")

    def build(self, commit: Commit):
        # checkout pytorch commit
        print(f"Checking out pytorch commit {commit.sha} ...", end="", flush=True)
        gitutils.checkout_git_commit(self.srcpath, commit.sha)
        print("done.")
        # checkout pytorch deps commit
        ctime = datetime.strptime(commit.ctime.split(" ")[0], "%Y-%m-%d")
        self.checkout_deps(ctime)
        # setup environment variables
        build_env = self.setup_build_env(os.environ.copy())
        # build pytorch
        print(f"Building pytorch commit {commit.sha} ...", end="", flush=True)
        # Check if version.py exists, if it does, remove it.
        # This is to force pytorch update the version.py file upon incremental compilation
        version_py_path = os.path.join(self.srcpath, "torch/version.py")
        if os.path.exists(version_py_path):
            os.remove(version_py_path)
        command = "python setup.py install"
        subprocess.check_call(command, cwd=self.srcpath, env=build_env, shell=True)
        print("done")
        # build pytorch lazy tensor if needed
        self._build_lazy_tensor(commit, build_env)
        self.build_install_deps(build_env)

    def cleanup(self, commit: Commit):
        print(f"Cleaning up packages from commit {commit.sha} ...", end="", flush=True)
        packages = ["torch", "torchtext", "torchvision"]
        command = "pip uninstall -y " + " ".join(packages) + " &> /dev/null "
        subprocess.check_call(command, shell=True)
        print("done")

class TorchBench:
    srcpath: str # path to pytorch/benchmark source code
    timelimit: int # timeout limit in minutes
    workdir: str
    devbig: str
    models: List[str]
    torch_src: TorchSource

    def __init__(self, srcpath: str,
                 torch_src: TorchSource,
                 timelimit: int,
                 workdir: str,
                 devbig: str):
        self.srcpath = srcpath
        self.torch_src = torch_src
        self.timelimit = timelimit
        self.workdir = workdir
        self.devbig = devbig
        self.models = list()

    def prep(self) -> bool:
        # Verify the code in srcpath is pytorch/benchmark
        repo_origin_url = gitutils.get_git_origin(self.srcpath)
        if not repo_origin_url == TORCHBENCH_GITREPO:
            print(f"WARNING: Unmatched repo origin url: {repo_origin_url} with standard {TORCHBENCH_GITREPO}")
        # get list of models
        self.models = [ model for model in os.listdir(os.path.join(self.srcpath, "torchbenchmark", "models"))
                        if os.path.isdir(os.path.join(self.srcpath, "torchbenchmark", "models", model)) ]
        return True

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
        print(f"Running TorchBench for commit: {commit.sha}, filter {bmfilter} ...", end="", flush=True)
        if not self.devbig:
            command = f"""bash .github/scripts/run.sh "{output_dir}" "{bmfilter}" 2>&1 | tee {output_dir}/benchmark.log"""
        else:
            command = f"""bash .github/scripts/run-devbig.sh  "{output_dir}" "{bmfilter}" "{self.devbig}" 2>&1 | tee {output_dir}/benchmark.log"""
        try:
            subprocess.check_call(command, cwd=self.srcpath, shell=True, timeout=self.timelimit * 60)
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
        self.torch_src.cleanup(commit)
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
                 devbig: str,
                 build_lazy: bool = False,
                 debug: bool = False):
        self.workdir = workdir
        self.start = start
        self.end = end
        self.threshold = threshold
        self.direction = direction
        self.targets = targets
        self.bisectq = list()
        self.result = list()
        self.torch_src = TorchSource(srcpath = torch_src, build_lazy=build_lazy)
        self.bench = TorchBench(srcpath = bench_src,
                                torch_src = self.torch_src,
                                timelimit = timeout,
                                devbig = devbig,
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
        if not self.torch_src.prep():
            return False
        if not self.torch_src.init_commits(self.start, self.end, self.abtest):
            return False
        if not self.bench.prep():
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

    def output_abtest_result(self):
        abtest_result = analyze_abtest_result_dir(self.workdir)
        with open(self.output_json, 'w') as outfile:
            outfile.write(abtest_result)
        print(abtest_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",
                        help="bisection working directory",
                        type=exist_dir_path)
    parser.add_argument("--pytorch-src",
                        help="the directory of pytorch source code git repository",
                        type=exist_dir_path)
    parser.add_argument("--torchbench-src",
                        help="the directory of torchbench source code git repository",
                        type=exist_dir_path)
    parser.add_argument("--config",
                        help="the bisection configuration in YAML format")
    parser.add_argument("--output",
                        help="the output json file")
    parser.add_argument("--analyze-result",
                        help="specify the the output result directory to analyze")
    # running on devbig
    parser.add_argument("--devbig",
                        type=str,
                        help="if running on devbig, specify the devbig conda env")
    # by default, do not build lazy tensor
    parser.add_argument("--build-lazy",
                        action='store_true',
                        help="build lazy tensor feature in PyTorch")
    # by default, debug mode is disabled
    parser.add_argument("--debug",
                        help="run in debug mode, if the result json exists, use it directly",
                        action='store_true')
    args = parser.parse_args()

    # If this is to print the overview of a test result, don't need to run the actual execution
    if args.analyze_result:
        print(analyze_abtest_result_dir(args.analyze_result))
        exit(0)

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

    bisection = TorchBenchBisection(workdir=args.work_dir,
                                    torch_src=args.pytorch_src,
                                    bench_src=args.torchbench_src,
                                    start=bisect_config["start"],
                                    end=bisect_config["end"],
                                    threshold=bisect_config["threshold"],
                                    direction=bisect_config["direction"],
                                    timeout=bisect_config["timeout"],
                                    targets=targets,
                                    output_json=args.output,
                                    devbig=args.devbig,
                                    build_lazy=True,
                                    debug=args.debug)
    assert bisection.prep(), "The working condition of bisection is not satisfied."
    print("Preparation steps ok. Commit to bisect: " + " ".join([str(x) for x in bisection.torch_src.commits]))
    bisection.run()
    if bisection.abtest:
        bisection.output_abtest_result()
    else:
        bisection.output()
