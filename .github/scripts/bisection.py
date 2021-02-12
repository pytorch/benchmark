"""bisection.py
Runs bisection to determine PRs that cause performance regression.
Performance regression is defined by TorchBench score drop greater than the threshold.

Usage:
  python bisection.py --pytorch-src <PYTORCH_SRC_DIR> \
    --torchbench-src <TORCHBENCH_SRC_DIR> \
    --start <SHA> --end <SHA> --threshold <SCORE_THRESHOLD> \
    --timeout <TIMEOUT_IN_MINS>

# threshold: the TorchBench threshold to identify regression
# timeout: if a PR execution exceeds timeout limit, it will be terminated with score zero
"""

import os
import argparse

# Bisection Algorithm: for the bisection range [start, end]
# Step 1: Fetch commit list: [start, ..., mid, ..., end]
# Step 2: Put pair (start, end) into queue bisectq
# Step 3: Get the first pair (start, end) in bisectq. If end.index - start.index == 1, return pair(start, end)
# Step 4: Else, test benchmark on commit start and end
# If abs(end.score - start.score) < threshold: delete this range
# If abs(end.score - start.score) > threshold:
#     Test mid, if abs(end.score - mid.score) > threshold: insert (mid, end) into the bisectq
#               if abs(start.score - mid.score) > threshold: insert (start, mid) into the bisectq
# Step 5: goto step 2 until bisectq is empty

## Helper functions
def exist_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

## Class definitions
class Commit:
    sha: str
    score: float

class TorchBench:
    srcpath: str # path to pytorch/benchmark source code
    branch: str
    timeout_limit: int # timeout limit in minutes

    def __init__(self, srcpath: str,
                 branch: str = "0.1"):
        self.srcpath = srcpath
        self.branch = branch

    def verify(self) -> bool:
        pass
        
    def build_benchmark(self):
        # Checkout branch
        pass

    def run_benchmark(self):
        pass

class TorchSource:
    srcpath: str
    commits: List[Commit]
    # Map from commit SHA to index in commits
    commit_dict: Dict[str, int]
    def __init__(self, srcpath: str):
        self.srcpath = srcpath

    def verify() -> bool:
        pass
 
    def init_commits(self, start: str, end: str) -> List[str]:
        pass
    
    def get_mid_commit(start: str, end: str) -> Option[str]:
        pass

    def checkout(self, commit: str):
        pass

    def build(self, commit: str):
        pass

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
    output_json: str

    def __init__(self,
                 workdir: str,
                 pytorch_src: str,
                 bench_src: str,
                 start: str,
                 end: str,
                 threshold: int,
                 timeout: int,
                 output_json: str):
        self.start = start
        self.end = end
        self.threshold = threshold
        self.timeout = timeout
        self.output = output
        self.bisectq = list()
        self.torch_src = TorchSource(srcpath=pytorch_src)
        self.bench = TorchBench(srcpath=bench_src)
        self.output_json = output_json

    # verify all working conditions satisfy
    def verify(self):
        pass

    def prep_bisection(self):
        def get_commits(self) -> bool:
            commits = torch_src.get_commits(start, end)
            self.bisectq.append(commits)
        
    def run_bisection(self):
        while len(commit_ranges):

    def print_result(self):
        print(f"PR Bisection from {start} to {end} is successful.")
        for pair in result:
            print(f"\tcommit {pair[0].sha} score {pair[0].score}, commit {pair[1].sha} score {pair[1].score}")

    def dump_result(self):
        
        
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
                                    pytorch_src=args.pytorch_src,
                                    bench_src=args.torchbench_src,
                                    start=args.start,
                                    end=args.end,
                                    threshold=args.threshold,
                                    timeout=args.timeout
                                    output=args.output)
    assert bisection.verify(), "The working condition of bisection is not satisfied."
    bisection.prep()
    bisection.run()
    bisection.print_result()
    bisection.dump_result()
