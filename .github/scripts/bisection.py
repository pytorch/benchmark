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

class Commit:
    sha: str
    score: float
    index: int

class TorchBench:
    path: str # path to pytorch/benchmark source code
    output_dir: str # output directory that stores result
    timeout_limit: int # timeout limit in minutes
    def build_benchmark():
        pass
    def run_benchmark():
        pass

class TorchSource:
    path: str
    commits: List[Commit]
    # Map from commit SHA to index in commits
    commit_dict: Dict[str, int]
    def get_commits(start: str, end: str) -> List[str]:
        pass
    def checkout(commit):
        pass
    def build():
        pass

class TorchBenchBisection:
    start: str
    end: str
    threshold: int
    commit_ranges: List[List[str]]
    torch_src: TorchSource

    def run_bisection():
        while len(commit_ranges):
            
    def get_commits(start: str, end: str):
        commits = torch_src.get_commits(start, end)
        commit_ranges.append(commits)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-src",
                        help="the directory of pytorch source code git repository")
    parser.add_argument("--torchbench-src",
                        help="the directory of torchbench source code git repository")
    parser.add_argument("--start",
                        help="7-digit SHA hash of the start commit to bisect")
    parser.add_argument("--end",
                        help="7-digit SHA hash of the end commit to bisect")
    parser.add_argument("--threshold",
                        help="the torchbench score threshold to report a regression")
    parser.add_argument("--timeout",
                        help="the maximum time to run the benchmark")
    parser.add_argument("--output",
                        help="the output file name and path")
