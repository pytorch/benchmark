import argparse
import sys
import subprocess
from pathlib import Path

from aicluster import run_aicluster_benchmark

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

with add_path(str(REPO_ROOT)):
    from userbenchmark import get_userbenchmarks_by_platform

def run_userbenchmark(ub_name, dryrun=True):
    workdir = REPO_ROOT
    command = [sys.executable, "run_benchmark.py", ub_name]
    print(f"Running user benchmark command: {command}")
    if not dryrun:
        subprocess.check_call(command, cwd=workdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["gcp_a100", "aws_t4_metal", "ai_cluster"], required=True, help="specify the benchmark platform.")
    parser.add_argument("--dryrun", action="store_true", help="only dry run the command.")
    args = parser.parse_args()
    benchmarks = get_userbenchmarks_by_platform(args.platform)
    if args.platform == "ai-cluster":
        assert not args.dryrun, "AICluster workflow doesn't support dryrun."
        for ub in benchmarks:
            run_aicluster_benchmark(ub, check_success=True, upload_scribe=True)
    else:
        for ub in benchmarks:
            run_userbenchmark(ub_name=ub, dryrun=args.dryrun)
