import argparse
import os
import subprocess
import sys
from pathlib import Path

from aicluster import run_aicluster_benchmark

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()


class add_path:
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

    # Check if userbenchmark has an installer
    candidate_installer_path = os.path.join(
        workdir, "userbenchmark", ub_name, "install.py"
    )
    if os.path.exists(candidate_installer_path):
        install_command = [sys.executable, "install.py"]
        print(f"Running user benchmark installer: {install_command}")
        if not dryrun:
            subprocess.check_call(
                install_command, cwd=Path(candidate_installer_path).parent.resolve()
            )

    command = [sys.executable, "run_benchmark.py", ub_name]
    print(f"Running user benchmark command: {command}")
    if not dryrun:
        subprocess.check_call(command, cwd=workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        choices=["gcp_a100", "aws_t4_metal", "aws_c5_24xlarge", "ai_cluster"],
        required=True,
        help="specify the benchmark platform.",
    )
    parser.add_argument(
        "--dryrun", action="store_true", help="only dry run the command."
    )
    args = parser.parse_args()
    benchmarks = get_userbenchmarks_by_platform(args.platform)
    if args.platform == "ai_cluster":
        assert not args.dryrun, "AICluster workflow doesn't support dryrun."
        for ub in benchmarks:
            run_aicluster_benchmark(ub, check_success=True, upload_scribe=True)
    else:
        for ub in benchmarks:
            run_userbenchmark(ub_name=ub, dryrun=args.dryrun)
