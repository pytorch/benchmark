"""
Run Dev Infra nightly benchmarking.
"""
import subprocess
import argparse

from typing import List
from ..utils import REPO_PATH, get_output_dir

BM_NAME = "devinfra-nightly"

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="devinfra/cuda", help="The name of the config to run.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the benchmark.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    return parser.parse_args(args)

def run_benchmark(config, logdir, iter, dryrun=True):
    run_config_script = REPO_PATH.joinpath(".github", "scripts", "run-config.py")
    run_command = [str(run_config_script), "-c", config, "-b", str(REPO_PATH), "-o", str(logdir)]
    print(f"Iter {iter}: running {run_command}")
    if not dryrun:
        subprocess.check_call(run_command, shell=True)

def run(args: List[str]):
    args = parse_args(args)
    output_dir = get_output_dir(BM_NAME)
    logdir = output_dir.joinpath("logs")
    for iter in range(args.repeat):
        if args.repeat != 1:
            logdir = logdir.joinpath(f"iter-{iter}")
        run_benchmark(args.config, logdir, iter=iter, dryrun=args.dryrun)
