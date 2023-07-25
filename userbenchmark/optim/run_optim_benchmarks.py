#!/bin/bash python3
'''
This script is intended for the CI context only! The whole purpose behind this script is to enable
process/context/memory isolation across different models and devices. The OG script (which this
script calls) is the userbenchmark/optim/run.py script, which is better documented and what is
intended to be used locally. The current script is simply a wrapper that dispatches serial 
subprocesses to run the OG script and handles the metrics.json merging afterwards.

WARNING! Running this script will wipe clean the OUTPUT_DIR, .userbenchmark/optim/tmp!
'''

from pathlib import Path
import shutil
import subprocess
from typing import Any, List, Dict, Tuple
import argparse
import sys
import itertools
import json
from userbenchmark.utils import REPO_PATH, add_path, dump_output, get_output_json

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models


BM_NAME: str = 'optim'
MODEL_NAMES: List[str] = list_models()

# NOTE: While it is possible to run these benchmarks on CPU, we skip running on CPU in CI because CPU stats can be
# unstable and we had stopped reporting them. You'll still be able to use the run.py script to run CPU though, as
# it may be useful as a more local comparison point for implementations like forloop.
DEVICES: List[str] = ['cuda']

OUTPUT_DIR: Path = REPO_PATH.joinpath('.userbenchmark/optim/tmp')


# Capture the specified models and devices we want to run to avoid redundant work,
# but send the rest of the user arguments to the underlying optim benchmark runner.
def parse_args() -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    parser = argparse.ArgumentParser(description='Run optim benchmarks per model and device')
    parser.add_argument(
        '--models', '-m',
        nargs='*',
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help='List of models to run tests on')
    parser.add_argument(
        '--devices', '-d',
        nargs='*',
        default=DEVICES,
        choices=DEVICES,
        help='List of devices to run tests on')
    return parser.parse_known_args()


def main() -> None:
    args, optim_bm_args = parse_args()
    assert not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.glob("*")), \
           f'{OUTPUT_DIR} must be empty or nonexistent. Its contents will be wiped by this script.'

    # Run benchmarks in subprocesses to take isolate contexts and memory
    for m, d in itertools.product(args.models, args.devices):
        command = [sys.executable, '-m', 'userbenchmark.optim.run', '--continue-on-error',
                   '--output-dir', OUTPUT_DIR, '--models', m, '--devices', d] + optim_bm_args
        # Use check=True to force this process to go serially since our capacity
        # only safely allows 1 model at a time
        completed_process = subprocess.run(command, check=True)
        # While it is certainly unexpected for a subprocess to fail, we don't want to halt entirely
        # as there can be valuable benchmarks to gather from the other subprocesses.
        if completed_process.returncode != 0:
            print(f'OH NO, the subprocess for model {m} and device {d} exited with {completed_process.returncode}!')
    
    # Nightly CI expects ONE metrics json in .userbenchmark/optim, but we may have multiple, so
    # consolidate them into one file.
    aggregated_metrics = {}
    for file_path in Path(OUTPUT_DIR).glob("metrics*.json"):
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            aggregated_metrics.update(json_data['metrics'])
    dump_output(BM_NAME, get_output_json(BM_NAME, aggregated_metrics))

    # Gotta delete the tmp folder--otherwise the nightly CI will think there are multiple metrics jsons!
    shutil.rmtree(OUTPUT_DIR)


if __name__ == '__main__':
    main()
