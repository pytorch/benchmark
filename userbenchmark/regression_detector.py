"""
The regression detector of TorchBench Userbenchmark.
"""
import json
import argparse
import importlib
from dataclasses import dataclass, asdict
import os
import yaml
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, Dict
from .utils import PLATFORMS
from .utils import add_path, REPO_PATH, USERBENCHMARK_OUTPUT_PREFIX

with add_path(REPO_PATH):
    from utils.s3_utils import S3Client

@dataclass
class TorchBenchABTestMetric:
    control: float
    treatment: float
    delta: float

@dataclass
class TorchBenchABTestResult:
    control_env: Dict[str, str]
    treatment_env: Dict[str, str]
    details: Dict[str, TorchBenchABTestMetric]
    bisection: Optional[str]

def call_userbenchmark_detector(detector, start_file: str, end_file: str) -> Optional[TorchBenchABTestResult]:
    return detector(start_file, end_file)

def get_default_output_path(bm_name: str) -> str:
    output_path = os.path.join(REPO_PATH, USERBENCHMARK_OUTPUT_PREFIX, bm_name)
    fname = "regression-{}.yaml".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    return os.path.join(output_path, fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("userbenchmark", help="Name of the userbenchmark to detect regression.")
    # Local metrics file comparison
    parser.add_argument("--control", default=None, help="The control group metrics file for comparison.")
    parser.add_argument("--treatment", default=None, help="The treatment metrics file for comparison.")
    # S3 metrics file comparison
    parser.add_argument("--platform", choices=PLATFORMS, default=None, help="The name of platform of the regression.")
    parser.add_argument("--start-date", default=None, help="The start date to detect regression.")
    parser.add_argument("--end-date", default=None, help="The latest date to detect regression.")
    # output file path
    parser.add_argument("--output", default=None, help="Output path to print the regression detection file.")
    args = parser.parse_args()

    detector = importlib.import_module(f"{args.userbenchmark}.regression_detector").run

    if args.control and args.treatment:
        with open(args.control, "r") as cfptr:
            control = json.load(cfptr)
        with open(args.treatment, "r") as tfptr:
            treatment = json.load(tfptr)
        # Local file comparison, return the regression detection object
        result = call_userbenchmark_detector(detector, control, treatment)
        if result:
            if not args.output:
                # Write result to $REPO_DIR/.userbenchmark/<userbenchmark-name>/regression-<time>.json
                assert control["name"] == treatment["name"], f'Expected the same userbenchmark name from metrics files, \
                                                            but getting {control["name"]} and {treatment["name"]}.'
                bm_name = control["name"]
                args.output = get_default_output_path(bm_name)
            # dump result to yaml file
            result_dict = asdict(result)
            # create the output directory if doesn't exist
            output_dir = Path(os.path.dirname(args.output))
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as ofptr:
                ofptr.write(yaml.safe_dump(result_dict))
        else:
            print(f"No performance signal detected between file {args.control} and {args.treatment}.")
    else:
        # S3 path
        print("Comparison for metrics from Amazon S3 is WIP.")
