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
from typing import Optional
from userbenchmark.utils import PLATFORMS, USERBENCHMARK_OUTPUT_PREFIX, REPO_PATH, \
                                TorchBenchABTestResult


def call_userbenchmark_detector(detector, start_file: str, end_file: str) -> Optional[TorchBenchABTestResult]:
    return detector(start_file, end_file)

def get_default_output_path(bm_name: str) -> str:
    output_path = os.path.join(REPO_PATH, USERBENCHMARK_OUTPUT_PREFIX, bm_name)
    fname = "regression-{}.yaml".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    return os.path.join(output_path, fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Local metrics file comparison
    parser.add_argument("--control", default=None, help="The control group metrics file for comparison.")
    parser.add_argument("--treatment", default=None, help="The treatment metrics file for comparison.")

    # S3 metrics file comparison
    parser.add_argument("--name", help="Name of the userbenchmark to detect regression.")
    parser.add_argument("--platform", choices=PLATFORMS, default=None, help="The name of platform of the regression.")
    parser.add_argument("--start-date", default=None, help="The start date to detect regression.")
    parser.add_argument("--end-date", default=None, help="The latest date to detect regression.")
    # output file path
    parser.add_argument("--output", default=None, help="Output path to print the regression detection file.")
    args = parser.parse_args()

    if args.control and args.treatment:
        with open(args.control, "r") as cfptr:
            control = json.load(cfptr)
        with open(args.treatment, "r") as tfptr:
            treatment = json.load(tfptr)
        # Write result to $REPO_DIR/.userbenchmark/<userbenchmark-name>/regression-<time>.json
        assert control["name"] == treatment["name"], f'Expected the same userbenchmark name from metrics files, \
                                                    but getting {control["name"]} and {treatment["name"]}.'
        bm_name = control["name"]
        detector = importlib.import_module(f"userbenchmark.{bm_name}.regression_detector").run

        # Process control and treatment to include only shared keys
        filtered_control_metrics = {}
        control_only_metrics = {}
        filtered_treatment_metrics = {}
        treatment_only_metrics = {}
        for control_name, control_metric in control["metrics"].items():
            if control_name in treatment["metrics"]:
                filtered_control_metrics[control_name] = control_metric
            else:
                control_only_metrics[control_name] = control_metric
        for treatment_name, treatment_metric in treatment["metrics"].items():
            if treatment_name in control["metrics"]:
                filtered_treatment_metrics[treatment_name] = treatment_metric
            else:
                treatment_only_metrics[treatment_name] = treatment_metric
        control["metrics"] = filtered_control_metrics
        treatment["metrics"] = filtered_treatment_metrics
        assert filtered_control_metrics.keys() == filtered_treatment_metrics.keys()

        # Local file comparison, return the regression detection object
        result = call_userbenchmark_detector(detector, control, treatment)
        if result or control_only_metrics or treatment_only_metrics:
            if not args.output:
                args.output = get_default_output_path(bm_name)
            # dump result to yaml file
            result_dict = asdict(result)
            result_dict["control_only_metrics"] = control_only_metrics
            result_dict["treatment_only_metrics"] = treatment_only_metrics
            # create the output directory if doesn't exist
            output_dir = Path(os.path.dirname(args.output))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_yaml_str = yaml.safe_dump(result_dict, sort_keys=False)
            with open(args.output, "w") as ofptr:
                ofptr.write(output_yaml_str)
            print(output_yaml_str)
            print(f"Wrote above yaml to {args.output}.")
        else:
            print(f"No performance signal detected between file {args.control} and {args.treatment}.")
    else:
        # S3 path
        print("Comparison for metrics from Amazon S3 is WIP.")
