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
from typing import Any, Dict, Optional
from userbenchmark.utils import PLATFORMS, USERBENCHMARK_OUTPUT_PREFIX, REPO_PATH, \
                                TorchBenchABTestResult, get_date_from_metrics, \
                                get_ub_name, get_latest_n_jsons_from_s3
from utils.s3_utils import S3Client, USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT

GITHUB_ISSUE_TEMPLATE = """
TorchBench CI has detected a performance signal or runtime regression.

Base PyTorch commit: {start}

Affected PyTorch commit: {end}

Affected Tests:
{test_details}

Tests that were no longer run on affected commit:
{control_only_tests}

Tests that were newly added on affected commit:
{treatment_only_tests}

Runtime regressions found?
{runtime_regressions_msg}

GitHub workflow that triggered this issue: {github_run_url}

cc {owner}
"""

DEFAULT_GH_ISSUE_OWNER = "@xuzhao9"

def call_userbenchmark_detector(detector, start_file: str, end_file: str) -> Optional[TorchBenchABTestResult]:
    return detector(start_file, end_file)

def get_default_output_path(bm_name: str) -> str:
    # By default, write result to $REPO_DIR/.userbenchmark/<userbenchmark-name>/regression-<time>.json
    output_path = os.path.join(REPO_PATH, USERBENCHMARK_OUTPUT_PREFIX, bm_name)
    fname = "regression-{}.yaml".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    return os.path.join(output_path, fname)

def generate_regression_dict(control, treatment) -> Dict[Any, Any]:
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
        # organize metrics into a JSON dictionary
        result_dict = asdict(result)
        result_dict["control_only_metrics"] = control_only_metrics
        result_dict["treatment_only_metrics"] = treatment_only_metrics
    return result_dict if result_dict else {}
    
def process_regressions_into_yaml(regressions_dict, output_path: str, control_file: str, treatment_file: str) -> None:
    if regressions_dict == {}:
        print(f"No performance signal detected between file {control_file} and {treatment_file}.")
        return

    # create the output directory if doesn't exist
    output_dir = Path(os.path.dirname(output_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml_str = yaml.safe_dump(regressions_dict, sort_keys=False)
    print(output_yaml_str)
    with open(output_path, "w") as ofptr:
        ofptr.write(output_yaml_str)
    print(f"Wrote above yaml to {output_path}.")
        

def process_regressions_into_gh_issue(regressions_dict, owner: str, output_path: str, errors_path: str) -> None:
    troubled_tests = ""
    for test, stats in regressions_dict["details"].items():
        delta = stats["delta"]
        if delta != 0:
            sign = "+" if delta > 0 else ""
            troubled_tests += f"- {test}: {sign}{delta:.5f}%\n"
    
    control_only_tests = ""
    for test, stat in regressions_dict["control_only_metrics"].items():
        control_only_tests += f"- {test}: {stat}\n"

    treatment_only_tests = ""
    for test, stat in regressions_dict["treatment_only_metrics"].items():
        treatment_only_tests += f"- {test}: {stat}\n"

    control_commit = regressions_dict["control_env"]["pytorch_git_version"]
    treatment_commit = regressions_dict["treatment_env"]["pytorch_git_version"]

    runtime_regressions_msg = "No runtime errors were found in the " + \
                              "new benchmarks run--you are all good there!"

    errors_log_exists = Path(errors_path).exists()
    if errors_log_exists:
        runtime_regressions_msg = "An errors log was found. Please investigate runtime " + \
                                  "errors by looking into the logs of the workflow linked."

    if troubled_tests == "" and control_only_tests == "" and treatment_only_tests == "" and not errors_log_exists:
        print(f"No regressions found between {control_commit} and {treatment_commit}.")
        return

    fname = os.environ["GITHUB_ENV"]
    content = f"TORCHBENCH_REGRESSION_DETECTED='{treatment_commit}'\n"
    with open(fname, 'a') as fo:
        fo.write(content)

    github_run_id = os.environ.get("GITHUB_RUN_ID", None)
    github_run_url = "No URL found, please look for the failing action in " + \
                     "https://github.com/pytorch/benchmark/actions"
    if github_run_id is not None:
        github_run_url = f"https://github.com/pytorch/benchmark/actions/runs/{github_run_id}"

    issue_config: Dict[str, str] = {
        "start": control_commit,
        "end": treatment_commit,
        "test_details": troubled_tests,
        "control_only_tests": control_only_tests,
        "treatment_only_tests": treatment_only_tests,
        "runtime_regressions_msg": runtime_regressions_msg,
        "github_run_url": github_run_url,
        "owner": owner
    }
    
    issue_body = GITHUB_ISSUE_TEMPLATE.format(**issue_config)
    print(issue_body)
    with open(output_path, "w") as f:
        f.write(issue_body)


def get_best_start_date(latest_metrics_jsons, end_date: str):
    """Get the date closest to `end_date` from `latest_metrics_jsons`"""
    pass


def get_metrics_by_date(date_str: str):
    s3 = S3Client(USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT)
    metrics_json_file = s3.get_file_as_json(latest_metrics_jsons[date_str])
    with open(metrics_json_file, "r") as metrics_fp:
        return json.load(metrics_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Local metrics file comparison
    parser.add_argument("--control", default=None, help="The control group metrics file for comparison. "
                        "If unprovided, will attempt to download and compare the previous JSON from S3 "
                        "within the past week. The platform flag must be specified in this case.")
    parser.add_argument("--treatment", default=None, help="The treatment metrics file for comparison.")
    # S3 metrics file comparison
    parser.add_argument("--name", help="Name of the userbenchmark to detect regression.")
    parser.add_argument("--platform", choices=PLATFORMS, default=None, help="The name of platform of the regression.")
    parser.add_argument("--start-date", default=None, help="The start date to detect regression.")
    parser.add_argument("--end-date", default=None, help="The latest date to detect regression.")
    # output file path
    parser.add_argument("--output", default=None, help="Output path to print the regression detection file.")
    # GitHub issue details
    parser.add_argument("--owner", nargs="*", default=["@xuzhao9"], help="Owner(s) to cc on regression issues, e.g., @janeyx99.")
    parser.add_argument("--gh-issue-path", default="gh-issue.md", help="Output path to print the issue body")
    parser.add_argument("--errors-path", default="errors.txt",
                        help="Path to errors log generated by the benchmarks run. " +
                             "Its existence ONLY is used to detect whether runtime regressions occurred.")
    args = parser.parse_args()

    # User provided both control and treatment files
    if args.control and args.treatment:
        with open(args.control, "r") as cfptr:
            control = json.load(cfptr)
        with open(args.treatment, "r") as tfptr:
            treatment = json.load(tfptr)
        output_path = args.output if args.output else get_default_output_path(control["name"])
        regressions_dict = generate_regression_dict(control, treatment)
        process_regressions_into_yaml(regressions_dict, output_path, args.control, args.treatment)
        exit(0)

    # Query S3 to get control and treatment json files
    if not args.platform:
        raise ValueError("A platform must be specified with the --platform flag to retrieve the "
                         "previous metrics JSONs as control from S3.")
    # User only provide the treatement file, and expect us to download from S3
    control, treatment = None, None
    if not args.control and args.treatment:
        json_path = Path(args.treatment)
        assert json_path.exists(), f"Specified result json path {args.treatment} does not exist."
        end_date: str = get_date_from_metrics(json_path.stem)
        userbenchmark_name: str = get_ub_name(args.treatment)
        with open(json_path, "r") as cfptr:
            treatment = json.load(cfptr)
    else:
        assert args.name, f"To detect regression with S3, you must specify a userbenchmark name."
        assert args.end_date and (end_date := datetime.strptime(args.end_date, "%YY%mm%dd")), f"To detect regression with dates, you must specify an end date in format YYmmdd."
        userbenchmark_name = args.name
        treatment = None
    latest_metrics_jsons = get_latest_n_jsons_from_s3(7, userbenchmark_name, args.platform, end_date)
    # Download control from S3
    if len(latest_metrics_jsons) == 0:
        raise RuntimeWarning(f"No previous JSONS in a week found to compare against the end date {end_date}. No regression info has been generated.")
    start_date = args.start_date if args.start_date else get_best_start_date(latest_metrics_jsons, end_date)
    if not start_date:
        raise RuntimeWarning(f"No start date in previous JSONS found to compare against the end date {end_date}. No regression info has been generated.")

    print(f"[TorchBench Regression Detector] Detecting regression of {userbenchmark_name} on platform {args.platform}, start date: {start_date}, end date: {end_date}.")
    control = get_metrics_by_date(start_date) if not control else control
    treatment = get_metrics_by_date(end_date) if not treatment else treatment
    regressions_dict = generate_regression_dict(control, treatment)
    output_path = args.output if args.output else get_default_output_path(control["name"])
    process_regressions_into_yaml(regressions_dict, output_path, args.control, args.treatment)

    owner = " ".join(args.owner) if args.owner else DEFAULT_GH_ISSUE_OWNER
    process_regressions_into_gh_issue(regressions_dict, owner, args.gh_issue_path, args.errors_path)
