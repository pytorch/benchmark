import os
import sys
import yaml
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent
USERBENCHMARK_OUTPUT_PREFIX = ".userbenchmark"

PLATFORMS = [
    "gcp_a100",
    "aws_t4_metal",
]


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



@dataclass
class TorchBenchABTestMetric:
    control: Union[float, str]
    treatment: Union[float, str]
    delta: Union[float, str]

@dataclass
class TorchBenchABTestResult:
    name: str
    control_env: Dict[str, str]
    treatment_env: Dict[str, str]
    details: Dict[str, TorchBenchABTestMetric]
    control_only_metrics: Dict[str, float] = field(default_factory=dict)
    treatment_only_metrics: Dict[str, float] = field(default_factory=dict)
    # the repository to bisect, default to "pytorch"
    bisection: str = "pytorch"
    # can be "abtest" or "bisect"
    bisection_mode: str = "bisect"
    # the regression-*.yaml file path that generates this object
    bisection_config_file_path: Optional[str] = None


def parse_abtest_result_from_regression_file_for_bisect(regression_file: str):
    def _parse_dict_to_abtestmetric(details_dict) -> Dict[str, TorchBenchABTestMetric]:
        r = {}
        for name in details_dict:
            r[name] = TorchBenchABTestMetric(control=details_dict[name]["control"],
                                             treatment=details_dict[name]["treatment"],
                                             delta=details_dict[name]["delta"])
        return r
    with open(regression_file, "r") as rf:
        regression_dict = yaml.safe_load(rf)
    return TorchBenchABTestResult(
        name=regression_dict["name"],
        control_env=regression_dict["control_env"],
        treatment_env=regression_dict["treatment_env"],
        details=_parse_dict_to_abtestmetric(regression_dict["details"]),
        control_only_metrics=regression_dict["control_only_metrics"],
        treatment_only_metrics=regression_dict["treatment_only_metrics"],
        bisection=regression_dict["bisection"],
        bisection_mode=regression_dict["bisection_mode"],
        bisection_config_file_path=regression_file if not regression_dict["bisection_config_file_path"] \
                                    else regression_dict["bisection_config_file_path"],
    )


def get_output_json(bm_name, metrics) -> Dict[str, Any]:
    import torch
    return {
        "name": bm_name,
        "environ": {
            "pytorch_git_version": getattr(torch.version, "git_version", "<unknown>"),
            "pytorch_version": torch.__version__,
        },
        "metrics": metrics,
    }


def get_output_dir(bm_name: str) -> Path:
    import torch
    IS_FBCODE = False if hasattr(torch.version, "git_version") else True
    if not IS_FBCODE:
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        target_dir = current_dir.parent.joinpath(USERBENCHMARK_OUTPUT_PREFIX, bm_name)
    else:
        target_dir = Path(f"/tmp/{bm_name}")
    return target_dir


def get_default_output_json_path(bm_name: str, target_dir: Path=None) -> str:
    if target_dir is None:
        target_dir = get_output_dir(bm_name)
    target_dir.mkdir(exist_ok=True, parents=True)
    fname = "metrics-{}.json".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    full_fname = os.path.join(target_dir, fname)
    return full_fname

def get_default_debug_output_dir(metrics_json: str) -> Path:
    metrics_json_path = Path(metrics_json)
    metrics_json_dir = metrics_json_path.parent
    metrics_datetime = datetime.strptime(metrics_json_path.name, "metrics-%Y%m%d%H%M%S.json")
    debug_output_dir = metrics_json_dir.joinpath("output-" + datetime.strftime(metrics_datetime, "%Y%m%d%H%M%S"))
    return debug_output_dir

def dump_output(bm_name: str, output: Any, target_dir: Path=None) -> None:
    full_fname = get_default_output_json_path(bm_name, target_dir=target_dir)
    with open(full_fname, 'w') as f:
        json.dump(output, f, indent=4)


def get_date_from_metrics(metrics_file: str) -> str:
    assert metrics_file.startswith("metrics-") or metrics_file.startswith("regression-"), f"Unknown metrics or regression file name format: {metrics_file}"
    # metrics_file usually looks like metrics-%Y%m%d%H%M%S or regression-%Y%m%d%H%M%S
    stripped_filename = metrics_file.split("-")[1]
    datetime_obj = datetime.strptime(stripped_filename, "%Y%m%d%H%M%S")
    return datetime.strftime(datetime_obj, "%Y-%m-%d")


def get_ub_name(metrics_file_path: str) -> str:
    if metrics_file_path.endswith(".json"):
        with open(metrics_file_path, "r") as mf:
            metrics = json.load(mf)
        return metrics["name"]
    elif metrics_file_path.endswith(".yaml"):
        with open(metrics_file_path, "r") as mf:
            regression = yaml.safe_load(mf)
        return regression["name"]
    print(f"Unknown metrics or regression file name path: {metrics_file_path}")
    exit(1)


def get_date_from_metrics_s3_key(metrics_s3_key: str) -> datetime:
    metrics_s3_json_filename = metrics_s3_key.split('/')[-1]
    return datetime.strptime(metrics_s3_json_filename, 'metrics-%Y%m%d%H%M%S.json') if metrics_s3_key.endswith('.json') \
        else datetime.strptime(metrics_s3_json_filename, 'regression-%Y%m%d%H%M%S.yaml')


def get_latest_files_in_s3_from_last_n_days(bm_name: str, platform_name: str, date: datetime, cond: Callable, ndays: int=7, limit: int=100) -> List[str]:
    """Retrieves the most recent n day metrics json filenames from S3 before the given date, inclusive of that date.
       If fewer than n days are found, returns all found items without erroring, even if there were no items.
       Returns maximum 100 results by default. """
    with add_path(str(REPO_PATH)):
        from utils.s3_utils import S3Client, USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT
    s3 = S3Client(USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT)
    directory = f'{bm_name}/{platform_name}'

    if not s3.exists(None, directory):
        return []

    previous_json_files = []
    current_date = date
    while len(previous_json_files) < limit and current_date >= date - timedelta(days=ndays):
        current_date_str = current_date.strftime('%Y-%m-%d')
        current_directory = f'{directory}/{current_date_str}'

        if s3.exists(None, current_directory):
            files = s3.list_directory(current_directory)
            metric_jsons = [f for f in files if cond(f)]
            metric_jsons.sort(key=lambda x: get_date_from_metrics_s3_key(x), reverse=True)
            previous_json_files.extend(metric_jsons[:limit - len(previous_json_files)])

        # Move on to the previous date.
        current_date -= timedelta(days=1)

    return previous_json_files
