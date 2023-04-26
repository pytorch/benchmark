import os
import sys
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


with add_path(str(REPO_PATH)):
    from utils.s3_utils import S3Client, USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT


@dataclass
class TorchBenchABTestMetric:
    control: float
    treatment: float
    delta: float


@dataclass
class TorchBenchABTestResult:
    control_env: Dict[str, str]
    treatment_env: Dict[str, str]
    bisection: Optional[str]
    details: Dict[str, TorchBenchABTestMetric]


def get_output_json(bm_name, metrics) -> Dict[str, Any]:
    import torch
    return {
        "name": bm_name,
        "environ": {"pytorch_git_version": torch.version.git_version},
        "metrics": metrics,
    }


def dump_output(bm_name, output, target_dir: Path=None) -> None:
    if target_dir is None:
        target_dir = get_output_dir(bm_name)
    fname = "metrics-{}.json".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    full_fname = os.path.join(target_dir, fname)
    with open(full_fname, 'w') as f:
        json.dump(output, f, indent=4)


def get_date_from_metrics(metrics_file: str) -> str:
    datetime_obj = datetime.strptime(metrics_file, "metrics-%Y%m%d%H%M%S")
    return datetime.strftime(datetime_obj, "%Y-%m-%d")


def get_ub_name(metrics_file_path: str) -> str:
    with open(metrics_file_path, "r") as mf:
        metrics = json.load(mf)
    return metrics["name"]


def get_output_dir(bm_name) -> Path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    target_dir = current_dir.parent.joinpath(USERBENCHMARK_OUTPUT_PREFIX, bm_name)
    target_dir.mkdir(exist_ok=True, parents=True)
    return target_dir


def get_latest_n_jsons_from_s3(n: int, bm_name: str, platform_name: str, date: str) -> List[str]:
    """Retrieves the most recent n metrics json filenames from S3 the WEEK BEFORE the given date, exclusive of that date.
       If fewer than n items are found, returns all found items without erroring, even if there were no items. """
    s3 = S3Client(USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT)
    directory = f'{bm_name}/{platform_name}'

    if not s3.exists(None, directory):
        return []

    previous_json_files = []
    start_date = datetime.strptime(date, '%Y-%m-%d')
    current_date = start_date - timedelta(days=1)
    while len(previous_json_files) < n and current_date >= start_date - timedelta(days=7):
        current_date_str = current_date.strftime('%Y-%m-%d')
        current_directory = f'{directory}/{current_date_str}'

        if s3.exists(None, current_directory):
            files = s3.list_directory(current_directory)
            metric_jsons = [f for f in files if f.endswith('.json') and 'metrics' in f]
            metric_jsons.sort(key=lambda x: datetime.strptime(x.split('/')[-1].split('-')[-1].split('.')[0], '%Y%m%d%H%M%S'), reverse=True)
            previous_json_files.extend(metric_jsons[:n - len(previous_json_files)])

        # Move on to the previous date.
        current_date -= timedelta(days=1)

    return previous_json_files
