import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
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
    from utils.s3_utils import S3Client

USERBENCHMARK_S3_BUCKET = "ossci-metrics"
USERBENCHMARK_S3_OBJECT = "torchbench-userbenchmark"

def get_date_from_metrics(metrics_file: str):
    datetime_obj = datetime.strptime(metrics_file, "metrics-%Y%m%d%H%M%S")
    return datetime.strftime(datetime_obj, "%Y-%m-%d")

def get_ub_name(metrics_file_path: str):
    with open(metrics_file_path, "r") as mf:
        metrics = json.load(mf)
    return metrics["name"]

def upload_s3(ub_name: str, platform_name: str, date_str: str, file_path: Path):
    """S3 path:
        s3://ossci-metrics/torchbench_userbenchmark/<userbenchmark-name>/<platform-name>/<date>/metrics-<time>.json"""
    s3client = S3Client(USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT)
    prefix = f"{ub_name}/{platform_name}/{date_str}"
    s3client.upload_file(prefix=prefix, file_path=file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--userbenchmark_platform", required=True,
                        help='Name of the userbenchmark platform')
    parser.add_argument("--userbenchmark_json", required=True,
                        help='Upload userbenchmark json data')
    args = parser.parse_args()
    json_path = Path(args.userbenchmark_json)
    assert json_path.exists(), f"Specified result json path {args.userbenchmark_json} does not exist."
    date_str = get_date_from_metrics(json_path.stem)
    ub_name = get_ub_name(args.userbenchmark_json)
    upload_s3(ub_name, args.userbenchmark_platform, date_str, json_path)
