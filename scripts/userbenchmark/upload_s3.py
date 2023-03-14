import argparse
import json
from pathlib import Path
from datetime import datetime

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
    pass

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
