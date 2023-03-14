"""
The regression detector of TorchBench Userbenchmark.
"""
import sys
import argparse
import importlib
from pathlib import Path
from typing import List, Tuple, Optional
from .utils import PLATFORMS
from .utils import add_path, REPO_PATH

with add_path(REPO_PATH):
    from utils.s3_utils import S3Client

def get_available_dates(userbenchmark: str, platform: str) -> List[str]:
    """s3://ossci-metrics/torchbench/userbenchmark/<userbenchmark-name>/<platform>/<date>/metrics-<time>.json"""
    pass


def get_start_end_dates(available_dates: List[str], start_date: str, end_date: str) -> Tuple[str, str]:
    return "", ""


def download_result_file(result_date: str) -> str:
    return ""


def call_userbenchmark_detector(detector, start_file: str, end_file: str) -> Optional[str]:
    return detector(start_file, end_file)


def upload_detection_result(userbenchmark: str, platform: str, start_date: str, end_date: str, file_path: Path):
    """s3://ossci-metrics/torchbench/userbenchmark/<userbenchmark-name>/<platform>/<end_date>/regression-detector-<start_date>-<end_date>.yaml"""
    # check the file path format is correct
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("userbenchmark", help="Name of the userbenchmark to detect regression.")
    parser.add_argument("--platform", choices=PLATFORMS, help="The name of platform of the regression.")
    parser.add_argument("--start-date", default=None, help="The start date to detect regression.")
    parser.add_argument("--end-date", default="latest", help="The latest date to detect regression.")
    args = parser.parse_args()
    detector = importlib.import_module(f"{args.userbenchmark}.regression_detector").run
    available_dates = get_available_dates(userbenchmark=args.userbenchmark)
    start_date, end_date = get_start_end_dates(available_dates, args.start_date, args.end_date)
    if not start_date or not end_date:
        # Not enough metric files to detect the regression
        sys.exit(0)
    start_file, end_file = download_result_file(start_date), download_result_file(end_date)
    result = call_userbenchmark_detector(detector, start_file, end_file)
    if not result:
        # No regression detected
        sys.exit(0)
    # detected regression, upload the result
    upload_detection_result(result)
