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

def call_userbenchmark_detector(detector, start_file: str, end_file: str) -> Optional[str]:
    return detector(start_file, end_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("userbenchmark", help="Name of the userbenchmark to detect regression.")
    # Local metrics file comparison
    parser.add_argument("--control", default=None, help="The control group metrics file for comparison.")
    parser.add_argument("--treatment", default="latest", help="The treatment metrics file for comparison.")
    # S3 metrics file comparison
    parser.add_argument("--platform", choices=PLATFORMS, default=None, help="The name of platform of the regression.")
    parser.add_argument("--start-date", default=None, help="The start date to detect regression.")
    parser.add_argument("--end-date", default=None, help="The latest date to detect regression.")
    args = parser.parse_args()

    detector = importlib.import_module(f"{args.userbenchmark}.regression_detector").run

    if args.control and args.treatment:
        # Local file comparison, return the regression detection result file path
        result = call_userbenchmark_detector(detector, args.control, args.treatment)
        # Print the file content
        with open(result, "r") as res:
            print(res.read())
    else:
        # S3 path
        print("Comparison from S3 metrics is WIP.")
