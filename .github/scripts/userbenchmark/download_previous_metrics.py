from typing import List
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
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
    from utils.s3_utils import S3Client, USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT
    from userbenchmark.utils import get_date_from_metrics, get_ub_name, get_latest_n_jsons_from_s3


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--platform-name', '-p',
        required=True,
        help='The platform on which the benchmarks were run'
    )
    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Name of the metrics JSON file')
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    json_path = Path(args.file)
    assert json_path.exists(), f"Specified result json path {args.userbenchmark_json} does not exist."
    date: str = get_date_from_metrics(json_path.stem)
    userbenchmark_name: str = get_ub_name(args.file)
    latest_metrics_jsons = get_latest_n_jsons_from_s3(1, userbenchmark_name, args.platform_name, date)

    if len(latest_metrics_jsons) > 0:
        s3 = S3Client(USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT)
        s3.download_file(latest_metrics_jsons[0], '.')
        
        # print the downloaded file name
        print(S3Client.get_filename_from_key(latest_metrics_jsons[0]))
