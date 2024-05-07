import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


class add_path:
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
    from userbenchmark.utils import get_date_from_metrics, get_ub_name
    from utils.s3_utils import (
        S3Client,
        USERBENCHMARK_S3_BUCKET,
        USERBENCHMARK_S3_OBJECT,
    )


def upload_s3(ub_name: str, platform_name: str, date_str: str, file_path: Path):
    """S3 path:
    s3://ossci-metrics/torchbench_userbenchmark/<userbenchmark-name>/<platform-name>/<date>/metrics-<YYmmddHHMMSS>.json
    s3://ossci-metrics/torchbench_userbenchmark/<userbenchmark-name>/<platform-name>/<date>/regression-<YYmmddHHMMSS>.yaml
    """
    s3client = S3Client(USERBENCHMARK_S3_BUCKET, USERBENCHMARK_S3_OBJECT)
    prefix = f"{ub_name}/{platform_name}/{date_str}"
    s3client.upload_file(prefix=prefix, file_path=file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--userbenchmark_platform",
        required=True,
        help="Name of the userbenchmark platform",
    )
    parser.add_argument(
        "--upload-file",
        required=True,
        help="Upload userbenchmark json or regression yaml file.",
    )
    args = parser.parse_args()
    upload_file_path = Path(args.upload_file)
    assert (
        upload_file_path.exists()
    ), f"Specified result json path {args.upload_file} does not exist."
    date_str = get_date_from_metrics(upload_file_path.stem)
    ub_name = get_ub_name(args.upload_file)
    upload_s3(ub_name, args.userbenchmark_platform, date_str, upload_file_path)
