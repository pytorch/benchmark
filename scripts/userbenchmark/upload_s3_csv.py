import argparse
import os
import re
import sys
from datetime import datetime
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
    from utils.s3_utils import S3Client, USERBENCHMARK_S3_BUCKET


def upload_s3(
    s3_object: str,
    ub_name: str,
    workflow_run_id: str,
    workflow_run_attempt: str,
    file_path: Path,
    dryrun: bool,
):
    """S3 path:
    s3://ossci-metrics/<s3_object>/<ub_name>/<workflow_run_id>/<workflow_run_attempt>/file_name
    """
    s3client = S3Client(USERBENCHMARK_S3_BUCKET, s3_object)
    prefix = f"{ub_name}/{workflow_run_id}/{workflow_run_attempt}"
    print(f"Uploading to prefix: {prefix}")
    if not dryrun:
        s3client.upload_file(prefix=prefix, file_path=file_path)


def _get_files_to_upload(file_path: str, match_filename: str):
    filename_regex = re.compile(match_filename)
    return [
        file_name
        for file_name in os.listdir(file_path)
        if filename_regex.match(file_name)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s3-prefix",
        required=True,
        help="S3 path prefix",
    )
    parser.add_argument(
        "--userbenchmark",
        required=True,
        help="Name of the userbenchmark.",
    )
    parser.add_argument(
        "--upload-path",
        required=True,
        help="Local directory contains files to upload.",
    )
    parser.add_argument(
        "--match-filename",
        required=True,
        help="Filename regex matched to upload.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dryrun the upload",
    )
    args = parser.parse_args()

    files_to_upload = _get_files_to_upload(args.upload_path, args.match_filename)
    workflow_run_id = os.environ.get("WORKFLOW_RUN_ID", 0)
    workflow_run_attempt = os.environ.get("WORKFLOW_RUN_ATTEMPT", 0)

    for file in files_to_upload:
        file_path = Path(args.upload_path).joinpath(file)
        upload_s3(
            s3_object=args.s3_prefix,
            ub_name=args.userbenchmark,
            workflow_run_id=workflow_run_id,
            workflow_run_attempt=workflow_run_attempt,
            file_path=file_path,
            dryrun=args.dryrun,
        )
