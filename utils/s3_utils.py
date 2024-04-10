import json
import os
from pathlib import Path
from typing import Any, List, Optional

import boto3
import yaml

USERBENCHMARK_S3_BUCKET = "ossci-metrics"
USERBENCHMARK_S3_OBJECT = "torchbench-userbenchmark"
REPO_ROOT = Path(__file__).parent.parent


class S3Client:
    def __init__(self, bucket, object):
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.object = object

    def download_file(self, key: str, dest_dir: str) -> None:
        filename = S3Client.get_filename_from_key(key)
        assert filename, f"Expected non-empty filename from key {key}."
        with open(os.path.join(dest_dir, filename), "wb") as f:
            self.s3.download_fileobj(self.bucket, key, f)

    def upload_file(self, prefix: str, file_path: Path) -> None:
        file_name = file_path.name
        s3_key = (
            f"{self.object}/{prefix}/{file_name}"
            if prefix
            else f"{self.object}/{file_name}"
        )
        response = self.s3.upload_file(str(file_path), self.bucket, s3_key)
        print(f"S3 client response: {response}")

    def get_file_as_json(self, key: str) -> Any:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def get_file_as_yaml(self, key: str) -> Any:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return yaml.safe_load(obj["Body"].read().decode("utf-8"))

    def exists(self, prefix: str, file_name: str) -> Optional[str]:
        """Test if the key object/prefix/file_name exists in the S3 bucket.
        If True, return the S3 object key. Return None otherwise."""
        s3_key = (
            f"{self.object}/{prefix}/{file_name}"
            if prefix
            else f"{self.object}/{file_name}"
        )
        result = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=s3_key)
        if "Contents" in result:
            return s3_key
        return None

    def list_directory(self, directory=None) -> List[str]:
        """List the directory files in the S3 bucket path.
        If the directory doesn't exist, report an error."""
        prefix = f"{self.object}/{directory}/" if directory else f"{self.object}/"
        pages = self.s3.get_paginator("list_objects").paginate(
            Bucket=self.bucket, Prefix=prefix
        )
        keys = filter(
            lambda x: not x == prefix, [e["Key"] for p in pages for e in p["Contents"]]
        )
        return list(keys)

    def get_filename_from_key(object_key: str) -> str:
        filename = object_key.split("/")[-1]
        return filename


def decompress_s3_data(s3_tarball_path: Path):
    assert str(s3_tarball_path.absolute()).endswith(
        ".tar.gz"
    ), f"Expected .tar.gz file path but get {s3_tarball_path}."
    import tarfile

    data_dir = os.path.join(REPO_ROOT, "torchbenchmark", "data")
    # Hide decompressed file in .data directory so that they won't be checked in
    decompress_dir = os.path.join(data_dir, ".data")
    os.makedirs(decompress_dir, exist_ok=True)
    # Decompress tar.gz file
    directory_name = s3_tarball_path.stem
    target_directory_path = Path(os.path.join(decompress_dir, directory_name))
    # If the directory already exists, we assume it has been decompressed before
    # skip decompression in this case
    if target_directory_path.exists():
        print("OK")
        return
    print(f"decompressing input tarball: {s3_tarball_path}...", end="", flush=True)
    tar = tarfile.open(s3_tarball_path)
    tar.extractall(path=decompress_dir)
    tar.close()
    print("OK")


def checkout_s3_data(data_type: str, name: str, decompress: bool = True):
    S3_URL_BASE = "https://ossci-datasets.s3.amazonaws.com/torchbench"
    download_dir = REPO_ROOT.joinpath("torchbenchmark")
    index_file = REPO_ROOT.joinpath("torchbenchmark", "data", "index.yaml")
    import requests

    with open(index_file, "r") as ind:
        index = yaml.safe_load(ind)
    assert (
        data_type == "INPUT_TARBALLS" or data_type == "MODEL_PKLS"
    ), f"Expected data type either INPUT_TARBALLS or MODEL_PKLS, get {data_type}."
    assert (
        name in index[data_type]
    ), f"Cannot find specified file name {name} in {index_file}."
    data_file = name
    data_path_segment = (
        f"data/{data_file}" if data_type == "INPUT_TARBALLS" else f"models/{data_file}"
    )
    full_path = download_dir.joinpath(data_path_segment)
    s3_url = f"{S3_URL_BASE}/{data_path_segment}"
    # Download if the tarball file does not exist
    if not full_path.exists():
        r = requests.get(s3_url, allow_redirects=True)
        with open(str(full_path.absolute()), "wb") as output:
            print(f"Checking out {s3_url} to {full_path}")
            output.write(r.content)
    if decompress:
        decompress_s3_data(full_path)
