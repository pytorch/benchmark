"""
The script to upload TorchBench CI result from S3 to Scribe (Internal).
To run this script, users need to set two environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
It assumes the following hierarchy of the result directory:
torchbench-aicluster-metrics/
 |-distributed
   |-metrics-20220805192500.json
"""
import boto3
import sys
import os
import argparse
from pathlib import Path

AICLUSTER_S3_BUCKET = "ossci-metrics"
AICLUSTER_S3_OBJECT = "torchbench-aicluster-metrics"
INDEX_FILE_NAME = "index.yaml"

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
    pass

class S3Client:
    def __init__(self, bucket=AICLUSTER_S3_BUCKET, object=AICLUSTER_S3_OBJECT):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.object = object

    def download_file(self, key, dest_dir):
        filename = self.get_filename_from_key(key)
        assert filename, f"Expected non-empty filename from key {key}."
        with open(os.path.join(dest_dir, filename), 'wb') as f:
            self.s3.download_fileobj(self.bucket, key, f)

    def upload_file(self, key, file_path):
        response = self.s3.upload_file(file_path, self.bucket, key)
        print(f"S3 client response: {response}")

    def exists(self, prefix, file_name):
        """Test if the key object/prefix/file_name exists in the S3 bucket.
           If True, return the S3 object key. Return None otherwise. """
        s3_key = f"{self.object}/{prefix}/{file_name}"
        result = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=s3_key)
        if 'Contents' in result:
            return s3_key
        return None

    def list_directory(self, directory=None):
        """List the directory files in the S3 bucket path.
           If the directory doesn't exist, report an error. """
        prefix = f"{self.object}/{directory}/" if directory else f"{self.object}/"
        pages = self.s3.get_paginator("list_objects").paginate(Bucket=self.bucket, Prefix=prefix)
        keys = filter(lambda x: not x == prefix, [e['Key'] for p in pages for e in p['Contents']])
        return list(keys)
    
    def get_filename_from_key(object_key):
        filename = object_key.split('/')[-1]
        return filename

def determine_success_today(index):
    """
    Determine whether today's run is successful.
    """
    # get today's date in UTC
    # check if today's json exists in the json file
    # return result
    pass

def get_metrics_index(s3, benchmark_name, work_dir):
    """
    1. Try to download the index file from S3, 
    2. if not found, create an initial one with the metrics files from S3 directory
    3. Otherwise, compare the downloaded index file with the metrics file list, update the index file, and return
    """
    def gen_index_obj(s3, index_key):
        "download and load the index file if exists, otherwise, return empty object."
        if not index_key:
            return {}
        s3.download_file()
    def update_index_from_metrics():
        pass
    index_key = s3.exists(prefix=benchmark_name, file_name=INDEX_FILE_NAME)
    index_obj = gen_index_obj(index_key)
    metric_files = s3.list_directory(directory=None)
    updated_index = update_index_from_metrics(index_obj, metric_files)
    return updated_index

def upload_scribe(s3, index):
    """
    for each 'uploaded: False' file in index
      1. download it from S3
      2. upload it to scribe
      3. if success, update the index file with 'uploaded: True'
      4. upload the updated index file to S3
    """
    pass

def get_work_dir(benchmark_name):
    workdir = Path(REPO_ROOT).joinpath(".userbenchmark").joinpath(benchmark_name).joinpath("logs")
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir

def run_aicluster_benchmark(benchmark_name: str, dryrun=False, upload_scribe=True):
    work_dir = get_work_dir(benchmark_name)
    print(f"Running benchmark {benchmark_name} on aicluster, work directory: {work_dir}")
    s3 = S3Client()
    # get the benchmark metrics index or create a new one
    index = get_metrics_index(s3, benchmark_name, work_dir)
    # if today's run is not successful, exit immediately
    determine_success_today(index)
    # upload to scribe by the index
    if upload_scribe:
        upload_scribe(s3, index, work_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Name of the benchmark to run.")
    parser.add_argument("--success-today", action="store_true", help="Determine whether the run is succeeded today.")
    parser.add_argument("--upload-scribe", action="store_true", help="Update the result to Scribe.")
    args = parser.parse_args()
    run_aicluster_benchmark(benchmark_name=args.benchmark, upload_scribe=args.upload_scribe)
