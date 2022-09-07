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
import datetime
import yaml
import argparse
import subprocess
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
    from scripts.userbenchmark.upload_scribe import TorchBenchUserbenchmarkUploader, process_benchmark_json

class S3Client:
    def __init__(self, bucket=AICLUSTER_S3_BUCKET, object=AICLUSTER_S3_OBJECT):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.object = object

    def download_file(self, key, dest_dir):
        filename = S3Client.get_filename_from_key(key)
        assert filename, f"Expected non-empty filename from key {key}."
        with open(os.path.join(dest_dir, filename), 'wb') as f:
            self.s3.download_fileobj(self.bucket, key, f)

    def upload_file(self, prefix, file_path):
        file_name = file_path.name
        s3_key = f"{self.object}/{prefix}/{file_name}" if prefix else f"{self.object}/{file_name}"
        response = self.s3.upload_file(str(file_path), self.bucket, s3_key)
        print(f"S3 client response: {response}")

    def exists(self, prefix, file_name):
        """Test if the key object/prefix/file_name exists in the S3 bucket.
           If True, return the S3 object key. Return None otherwise. """
        s3_key = f"{self.object}/{prefix}/{file_name}" if prefix else f"{self.object}/{file_name}"
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

def determine_success_today(index, allow_yesterday=True):
    """
    Determine whether today or yesterday's run is successful.
    """
    # get today's date in UTC
    today = datetime.datetime.utcnow().date()
    today_str = f"metrics-{today.strftime('%Y%m%d')}"
    yesterday = (today - datetime.timedelta(days=1))
    yesterday_str = f"metrics-{yesterday.strftime('%Y%m%d')}"
    for index_key in index:
        # check if today or yesterday's date exists in the index
        if today_str in index_key:
            print(f"Found today run log: {index_key} ")
            return True
        if allow_yesterday and yesterday_str in index_key:
            print(f"Found yesterday run log: {index_key} ")
            return True
    # not found, the last run probably failed
    return False

def get_metrics_index(s3, benchmark_name, work_dir):
    """
    1. Try to download the index file from S3, 
    2. if not found, create an initial one with the metrics files from S3 directory
    3. Otherwise, compare the downloaded index file with the metrics file list, update the index file, and return
    """
    def gen_index_obj(index_key):
        "download and load the index file if exists, otherwise, return empty object."
        if not index_key:
            return {}
        filename = S3Client.get_filename_from_key(index_key)
        s3.download_file(index_key, work_dir)
        with open(work_dir.joinpath(filename), "r") as index_f:
            index = yaml.safe_load(index_f)
        return index
    def filter_metric_files(metric_files):
        filtered_metrics = list(filter(lambda x: S3Client.get_filename_from_key(x) \
                                and S3Client.get_filename_from_key(x).startswith("metrics-") \
                                and x.endswith(".json"), \
                                s3.list_directory(directory=None)))
        return filtered_metrics
    def update_index_from_metrics(index, metric_files):
        metric_filenames = list(map(lambda x: S3Client.get_filename_from_key(x), metric_files))
        for metric_filename in metric_filenames:
            if not metric_filename in index:
                index[metric_filename] = {}
                index[metric_filename]["uploaded-scribe"] = False
        return index
    index_key = s3.exists(prefix=benchmark_name, file_name=INDEX_FILE_NAME)
    index = gen_index_obj(index_key)
    metric_files = filter_metric_files(s3.list_directory(directory=None))
    updated_index = update_index_from_metrics(index, metric_files)
    return updated_index

def upload_metrics_to_scribe(s3, benchmark_name, index, work_dir):
    """
    for each 'uploaded-scrbe: False' file in index
      1. download it from S3
      2. upload it to scribe
      3. if success, update the index file with 'uploaded-scribe: True'
    upload the updated index file to S3 after processing all files
    """
    try:
        for index_key in index:
            assert "uploaded-scribe" in index[index_key], \
                f"Index key {index_key} missing field uploaded-scribe!"
        index_file_path = work_dir.joinpath(INDEX_FILE_NAME)
        with open(index_file_path, "w") as index_file:
            yaml.safe_dump(index, index_file)
        need_upload_metrics = filter(lambda x: not index[x]["uploaded-scribe"], index.keys())
        for upload_metrics in need_upload_metrics:
            # download the metrics file from S3 to work_dir
            print(f"Downloading metrics file {upload_metrics} to local.")
            metrics_key = s3.exists(prefix=None, file_name=upload_metrics)
            assert metrics_key, f"Expected metrics file {upload_metrics} does not exist."
            s3.download_file(metrics_key, work_dir)
            # upload it to scribe
            print(f"Uploading metrics file {upload_metrics} to scribe.")
            metrics_path = str(work_dir.joinpath(upload_metrics).resolve())
            with open(metrics_path, "r") as mp:
                benchmark_time, benchmark_data = process_benchmark_json(mp)
            uploader = TorchBenchUserbenchmarkUploader()
            # user who run the benchmark on ai cluster
            uploader.UNIX_USER = "diegosarina"
            uploader.SUBMISSION_GROUP_GUID = "ai-cluster"
            uploader.post_userbenchmark_results(benchmark_time, benchmark_data)
            # update the index file
            index[upload_metrics]["uploaded-scribe"] = True
            with open(index_file_path, "w") as index_file:
                yaml.safe_dump(index, index_file)
    except subprocess.SubprocessError:
        print(f"Failed to upload the file to scribe.")
    finally:
        # upload the result index file to S3
        s3.upload_file(prefix=benchmark_name, file_path=index_file_path)

def get_work_dir(benchmark_name):
    workdir = Path(REPO_ROOT).joinpath(".userbenchmark").joinpath(benchmark_name).joinpath("logs")
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir

def run_aicluster_benchmark(benchmark_name: str, check_success=True, upload_scribe=True):
    work_dir = get_work_dir(benchmark_name)
    print(f"Running benchmark {benchmark_name} on aicluster, work directory: {work_dir}")
    s3 = S3Client()
    # get the benchmark metrics index or create a new one
    index = get_metrics_index(s3, benchmark_name, work_dir)
    # if the previous run is not successful, exit immediately
    if check_success and not determine_success_today(index):
        assert False, f"Don't find the last successful run in index: { index }. Please report a bug."
    # upload to scribe by the index
    if upload_scribe:
        upload_metrics_to_scribe(s3, benchmark_name, index, work_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Name of the benchmark to run.")
    parser.add_argument("--check-success", action="store_true", help="Determine whether checking the run is successful in the last two days.")
    parser.add_argument("--upload-scribe", action="store_true", help="Update the result to Scribe.")
    args = parser.parse_args()
    run_aicluster_benchmark(benchmark_name=args.benchmark, check_success=args.check_success, upload_scribe=args.upload_scribe)
