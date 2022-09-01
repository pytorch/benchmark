"""
The script to upload TorchBench nightly CI result to Amazon S3.
It assumes the following hierarchy of the result directory:
torchbench-aicluster-metrics/
 |-distributed
   |-metrics-20220805192500.json
"""
import boto3
import os
import argparse
import botocore
from pathlib import Path

AICLUSTER_S3_BUCKET = "ossci-metrics"
AICLUSTER_S3_OBJECT = "torchbench-aicluster-metrics"
REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

class S3Client:
    def __init__(self, bucket=AICLUSTER_S3_BUCKET, object=AICLUSTER_S3_OBJECT):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.object = object

    def get_s3_path(self, prefix, file_name, obj_path=False):
        if obj_path:
            return "/".join([self.object, prefix, file_name])
        else:
            return "/".join([prefix, file_name])

    def download_file(self, prefix, file_name, dest_path):
        s3_path = self.get_s3_path(prefix, file_name, obj_path=True)
        with open(os.path.join(dest_path, file_name), 'wb') as f:
            self.s3.download_fileobj(self.bucket, s3_path, f)

    def upload_file(self, prefix, file_name):
        s3_path = self.get_s3_path(prefix, file_name)
        response = self.s3.upload_file(s3_path, self.bucket, self.object)
        print(f"S3 client response: {response}")

    def exists(self, prefix, file_name):
        try:
            s3_path = self.get_s3_path(prefix, file_name, obj_path=True)
            self.s3.Object(self.bucket, s3_path).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            else:
                raise RuntimeError(f"Error when checking s3 file {file_name} exist: {e}")
        return True

    def list_directory(self, directory):
        """List the directory files in the S3 bucket path.
           If the directory doesn't exist, report an error. """
        pass

def get_metrics_index(s3, work_dir):
    """
    1. List the metrics files from S3 directory
    2. Try download the index file from S3, if not found, create an initial one
    3. Otherwise, compare the downloaded index file with the metrics file list, update the index file, and return
    """
    pass

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
    index = get_metrics_index(s3, work_dir)
    # upload to scribe by the index
    if upload_scribe:
        upload_scribe(s3, index, work_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="Name of the benchmark to run.")
    parser.add_argument("--upload-scribe", action="store_true", help="Update the result to Scribe.")
    args = parser.parse_args()
    run_aicluster_benchmark(benchmark_name=args.benchmark, upload_scribe=args.upload_scribe)
