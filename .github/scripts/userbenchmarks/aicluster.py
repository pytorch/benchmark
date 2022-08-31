import boto3
import os
import botocore

AICLUSTER_S3_BUCKET = "ossci-metrics"
AICLUSTER_S3_OBJECT = "torchbench-aicluster-metrics"

class S3Client:
    def __init__(self, bucket=AICLUSTER_S3_BUCKET, object=AICLUSTER_S3_OBJECT):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.object = object

    def download_file(self, file_name, dest_path):
        with open(os.path.join(dest_path, file_name), 'wb') as f:
            self.s3.download_fileobj(self.bucket, self.object, f)

    def upload_file(self, file_name):
        response = self.s3.upload_file(file_name, self.bucket, self.object)
        print(f"S3 client response: {response}")
    
    def exists(self, file_name):
        try:
            self.s3.Object(self.bucket, f"{self.object}/{file_name}").load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            else:
                raise RuntimeError(f"Error when checking s3 file {file_name} exist: {e}")
        return True

def run_aicluster_benchmark(benchmark_name: str, dryrun=False):
    print(f"Running benchmark on aicluster... {benchmark_name}")
    pass
