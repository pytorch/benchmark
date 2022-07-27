"""Scribe Uploader for Pytorch Benchmark Data

Currently supports userbenchmark result json file.
"""

import argparse
import time
import json
import os
import requests
from collections import defaultdict
from datetime import datetime

def get_metrics_date_from_file(fname: str) -> str:
    bname = os.path.basename(fname)
    dt = datetime.strptime(bname, "metrics-%Y%m%d%H%M%S.json")
    return dt.strftime("%Y-%m-%d")

class ScribeUploader:
    def __init__(self, category):
        self.category = category

    def format_message(self, field_dict):
        assert 'time' in field_dict, "Missing required Scribe field 'time'"
        message = defaultdict(dict)
        for field, value in field_dict.items():
            if field in self.schema['normal']:
                message['normal'][field] = str(value)
            elif field in self.schema['int']:
                message['int'][field] = int(value)
            elif field in self.schema['float']:
                message['float'][field] = float(value)
            else:
                raise ValueError("Field {} is not currently used, "
                                 "be intentional about adding new fields".format(field))
        return message

    def upload(self, messages: list):
        access_token = os.environ.get("TORCHBENCH_USERBENCHMARK_SCRIBE_GRAPHQL_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("Can't find access token from environment variable")
        url = "https://graph.facebook.com/scribe_logs"
        r = requests.post(
            url,
            data={
                "access_token": access_token,
                "logs": json.dumps(
                    [
                        {
                            "category": self.category,
                            "message": json.dumps(message),
                            "line_escape": False,
                        }
                        for message in messages
                    ]
                ),
            },
        )
        print(r.text)
        r.raise_for_status()

class TorchBenchUserbenchmarkUploader(ScribeUploader):
    CLIENT_NAME = 'torchbench_userbenchmark_upload_scribe.py'
    UNIX_USER = 'torchbench_userbenchmark_gcp_a100_ci'
    SUBMISSION_GROUP_GUID = 'oss-ci-gcp-a100'

    def __init__(self):
        super().__init__('perfpipe_pytorch_adhoc_benchmarks')
        self.schema = {
            'int': [
                'time',                     # timestamp of upload
            ],
            # string fields
            'normal': [
                'benchmark_date',           # date of benchmark
                'client_name',              # name of upload client (logger)
                'unix_user',                # name of upload user
                'submission_group_guid',    # name of data batch (for debugging)
                'pytorch_git_version',      # pytorch version
                'metric_id',                # id of the metric (e.g., adhoc.nvfuser.nvfuser:autogen-42)
            ],
            # float perf metrics go here
            'float': [
                'metric_value'
            ]
        }

    def get_metric_name(self, bm_name, metric_name):
        return f"adhoc.{bm_name}.{metric_name}"

    def post_userbenchmark_results(self, bm_time, bm_data):
        messages = []
        bm_name = bm_data["name"]
        base_message = {
            'time': int(time.time()),
            'benchmark_date': bm_time,
            'client_name': self.CLIENT_NAME,
            'unix_user': self.UNIX_USER,
            'submission_group_guid': self.SUBMISSION_GROUP_GUID,
            'pytorch_git_version': bm_data["environ"]["pytorch_git_version"]
        }
        # construct message and upload
        for metric in bm_data["metrics"]:
            msg = base_message.copy()
            metric_name = self.get_metric_name(bm_name, metric)
            msg['metric_id'] = metric_name
            msg['metric_value'] = bm_data['metrics'][metric]
            formatted_msg = self.format_message(msg)
            messages.append(formatted_msg)
        # print(messages)
        self.upload(messages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--userbenchmark_json", required=True,
                        type=argparse.FileType('r'),
                        help='Upload userbenchmark json data')
    args = parser.parse_args()
    # Result sanity check
    json_name = os.path.basename(args.userbenchmark_json.name)
    benchmark_time = get_metrics_date_from_file(json_name)
    benchmark_data = json.load(args.userbenchmark_json)
    # use uploader
    uploader = TorchBenchUserbenchmarkUploader()
    uploader.post_userbenchmark_results(benchmark_time, benchmark_data)