"""Scribe Uploader for Pytorch Benchmark Data

Currently supports userbenchmark result json file.
"""

import argparse
import time
import multiprocessing
import json
import os
import requests
import subprocess
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
        access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN")
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
    def __init__(self, benchmark_time):
        super().__init__('perfpipe_pytorch_benchmarks')
        self.schema = {
            'int': [
                'time',
            ],
            'normal': [
                'pytorch_git_version',
            ],
            'float': [ ]
        }

    def post_userbenchmark_results(self, bm_time, bm_data):
        messages = []
        # update schema
        for metrics in bm_data["metrics"]:
            pass

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