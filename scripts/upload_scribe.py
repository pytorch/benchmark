"""Scribe Uploader for Pytorch Benchmark Data

Currently supports data in pytest-benchmark format but can be extended.

New fields can be added just by modifying the schema in this file, schema
checking is only here to encourage reusing existing fields and avoiding typos.
"""

import argparse
import time
import multiprocessing
import json
import os
import requests
import subprocess
from collections import defaultdict

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

    def _upload_intern(self, messages: list):
        for m in messages:
            json_str = json.dumps(m)
            cmd = ['scribe_cat', self.category, json_str]
            subprocess.run(cmd)

    def upload(self, messages: list):
        if os.environ.get('SCRIBE_INTERN'):
            return self._upload_intern(messages)
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

class PytorchBenchmarkUploader(ScribeUploader):
    def __init__(self):
        super().__init__('perfpipe_pytorch_benchmarks')
        self.schema = {
            'int': [
                'time', 'rounds',
            ],
            'normal': [
                'benchmark_group', 'benchmark_name',
                'benchmark_class', 'benchmark_time',
                'git_repo', 'git_commit_id', 'git_branch',
                'git_commit_time', 'git_dirty',
                'pytorch_version', 'python_version',
                'torchtext_version', 'torchvision_version',
                'machine_kernel', 'machine_processor', 'machine_hostname',
                'github_run_id', 'torchbench_score_version',
            ],
            'float': [
                'stddev', 'min', 'median', 'max', 'mean', 'runtime',
                'torchbench_score',
                'torchbench_score_jit_speedup',
                'torchbench_subscore_cpu_train',
                'torchbench_subscore_cpu_infer',
                'torchbench_subscore_gpu_train',
                'torchbench_subscore_gpu_infer',
            ]
        }

    def post_pytest_benchmarks(self, pytest_json, max_data_upload=100):
        machine_info = pytest_json['machine_info']
        commit_info = pytest_json['commit_info']
        upload_time = int(time.time())
        messages = []
        for b in pytest_json['benchmarks']:
            base_msg = {
                "time": upload_time,
                "benchmark_group": b['group'],
                "benchmark_name": b['name'],
                "benchmark_class": b['fullname'],
                "benchmark_time": pytest_json['datetime'],
                "git_repo": commit_info['project'],
                "git_commit_id": commit_info['id'],
                "git_branch": commit_info['branch'],
                "git_commit_time": commit_info['time'],
                "git_dirty": commit_info['dirty'],
                "pytorch_version": machine_info.get('pytorch_version', None),
                "torchtext_version": machine_info.get('torchtext_version', None),
                "torchvision_version": machine_info.get('torchvision_version', None),
                "python_version": machine_info['python_implementation_version'],
                "machine_kernel": machine_info['release'],
                "machine_processor": machine_info['processor'],
                "machine_hostname": machine_info['node'],
                "github_run_id": machine_info.get('github_run_id', None),
                "torchbench_score_version": machine_info.get('torchbench_score_version', None),
            }

            stats_msg = {"stddev": b['stats']['stddev'],
                "rounds": b['stats']['rounds'],
                "min": b['stats']['min'],
                "median": b['stats']['median'],
                "max": b['stats']['max'],
                "mean": b['stats']['mean'],
            }
            stats_msg.update(base_msg)
            messages.append(self.format_message(stats_msg))

            if 'data' in b['stats']:
                for runtime in b['stats']['data'][:max_data_upload]:
                    runtime_msg = {"runtime":  runtime}
                    runtime_msg.update(base_msg)
                    messages.append(self.format_message(runtime_msg))

        self.upload(messages)

    def post_torchbench_score(self, pytest_json, score):
        machine_info = pytest_json['machine_info']
        commit_info = pytest_json['commit_info']
        upload_time = int(time.time())
        scribe_message = {
            "time": upload_time,
            "benchmark_time": pytest_json['datetime'],
            "git_repo": commit_info['project'],
            "git_commit_id": commit_info['id'],
            "git_branch": commit_info['branch'],
            "git_commit_time": commit_info['time'],
            "git_dirty": commit_info['dirty'],
            "pytorch_version": machine_info.get('pytorch_version', None),
            "torchtext_version": machine_info.get('torchtext_version', None),
            "torchvision_version": machine_info.get('torchvision_version', None),
            "python_version": machine_info['python_implementation_version'],
            "machine_kernel": machine_info['release'],
            "machine_processor": machine_info['processor'],
            "machine_hostname": machine_info['node'],
            "github_run_id": machine_info.get('github_run_id', None),
            "torchbench_score_version": machine_info.get('torchbench_score_version', None),
            "torchbench_score": score["score"]["total"],
            "torchbench_score_jit_speedup": score["score"]["jit-speedup"],
            "torchbench_subscore_cpu_train": score["score"]["subscore-cpu-train"],
            "torchbench_subscore_cpu_infer": score["score"]["subscore-cpu-eval"],
            "torchbench_subscore_gpu_train": score["score"]["subscore-cuda-train"],
            "torchbench_subscore_gpu_infer": score["score"]["subscore-cuda-eval"],
        }
        m = self.format_message(scribe_message)
        self.upload([m])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest_bench_json", required=True,
                        type=argparse.FileType('r'),
                        help='Upload json data formatted by pytest-benchmark module')
    parser.add_argument("--torchbench_score_file", required=True,
                        type=argparse.FileType('r'),
                        help="torchbench score file to include")
    args = parser.parse_args()
    
    # Result sanity check
    json_name = os.path.basename(args.pytest_bench_json.name)
    json_score = json.load(args.torchbench_score_file)
    score_data = None
    for data in json_score:
        if os.path.basename(data["file"]) == json_name:
            score_data = data
    assert score_data, f"Can't find {json_name} score in {args.torchbench_score_file}. Stop."
    benchmark_uploader = PytorchBenchmarkUploader()
    json_data = json.load(args.pytest_bench_json)
    benchmark_uploader.post_pytest_benchmarks(json_data)
    benchmark_uploader.post_torchbench_score(json_data, score_data)
