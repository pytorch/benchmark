"""Scribe Uploader for Pytorch Benchmark V2 Data

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

TORCHBENCH_V2_SCORE_SCHEMA = [
    'total',
    'delta',
    'cuda-train-overall',
    'cuda-train-nlp',
    'cuda-train-classification',
    'cuda-train-segmentation',
    'cuda-train-speech',
    'cuda-train-recommendation',
    'cuda-eval-overall',
    'cuda-eval-nlp',
    'cuda-eval-classification',
    'cuda-eval-segmentation',
    'cuda-eval-speech',
    'cuda-eval-recommendation',
    'cpu-train-overall',
    'cpu-train-nlp',
    'cpu-train-classification',
    'cpu-train-segmentation',
    'cpu-train-speech',
    'cpu-train-recommendation',
    'cpu-eval-overall',
    'cpu-eval-nlp',
    'cpu-eval-classification',
    'cpu-eval-segmentation',
    'cpu-eval-speech',
    'cpu-eval-recommendation',
]

def decorate_torchbench_score_schema(schema):
    return f"torchbench_score_{schema}"

def rds_submit(data):
    i, n, table, chunk = data
    try:
        from scribe import rds_write
    except ImportError:
        # If the utils haven't been grabbed from pytorch/pytorch/tools/stats/scribe.py,
        # give up
        print("Unable to import rds utilities, download them from https://github.com/pytorch/pytorch/raw/master/tools/stats/scribe.py")
        return
    rds_write(table, chunk)
    print(f"[rds] Wrote chunk {i} / {n}")


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

    def upload_rds(self, messages: list):
        """
        Upload Scribe messages to the DB behind https://metrics.pytorch.org.
        """
        try:
            from scribe import register_rds_schema
        except ImportError:
            # If the utils haven't been grabbed from pytorch/pytorch/tools/stats/scribe.py,
            # give up
            print("Unable to import rds utilities, download them from https://github.com/pytorch/pytorch/raw/master/tools/stats/scribe.py")
            return

        # Flatten schema and re-name the types into what RDS can handle
        flat_schema = {}
        scuba_name_remap = {
            "int": "int",
            "float": "float",
            "normal": "string",
        }
        for type, field_names in self.schema.items():
            for field_name in field_names:
                flat_schema[field_name] = scuba_name_remap[type]
        register_rds_schema(self.category, flat_schema)

        # Flatten each message into a key-value map and upload them
        def flatten_message(message):
            flat = {}
            for type_values in message.values():
                for field, value in type_values.items():
                    flat[field] = value
            return flat
        messages = [flatten_message(m) for m in messages]

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # messages is too large to send in one batch due to AWS lambda
        # limitations on payload size, so break it up and send it in parallel
        args = []
        for i, chunk in enumerate(chunks(messages, 100)):
            args.append((i, len(messages) / 100, self.category, chunk))

        with multiprocessing.Pool(20) as p:
            p.map(rds_submit, args)

    def upload(self, messages: list):
        self.upload_rds(messages)
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
            ]
        }
        # Append the TorchBench score schema
        self.schema['float'].extend(list(map(decorate_torchbench_score_schema, TORCHBENCH_V2_SCORE_SCHEMA)))

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
        }
        for s in TORCHBENCH_V2_SCORE_SCHEMA:
            decorated_schema = decorate_torchbench_score_schema(s)
            if s == "total" or s == "delta":
                scribe_message[decorated_schema] = score["score"][s]
            else:
                scribe_message[decorated_schema] = score["score"]["domain"][s]
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
