"""Scribe Uploader for Pytorch Microbenchmark Data"""
import argparse
import json
import hashlib
import os
import sys
import time

import numpy as np
import torch

from scripts.upload_scribe import ScribeUploader


class PytorchMicrobenchmarkUploader(ScribeUploader):
    def __init__(self):
        super().__init__('perfpipe_pytorch_microbenchmarks')
        self.schema = {
            'int': [
                'time',
                'is_debug_entry',
                'start_time',
                'end_time',
                'task_number',

                # # This is the version of the instruction count microbenchmark.
                # 'benchmark_version',

                # We don't need to record as many summary statistics for counts
                # as we do for times, because we retain them all in
                # `counts_json`.
                'count_min',
                # 'count_max',
                # 'count_p25', 'count_median', 'count_p75',
            ],
            'normal': [
                # # Unique ID for the run.
                # 'run_id',

                # # Metadata for ad-hoc analysis.
                # 'task_key',
                # 'counts_json',

                # # TODO: More environment info.
                # 'pytorch_version',
                # 'pytorch_git_version',
                # 'cuda_version',
                # 'python_version',
                # 'benchmark_name',
            ],
            'float': [
                # 't_min', 't_max', 't_mean',
                # 't_p01', 't_p25', 't_median', 't_p75', 't_p99',
                # 't_stddev',
            ]
        }

    def post_benchmarks(self, result_json):
        md5 = hashlib.md5()
        for key in result_json["values"].keys():
            md5.update(key.encode("utf-8"))
        assert result_json['md5'] == md5.hexdigest(), \
            f"Data reports version MD5 of {result_json['md5']}, however computed {md5.hexdigest()}"

        base_msg = {
            'time': int(time.time()),
            'is_debug_entry': 1,
            # 'run_id': os.getenv('GITHUB_RUN_ID'),
            'start_time': result_json['start_time'],
            'end_time': result_json['end_time'],
            # 'benchmark_version': result_json['version'],
            # 'pytorch_version': torch.__version__,
            # 'pytorch_git_version': torch.version.git_version,
            # 'cuda_version': torch.version.cuda,
            # 'python_version': '.'.join([str(i) for i in sys.version_info[:3]]),
            # 'benchmark_name': 'instruction_count',
        }

        messages = []
        for task_number, (key, measurements) in enumerate(result_json['values'].items()):
            times = measurements["times"]
            counts = measurements["counts"]
            stats_msg = {
                'task_number': task_number,
                # 'task_key': key,

                'count_min': min(counts),
                # 'count_max': max(counts),
                # 'count_p25': int(np.percentile(counts, 25)),
                # 'count_median': int(np.median(counts)),
                # 'count_p75': int(np.percentile(counts, 75)),
                # 'counts_json': json.dumps(counts),

                # 't_min': min(times),
                # 't_max': max(times),
                # 't_mean': float(np.mean(times)),
                # 't_p01': float(np.percentile(times, 1)),
                # 't_p25': float(np.percentile(times, 25)),
                # 't_median': float(np.median(times)),
                # 't_p75': float(np.percentile(times, 75)),
                # 't_p99': float(np.percentile(times, 99)),
                # 't_stddev': float(np.std(times)),
            }

            stats_msg.update(base_msg)
            messages.append(stats_msg)
        self.upload(messages[:2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--result_json', required=True,
        type=argparse.FileType('r'),
        help='Upload json data from instruction count microbenchmarks.'
    )
    args = parser.parse_args()

    json_data = json.load(args.result_json)
    benchmark_uploader = PytorchMicrobenchmarkUploader()
    benchmark_uploader.post_benchmarks(json_data)
