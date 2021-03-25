#!/usr/bin/env python
"""
The script to upload TorchBench nightly CI result to Amazon S3.
It assumes the following hierarchy of the result directory:

benchmark-results/
 |-result-directory-1
   |-result1.json
 |-result-directory-2
   |-result2.json

The command
`upload_s3.py --torchbench-result-dir benchmark-results/result-directory --gen-index --upload-s3`
will index all directories under `benchmark-results` and generate the `index.json` file.
Then it will upload the `result-directory` and `index.json` to Amazon S3 bucket.
"""

import argparse
import json
import re
import os
import boto3
from json import JSONEncoder
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

@dataclass
class TorchBenchData:
    torch_date: str
    relpath: str

class TBEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

S3_BUCKET = "ossci-metrics"
COMMIT_HISTORY: Dict[str, TorchBenchData] = dict()

def get_S3_object_from_bucket(bucket_name: str, object: str) -> Any:
    s3 = boto3.resource('s3')
    return s3.Object(bucket_name, object)

def analyze_json_result(json_file: str, relpath: str) -> (str, Optional[TorchBenchData]):
    with open(json_file, "r") as jfile:
        data = json.load(jfile)
    pytorch_version = data["machine_info"]["pytorch_version"]
    pytorch_date_result = re.search("dev([0-9]{8})", pytorch_version)
    if not pytorch_date_result:
        print(f"Found invalid pytorch nightly version {pytorch_version} in {json_file}, skip!")
        return ("", None)
    else:
        pytorch_date = pytorch_date_result.groups()[0]
        data = TorchBenchData(
            torch_date = pytorch_date,
            relpath = relpath)
        return (pytorch_date, data)

def gen_commit_history(result_dir: str):
    dirs = map(lambda x: os.path.join(result_dir, x), sorted(os.listdir(result_dir)))
    dirs = filter(lambda x: os.path.isdir(x), dirs)
    results = dict()
    for d in dirs:
        json_files = map(lambda x: os.path.join(d, x), sorted(os.listdir(d)))
        # Don't analyze empty files
        filtered_json_files = filter(lambda x: os.stat(x).st_size, json_files)
        for json_file in filtered_json_files:
            relpath = os.path.relpath(json_file, result_dir)
            (torch_date, tb_data) = analyze_json_result(json_file, relpath)
            if torch_date:
                # Only store the latest data
                results[torch_date] = tb_data
    out = []
    for key in sorted(results.keys()):
        test = dict()
        test["id"] = key
        test["result"] = results[key]
        out.append(test)
    return out

def upload_s3_file(key, body_file):
    with open(body_file, "r") as body:
        print(f"Uploading file {body_file} to S3 key: {key}")
        obj = get_S3_object_from_bucket(S3_BUCKET, key)
        obj.put(Body=body.read())

def is_nonempty_json(json_path):
    # Upload non-empty json files
    return (json_path.endswith(".json") and os.path.exists(json_path) and os.stat(json_path).st_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchbench-result-dir", required=True,
                    help="Specify the benchmark result directory")
    parser.add_argument("--gen-index", action="store_true",
                    help="Regenerate the benchmark index json")
    parser.add_argument("--upload-s3", action="store_true",
                    help="Upload the specified directory to Amazon S3")
    args = parser.parse_args()
    parent_dir = Path(args.torchbench_result_dir).parent.absolute()
    index_path = os.path.join(parent_dir, "index.json")
    # Generate the index.json file to the parent directory
    if args.gen_index:
        # Get parent directory 
        index_json = gen_commit_history(parent_dir)
        with open(index_path, "w") as out_file:
            out_file.write(json.dumps(index_json, indent=4, sort_keys=True, cls=TBEncoder))
    if args.upload_s3:
        # Upload index file
        index_key = "torchbench_v0_nightly/index.json"
        upload_s3_file(index_key, index_path)
        # Upload the result directory
        basedir = os.path.basename(args.torchbench_result_dir)
        result_dir = f"torchbench_v0_nightly/{basedir}/"
        for result_file in os.listdir(args.torchbench_result_dir):
            result_key = f"{result_dir}/{result_file}"
            result_path = os.path.join(args.torchbench_result_dir, result_file)
            if is_nonempty_json(result_path):
                upload_s3_file(result_key, result_path)
