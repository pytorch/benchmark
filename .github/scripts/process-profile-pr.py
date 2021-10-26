import os
import re
import argparse

MAGIC_PREFIX = "PROFILE_MODEL: "

def _parse_pr_body(body):
    magic_lines = list(filter(lambda x: MAGIC_PREFIX in x, body.splitlines()))
    if len(magic_lines):
        print(magic_lines[0][len(MAGIC_PREFIX):].strip())

def _parse_batch_test_log(log):
    batches = []
    batch_test_result = {}
    regex_keys = ["bs", "gpu", "cpu_dispatch", "cpu_total"]
    regex_dict = {
        "bs": re.compile('batch test, bs=([0-9]+)'),
        "gpu": re.compile('GPU Time: ([0-9.]+) milliseconds'),
        "cpu_dispatch": re.compile('CPU Dispatch Time: ([0-9.]+) milliseconds'),
        "cpu_total": re.compile('CPU Total Wall Time: ([0-9.]+) milliseconds')
    }
    for line in log.splitlines():
       groups = list(map(lambda x: regex_dict[x].search(line), regex_keys))
       if len(groups[0]):
           batches.append(int(groups[0][1]))
           batch_test_result[batches[-1]] = {}
       for x in range(1, len(groups)):
           batch_test_result[batches[-1]][regex_keys[x]] = float(groups[x][1])
    print(_visualize_batch_test_result(batch_test_result))

def _visualize_batch_test_result():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-body", type=exist_file)
    parser.add_argument("--log", type=exist_file)
    args = parser.parse_arguments()
    if args.pr_body:
        with open(args.pr_body, "r") as body_file:
            body = body_file.read()
        _parse_pr_body(body)
    if args.log:
        with open(args.log, "r") as log_file:
            log = log_file.read()
        _parse_batch_test_log(log)
