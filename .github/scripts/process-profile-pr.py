import os
import re
import tabulate
import argparse

MAGIC_PREFIX = "PROFILE_MODEL: "

def _parse_pr_body(body):
    magic_lines = list(filter(lambda x: MAGIC_PREFIX in x, body.splitlines()))
    if len(magic_lines):
        print(magic_lines[0][len(MAGIC_PREFIX):].strip())

def _parse_batch_test_log(log, csv):
    batch_test_result = []
    regex_keys = ["bs", "gpu", "cpu_dispatch", "cpu_total"]
    regex_dict = {
        "bs": re.compile('batch test, bs=([0-9]+)'),
        "gpu": re.compile('GPU Time:\s*([0-9.]+) milliseconds'),
        "cpu_dispatch": re.compile('CPU Dispatch Time:\s*([0-9.]+) milliseconds'),
        "cpu_total": re.compile('CPU Total Wall Time:\s*([0-9.]+) milliseconds')
    }
    for line in log.splitlines():
       matches = list(map(lambda x: None if not regex_dict[x].search(line) else regex_dict[x].search(line).groups(), regex_keys))
       for x in range(len(matches)):
           if matches[x]:
               if x == 0:
                   batch_test_result.append({})
               batch_test_result[-1][regex_keys[x]] = float(matches[x][0])
    print(_visualize_batch_test_result(regex_keys, batch_test_result, csv))

def _visualize_batch_test_result(keys, result, csv):
    output = [["Batch Size", "GPU Time", "CPU Dispatch Time", "Walltime", "GPU Delta"]]
    for index, res in enumerate(result):
        r = []
        for k in keys:
            r.append(str(res[k]))
        delta = '-' if index == 0 else str((res["gpu"] - result[index-1]["gpu"]) / result[index-1]["gpu"])
        r.append(delta)
        output.append(r)
    if not csv:
        return tabulate.tabulate(output, headers='firstrow')
    else:
        return "\n".join(map(lambda x: ",".join(x), output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-body", type=argparse.FileType("r"))
    parser.add_argument("--log", type=argparse.FileType("r"))
    parser.add_argument("--csv", action='store_true')
    args = parser.parse_args()
    if args.pr_body:
        body = args.pr_body.read()
        _parse_pr_body(body)
    if args.log:
        log = args.log.read()
        _parse_batch_test_log(log, args.csv)
