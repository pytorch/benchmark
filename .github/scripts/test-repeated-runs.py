import os
import re
import tabulate
import argparse

MAGIC_PREFIX = "STABLE_TEST_MODEL: "
THRESHOLD = 7

def _parse_pr_body(body):
    magic_lines = list(filter(lambda x: MAGIC_PREFIX == x[:len(MAGIC_PREFIX)], body.splitlines()))
    if len(magic_lines):
        return magic_lines[-1][len(MAGIC_PREFIX):].strip()

def _parse_repeated_test_log(log, csv):
    repeated_test_result = []
    regex_keys = ["gpu", "cpu_dispatch", "cpu_total"]
    regex_dict = {
        "gpu": re.compile('GPU Time:\s*([0-9.]+) milliseconds'),
        "cpu_dispatch": re.compile('CPU Dispatch Time:\s*([0-9.]+) milliseconds'),
        "cpu_total": re.compile('CPU Total Wall Time:\s*([0-9.]+) milliseconds')
    }
    for line in log.splitlines():
       matches = list(map(lambda x: None if not regex_dict[x].search(line) else regex_dict[x].search(line).groups(), regex_keys))
       for x in range(len(matches)):
           if matches[x]:
               if x == 0:
                   repeated_test_result.append({})
               repeated_test_result[-1][regex_keys[x]] = float(matches[x][0])
    print(_visualize_repeated_test_result(regex_keys, repeated_test_result, csv))
    cpu_total_times = list(map(lambda x: x["cpu_total"], repeated_test_result))
    return cpu_total_times

def _visualize_repeated_test_result(keys, result, csv):
    output = [["Run Number", "GPU Time", "CPU Dispatch Time", "Wall Time"]]
    for index, res in enumerate(result):
        r = [index]
        for k in keys:
            r.append(str(res[k]))
        output.append(r)
    if not csv:
        return tabulate.tabulate(output, headers='firstrow')
    else:
        return "\n".join(map(lambda x: ",".join(x), output))

def _is_stable(total_times):
    return ((max(total_times) - min(total_times)) / min(total_times) * 100) <= THRESHOLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-body", type=argparse.FileType("r"))
    parser.add_argument("--log", type=argparse.FileType("r"))
    parser.add_argument("--csv", action='store_true')
    args = parser.parse_args()
    if args.pr_body:
        body = args.pr_body.read()
        model = _parse_pr_body(body)
        print(model)
    if args.log:
        log = args.log.read()
        cpu_total_times = _parse_repeated_test_log(log, args.csv)
        if not _is_stable(cpu_total_times):
            print("GPU stability test failed. Please fix the model code and re-run the test.")
            exit(1)
