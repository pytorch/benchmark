import json
import os
import re
from pathlib import Path
import argparse

ATTRIBUTES = ["batch_size", "precision"]

def get_nonempty_json(d):
    r = []
    for f in filter(lambda x: x.endswith(".json"), os.listdir(d)):
        fullpath = os.path.join(d, f)
        if os.stat(fullpath).st_size:
            r.append(fullpath)
    return r

def process_json(result, f, base_key):
    with open(f, "r") as jf:
        tbo = json.load(jf)
    key = Path(f).stem
    for test in tbo:
        k = (test["name"], test["test"], test["device"])
        status = test["status"]
        if k not in result:
            result[k] = {}
        if key == base_key:
            result[k]["precision"] = test["precision"]
            result[k]["batch_size"] = test["batch_size"]
        result[k][key] = {}
        result[k][key]["status"] = status
        result[k][key]["results"] = test["results"]

def insert_if_nonexist(arr, k, loc=None):
    if k in arr:
        return
    if loc == None:
        arr.append(k)
        return
    arr.insert(loc, k)

# Result header
# Model(<test>-<device>); <base arg>; <arg1>; <arg2>; ...; <argn>
def generate_header(result, base_key):
    header = []
    args = []
    test = list(result.keys())[0][1]
    device = list(result.keys())[0][2]
    base_arg = None
    for t in result:
        assert t[1] == test, f"Both {t[1]} and {test} exist in result, can't analyze."
        assert t[2] == device, f"Both {t[2]} and {device} exist in result, can't analyze."
        result_keys = result[t].keys()
        for k in filter(lambda x: not x in ATTRIBUTES, result_keys):
            if k == base_key:
                insert_if_nonexist(args, f"{k} (latency)", loc=0)
            else:
                insert_if_nonexist(args, f"{k} (correctness)")
                insert_if_nonexist(args, f"{k} (latency)")
                insert_if_nonexist(args, f"{k} (speedup)")
                if "blade" in k:
                    # count torchdynamo subgraphs
                    if k == "torchdynamo-blade_optimize_dynamo":
                        insert_if_nonexist(args, f"{k} (subgraphs)")
                    # count blade clusters
                    insert_if_nonexist(args, f"{k} (clusters)")
                    # count blade compiled nodes
                    insert_if_nonexist(args, f"{k} (compiled)")

    header.append(f"Model({test}-{device})")
    header.append(f"precision")
    header.append(f"batch size")
    header.extend(args)
    return header

def split_header(header):
    regex = "(.*) \(([a-z]+)\)"
    g = re.match(regex, header).groups()
    return (g[0], g[1])

def is_ok(r):
    return r["status"] == "OK"

def find_result_by_header(r, header, base_arg):
    # tp: correct, latency, or speedup
    args, tp = header
    if tp == "correctness":
        if is_ok(r[args]) and "correctness" in r[args]["results"]:
            return r[args]["results"]["correctness"]
        else:
            return "N/A"
    elif tp == "latency":
        if is_ok(r[args]):
            return round(r[args]["results"]["latency_ms"], 3)
        else:
            return r[args]["status"]
    elif tp == "speedup":
        if is_ok(r[base_arg]) and is_ok(r[args]):
            return round(r[base_arg]["results"]["latency_ms"] / r[args]["results"]["latency_ms"], 3)
        else:
            return "N/A"
    elif tp == "clusters":
        if is_ok(r[args]):
            return r[args]["results"]["clusters"]
        else:
            return "N/A"
    elif tp == "subgraphs":
        if is_ok(r[args]):
            return r[args]["results"]["subgraphs"]
        else:
            return "N/A"
    elif tp == "compiled":
        if is_ok(r[args]):
            return r[args]["results"]["blade_compiled_nodes"]
        else:
            return "N/A"
    else:
        assert False, f"Found unknown type {tp}"

# Dump the result to csv, so that can be used in Google Sheets
def dump_result(result, header, base_key):
    s = [",".join(header) + "\n"]
    # sort models by their names in lowercase
    for k in sorted(result, key=lambda x: x[0].lower()):
        rt = [str(k[0]), str(result[k]["precision"]), str(result[k]["batch_size"])]
        for h in header[3:]:
            rt.append(str(find_result_by_header(result[k], split_header(h), base_key)))
        s.append(",".join(rt) + "\n")
    return "".join(s)

def analyze_result(result_dir: str, base_key: str) -> str:
    files = get_nonempty_json(result_dir)
    # make sure the baseline file exists
    file_keys = list(map(lambda x: Path(x).stem, files))
    assert base_key in file_keys, f"Baseline key {base_key} is not found in all files: {file_keys}."
    result = {}
    for f in files:
        process_json(result, f, base_key)
    header = generate_header(result, base_key)
    s = dump_result(result, header, base_key)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result-dir", required=True, help="Specify the result directory")
    parser.add_argument("-b", "--base-key", default="eager", help="Specify the baseline key")
    args = parser.parse_args()
    s = analyze_result(args.result_dir, args.base_key)
    print(s)
