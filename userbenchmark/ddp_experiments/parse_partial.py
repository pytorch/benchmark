import csv
import json
import copy
import argparse
from typing import OrderedDict
from dataclasses import dataclass
import os
import pickle
from collections import defaultdict
import tabulate
import sys

def get_job_result(args, job_id, worker_rank=0):
    root = os.path.join(args.results_dir, f"{job_id}_{worker_rank}_")
    err = root + "log.err"
    out = root + "log.out"
    pkl = root + "result.pkl"
    if not os.path.isfile(pkl):
        return False, f"waiting.. or {pkl} did not exist"

    with open(pkl, 'rb') as f:
        dat = pickle.load(f)
        assert isinstance(dat, tuple), f"Expected a tuple as result but got a {type(dat)}: {dat}"
        desc, payload = dat
        if desc == "error":
            # print(f"Got error: {desc}, traceback:\n{payload}")
            return False, payload
        elif desc == "success":
            # print(f"Success: {payload}")
            return True, payload

        # print(f"Unknown result: {dat}")
        return False, dat

    return False, None

def parse_data(args):
    """
    Schema:
    model_data["model"]["backend"][#nodes] = latency_median
    """
    model_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    with open(args.csv) as f:
        runs = csv.DictReader(f)
        for row in runs:
            model = row["model"]
            backend = row["backend"]
            nodes = row["nodes"]
            has_breaks = str(row["has_breaks"])
            job_id = row["job_id"]
            result_code, result_data = get_job_result(args, job_id)
            latency = f"{result_data['latency_median']:.3f}" if result_code else str(result_data)[:10]
            model_data[model][backend][nodes][has_breaks] = latency
    return model_data

def parse_allocation_reuse_data(args):
    model_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    with open(args.csv) as f:
        runs = csv.DictReader(f)
        for row in runs:
            job_id = row["job_id"]
            rank_id = 0
            pkl = os.path.join(args.results_dir, f"{job_id}_{rank_id}_result.pkl")
            with open(pkl, "rb") as f:
                dat = pickle.load(f)
            assert isinstance(dat, tuple)
            desc, payload = dat
            if desc != "success":
                print(f"job_id {job_id} failed")
                continue
            for item in payload:
                model = item["model_name"]
                backend = item["backend"]
                nodes = item["nodes"]
                has_breaks = str(item["has_breaks"])
                latency = item["result"]["latency_median"]
                model_data[model][backend][nodes][has_breaks] = f"{latency:.3f}"
    return model_data

def parse_partial(args):
    model_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    rank_id = 0
    log_path = os.path.join(args.results_dir, f"{args.job_id}_{rank_id}_log.out")
    with open(log_path, "r") as f:
        content = f.read()
        pieces = content.split("<RESULT>")
        pieces = [x.split("</RESULT>") for x in pieces]
        pieces = [x[0] for x in pieces if len(x) == 2]
        pieces = [json.loads(x) for x in pieces]
        for row in pieces:
            model = row["model_name"]
            backend = row["backend"]
            nodes = row["nodes"]
            has_breaks = str(row["has_breaks"])
            if isinstance(row["result"], dict):
                latency = float(row["result"]["latency_median"])
                if isinstance(model_data[model][backend][nodes][has_breaks], list):
                    model_data[model][backend][nodes][has_breaks].append(latency)
            else:
                model_data[model][backend][nodes][has_breaks] = "error"
    return model_data

def model_name(model):
    if "torchbenchmark.models." in model:
        model = model[len("torchbenchmark.models."):]
    if ".Model" in model:
        model = model[:model.find(".Model")]
    return model

def median(x):
    if len(x) == 0:
        return 0
    x = copy.copy(x)
    x = sorted(x)
    idx = int(len(x)/2)
    if len(x) % 2 == 0:
        return (x[idx - 1] + x[idx]) / 2
    else:
        return x[idx]

def print_model_table(args, model, model_data):
    node_counts = OrderedDict()
    for backend in model_data:
        for node in model_data[backend]:
            node_counts[node] = node  # hack orderedset
    node_counts = list(node_counts)
    node_counts = sorted(node_counts)
    rows = []
    for has_breaks in [False, True]:
        for backend in model_data:
            row = [f"{backend} {'w/' if has_breaks else 'wo/'}breaks", ]
            for node in node_counts:
                if node in model_data[backend]:
                    res = model_data[backend][node][str(has_breaks)]
                    if isinstance(res, list):
                        if len(res) > 0:
                            res = f"{min(res):.3f} : {median(res):.3f} : {max(res):.3f}"
                        else:
                            res = 0.0
                    row.append(res)
                else:
                    row.append("-")
            rows.append(row)

    hdr = ("backend", ) + tuple(f"{node}_latency" for node in node_counts)
    print(tabulate.tabulate(rows, headers=hdr))
    print()

def print_csv(args, data):
    csv_data = []
    node_counts = OrderedDict()
    for model in data:
        for backend in data[model]:
            for node in data[model][backend]:
                node_counts[node] = node  # hack orderedset
    node_counts = list(node_counts)
    node_counts = sorted(node_counts)
    labels = ["model", "has_ddp_breaks", "backend"]
    for node in node_counts:
        labels.append(f"{node}-node median")
        # labels.append(f"{node}-node min")
        # labels.append(f"{node}-node max")
    for has_breaks in [False, True]:
        for model in data:
            for backend in data[model]:
                row = {
                    "model": model,
                    "has_ddp_breaks": str(has_breaks),
                    "backend": backend,
                }
                for node in node_counts:
                    if node in data[model][backend]:
                        latency = data[model][backend][node][str(has_breaks)]
                    else:
                        latency = 0.

                    if isinstance(latency, list) and len(latency) == 0:
                        latency = 0.
                    node_label_median = f"{node}-node median"
                    node_label_min = f"{node}-node min"
                    node_label_max = f"{node}-node max"
                    latency_list = latency if isinstance(latency, list) else [latency]
                    row[node_label_median] = median(latency_list)
                    # row[node_label_min] = min(latency_list)
                    # row[node_label_max] = max(latency_list)
                csv_data.append(row)
    csv_writer = csv.DictWriter(sys.stdout, fieldnames=labels)
    csv_writer.writeheader()
    for row in csv_data:
        csv_writer.writerow(row)

def print_results(args, data):
    for model in data:
        print_model_table(args, model, data[model])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--csv_out", action="store_true")
    args = parser.parse_args()
    data = parse_partial(args)
    if args.csv_out:
        print_csv(args, data)
    else:
        print_results(args, data)

if __name__ == "__main__":
    main()
