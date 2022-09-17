import csv
import argparse
from typing import OrderedDict
from dataclasses import dataclass
import os
import pickle
from collections import defaultdict
import tabulate

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
            has_breaks = row["has_breaks"]
            job_id = row["job_id"]
            result_code, result_data = get_job_result(args, job_id)
            latency = f"{result_data['latency_median']:.3f}" if result_code else str(result_data)[:10]
            model_data[model][backend][nodes][has_breaks] = latency
    return model_data

def model_name(model):
    if "torchbenchmark.models." in model:
        model = model[len("torchbenchmark.models."):]
    if ".Model" in model:
        model = model[:model.find(".Model")]
    return model

def print_model_table(args, model, model_data):
    node_counts = OrderedDict()
    for backend in model_data:
        for node in model_data[backend]:
            node_counts[node] = node  # hack orderedset
    rows = []
    for backend in model_data:
        row = [backend, ]
        for node in node_counts:
            if node in model_data[backend]:
                row.append(f"{model_data[backend][node]['False']} / {model_data[backend][node]['True']}")
            else:
                row.append("-")
        rows.append(row)

    hdr = ("backend", ) + tuple(f"{node}_latency" for node in node_counts)
    print(f"{model_name(model)}:")
    print(tabulate.tabulate(rows, headers=hdr))
    print()

def print_results(args, data):
    print("        (without ddp breaks) / (with ddp breaks)")
    for model in data:
        print_model_table(args, model, data[model])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    data = parse_data(args)
    print_results(args, data)

if __name__ == "__main__":
    main()
