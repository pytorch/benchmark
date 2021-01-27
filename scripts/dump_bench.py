"""
Dump the contents of a pytest benchmark .json file.
"""
import argparse
import json
from tabulate import tabulate

def print_benchmark_stats(data):
    print_stats = ['min', 'max', 'mean', 'stddev', 'rounds', 'median']
    headers = ['name'] + print_stats 
    rows = []
    for benchmark in data['benchmarks']:
        row = [benchmark['name']]
        row += [benchmark['stats'][k] for k in print_stats]
        rows.append(row)
    print(tabulate(rows, headers=headers))
    print()

def print_kv_table(table_name, data):
    headers = [table_name, '']
    rows = [(k, data[k]) for k in data]
    print(tabulate(rows, headers=headers))
    print()

def print_other_info(data):
    print_kv_table('Machine Info',  data['machine_info'])
    print_kv_table('Commit Info',  data['commit_info'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file")
    parser.add_argument("--table", default="benchmarks",
                        choices=['benchmarks', 'other'],
                        help="which section of the json file to tablify")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    if args.table == 'benchmarks':
        print_benchmark_stats(data)
    elif args.table == 'other':
        print_other_info(data)