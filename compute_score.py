
"""
Compute the benchmark score given a frozen score configuration and current benchmark data.
"""
import argparse
import json
import math
import sys
import os

from torchbenchmark.score.compute_score import TorchBenchScore, TORCHBENCH_V0_SCORE
from tabulate import tabulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=['relative', 'v0'], default="relative",
        help="which score configuration to use- relative means use first data file as reference for others.")
    parser.add_argument("--benchmark_data_file",
        help="pytest-benchmark json file with current benchmark data")
    parser.add_argument("--benchmark_data_dir",
        help="directory containing multiple .json files for each of which to compute a score")
    args = parser.parse_args()

    if args.benchmark_data_file is None and args.benchmark_data_dir is None:
        parser.print_help(sys.stderr)
        raise ValueError("Invalid command-line arguments. You must specify a data file or a data dir.")

    files = []
    benchmark_data = []
    scores = []
    if args.benchmark_data_file is not None:
        with open(args.benchmark_data_file) as data_file:
            data = json.load(data_file)
            benchmark_data.append(data)
    elif args.benchmark_data_dir is not None:
        for f in sorted(os.listdir(args.benchmark_data_dir)):
            path = os.path.join(args.benchmark_data_dir, f)
            if os.path.isfile(path) and os.path.splitext(path)[1] == '.json':
                with open(path) as data_file:
                    data = json.load(data_file)
                    files.append(f)
                    benchmark_data.append(data)

    if args.configuration == "relative":
        ref_data = benchmark_data[0]
        score_config = TorchBenchScore(ref_data)
    elif args.configuration == "v0":
        score_config = TORCHBENCH_V0_SCORE
    else:
        raise ValueError("Invalid score configuration")

    results = [('File', 'Score', 'PyTorch Version')]
    for fname, data in zip(files, benchmark_data):
        score = score_config.compute_score(data)
        results.append((fname, score, data['machine_info']['pytorch_version']))

    print(tabulate(results, headers='firstrow'))
