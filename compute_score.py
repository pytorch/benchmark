
"""
Compute the benchmark score given a frozen score configuration and current benchmark data.
"""
import argparse
import json
import math
import yaml
import sys
import os

from torchbenchmark.score.compute_score import TorchBenchScore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_version", choices=['v1', 'v2'], default="v1",
        help="which version of score to use - choose from v1 or v2")
    parser.add_argument("--benchmark_data_file",
        help="pytest-benchmark json file with current benchmark data")
    parser.add_argument("--benchmark_data_dir",
        help="directory containing multiple .json files for each of which to compute a score")
    parser.add_argument("--relative", action='store_true',
        help="use the first json file in benchmark data dir instead of the reference yaml")
    parser.add_argument("--output-norm-only", action='store_true',
        help="use the benchmark data file specified to output reference norm yaml")
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
            files.append(args.benchmark_data_file)
            benchmark_data.append(data)
    elif args.benchmark_data_dir is not None:
        for f in sorted(os.listdir(args.benchmark_data_dir)):
            path = os.path.join(args.benchmark_data_dir, f)
            if os.path.isfile(path) and os.path.splitext(path)[1] == '.json':
                with open(path) as data_file:
                    data = json.load(data_file)
                    files.append(f)
                    benchmark_data.append(data)

    if args.output_norm_only:
        score_config = TorchBenchScore(ref_data=benchmark_data[0], version=args.score_version)
        print(yaml.dump(score_config.get_norm(benchmark_data[0])))
        exit(0)

    if args.relative:
        score_config = TorchBenchScore(ref_data=benchmark_data[0], version=args.score_version)
    else:
        score_config = TorchBenchScore(version=args.score_version)

    results = []
    for fname, data in zip(files, benchmark_data):
        result = {}
        score = score_config.compute_score(data)
        result["file"] = fname
        result["pytorch_version"] = data['machine_info']['pytorch_version']
        result["score"] = score
        results.append(result)

    print(json.dumps(results, indent=4))
