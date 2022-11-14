import argparse

from typing import List

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, default=[],
                        help="Specify one or more models to run. If not set, trigger a sweep-run on all models.")
    parser.add_argument("-o", "--output", type=str, help="The default output json file.")
    args, unknown_args = parser.parse_known_args(args)
    return args, unknown_args