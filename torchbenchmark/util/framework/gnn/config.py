import argparse


def parse_tb_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_type",
        choices=["dense", "sparse"],
        default="dense",
        help="Determine dense graph or sparse graph",
    )
    args, unknown_args = parser.parse_known_args(args)
    return args, unknown_args
