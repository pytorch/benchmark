import argparse

def parse_tb_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resize", choices=["default", "448x608"], default="default", help="Resize the image to specified size")
    args, unknown_args = parser.parse_known_args(args)
    return args, unknown_args