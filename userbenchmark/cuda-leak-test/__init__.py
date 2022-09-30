import argparse
from typing import List

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BERT_pytorch", help="Name of the model to test CUDA memory leak.")

def run(args: List[str]):
    pass