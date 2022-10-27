from typing import List
from .args import parse_args

def run(args: List[str]):
    args = parse_args(args)