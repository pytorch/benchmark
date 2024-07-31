import argparse
import os
import importlib

from typing import List

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, help="Backfill working dir")
    parser.add_argument("--start-date", required=True, help="Start date of the backfill, in format of YYYY-mm-dd")
    parser.add_argument("--end-date", required=True, help="End date of the backfill, in format of YYYY-mm-dd")
    parser.add_argument("--step", required=True, choices=["commit", "day", "week"], help="Backfill step")
    parser.add_argument("--torchbench-repo-path", required=True, default=os.path.dirname(__file__), help="Path to the torchbench repo (for debugging)")
    parser.add_argument("--userbenchmark", required=True, help="Name of the userbenchmark to backfill")
    return parser

def get_backfiller(bm_name: str, work_dir: str):
    try:
        backfill_mod = importlib.import_module(f"userbenchmark.{bm_name}.backfill")
    except:
        # fbcode
        backfill_mod = importlib.import_module(f"userbenchmark.fb.{bm_name}.backfill")
    return backfill_mod.Backfill(work_dir)

def run_step(backfill_mod, step: str, ub_args: List[str]):
    compile_fn = backfill_mod.compile_fn
    compile_fn(step)
    run_fn = backfill_mod.run_fn
    run_fn(ub_args)

if __name__ == "__main__":
    parser = get_parser()
    args, extra_args = parser.parse_known_args()
    backfiller = get_backfiller(args.userbenchmark, args.work_dir)
    backfiller.prep()
    steps = backfiller.get_steps(args.start_date, args.end_date, args.step)
    [ run_step(step) for step in steps ]
