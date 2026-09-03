"""TorchBench userbenchmark entrypoint for Maestro."""

import json
import sys
from pathlib import Path
from typing import List

from ..utils import dump_output, get_output_json
from .maestro import __version__ as maestro_version, run_benchmark

BM_NAME = "maestro"


def run(args: List[str]):
    """TorchBench userbenchmark hook: `python run_benchmark.py maestro-benchmark <config.yaml>`."""
    if len(args) != 1 or not isinstance(args[0], str):
        print(
            "Usage: python run_benchmark.py maestro-benchmark <config.yaml>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_file = Path(args[0])
    results_df = run_benchmark(config_file)
    if results_df is None:
        return

    metrics = get_output_json(BM_NAME, {})
    metrics["environ"].update(
        {
            "config_file": str(config_file.resolve()),
            "maestro_version": maestro_version,
            "python_version": sys.version,
        }
    )
    # to_json maps NaN to null so the file is valid JSON.
    metrics["results"] = json.loads(results_df.to_json(orient="records"))
    dump_output(BM_NAME, metrics)