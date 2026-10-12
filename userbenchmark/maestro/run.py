"""TorchBench userbenchmark entrypoint for Maestro."""

import json
import math
import sys
from pathlib import Path
from typing import List

from ..utils import dump_output, get_output_json
from .maestro import __version__ as maestro_version, run_benchmark

BM_NAME = "maestro"


def run(args: List[str]):
    """TorchBench hook: `torchrun run_benchmark.py maestro <config.yaml>`."""
    if len(args) != 1 or not isinstance(args[0], str):
        print(
            "Usage: torchrun --nproc-per-node=<gpus> run_benchmark.py maestro <config.yaml>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_file = Path(args[0])
    results_df = run_benchmark(config_file)
    from .maestro.core.utils.distributed import get_rank_and_world_size
    rank, _ = get_rank_and_world_size()

    if rank != 0:
        return

    metrics = get_output_json(BM_NAME, _flatten_metrics(results_df))
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


def _flatten_metrics(results_df):
    """Expose finite performance values in TorchBench's numeric metrics map."""
    metric_columns = {
        # "df column": ("alias", scale),
        "avg_latency_ns": ("avg_latency_ms", 1e-6),
        "min_latency_ns": ("min_latency_ms", 1e-6),
        "max_latency_ns": ("max_latency_ms", 1e-6),
        "p99_latency_ns": ("p99_latency_ms", 1e-6),
        "avg_bandwidth_gbps": ("avg_bandwidth_gbps", 1),
        "avg_shared_bw_gbps": ("avg_shared_bw_gbps", 1),
        "avg_overlap_pct": ("avg_overlap_pct", 1),
    }
    metrics = {}
    for row in results_df.to_dict(orient="records"):
        for column, (suffix, scale) in metric_columns.items():
            value = row.get(column)
            if value is None or not math.isfinite(float(value)):
                continue
            name = f"{row['pattern']}-{row['block_name']}-{suffix}"
            if name in metrics:
                raise ValueError(f"Duplicate Maestro metric name: {name}")
            metrics[name] = float(value) * scale
    return metrics
