"""CPU regressions for the TorchBench metrics adapter."""

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def entrypoint():
    package = "userbenchmark.maestro.maestro"
    engine = types.ModuleType(package)
    engine.__version__ = "1.0.1"
    engine.run_benchmark = lambda config: None
    distributed = types.ModuleType(f"{package}.core.utils.distributed")
    distributed.get_rank_and_world_size = lambda: (0, 2)
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("userbenchmark.maestro.run", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package: engine, distributed.__name__: distributed}):
        spec.loader.exec_module(module)
        yield module, distributed


def test_metrics_output_is_numeric_and_written_only_by_root(entrypoint, tmp_path):
    module, distributed = entrypoint
    results = pd.DataFrame(
        [{"pattern": "overlap", "block_name": "gemm", "avg_latency_ns": 25000,
          "min_latency_ns": 20000, "max_latency_ns": 30000,
          "p99_latency_ns": 29900, "avg_bandwidth_gbps": float("nan")}]
    )
    outputs = []
    module.run_benchmark = lambda config: results
    module.get_output_json = lambda name, metrics: {"name": name, "metrics": metrics, "environ": {}}
    module.dump_output = lambda name, output: outputs.append(output)

    module.run([str(tmp_path / "config.yaml")])
    assert len(outputs) == 1
    metrics = outputs[0]["metrics"]
    assert metrics["overlap-gemm-avg_latency_ms"] == pytest.approx(0.025)
    assert metrics["overlap-gemm-p99_latency_ms"] == pytest.approx(0.0299)
    assert "overlap-gemm-avg_bandwidth_gbps" not in metrics
    json.dumps(outputs[0], allow_nan=False)

    distributed.get_rank_and_world_size = lambda: (1, 2)
    module.run([str(tmp_path / "config.yaml")])
    assert len(outputs) == 1


def test_duplicate_metric_names_are_rejected(entrypoint):
    module, _ = entrypoint
    row = {"pattern": "same", "block_name": "block", "avg_latency_ns": 1000}
    with pytest.raises(ValueError, match="Duplicate Maestro metric"):
        module._flatten_metrics(pd.DataFrame([row, row]))


def test_dispatcher_import_failure_has_nonzero_exit(tmp_path):
    dispatcher = Path(__file__).resolve().parents[3] / "run_benchmark.py"
    (tmp_path / "userbenchmark" / "broken").mkdir(parents=True)
    (tmp_path / "userbenchmark" / "__init__.py").write_text("")
    (tmp_path / "userbenchmark" / "broken" / "run.py").write_text(
        "raise ImportError('missing benchmark dependency')\n"
    )
    script = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "sys.argv = ['run_benchmark.py', 'broken']; "
        f"runpy.run_path({str(dispatcher)!r}, run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 1
    assert "missing benchmark dependency" in result.stdout
