import importlib
import sys
import types

import pytest

import run_benchmark


def test_import_error_in_run_benchmark_exits_nonzero(monkeypatch):
    monkeypatch.setattr(run_benchmark, "list_benchmarks", lambda: {"fake": "fake"})
    monkeypatch.setattr(run_benchmark.importlib, "import_module", lambda _: (_ for _ in ()).throw(ImportError("boom")))

    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "fake"])
    with pytest.raises(SystemExit) as exc:
        run_benchmark.run()

    assert exc.value.code == 1


def test_run_exception_propagates_outside_import_handling(monkeypatch):
    def fake_run(_):
        raise RuntimeError("runtime failure")

    fake_module = types.SimpleNamespace(run=fake_run)

    monkeypatch.setattr(run_benchmark, "list_benchmarks", lambda: {"fake": "fake"})
    monkeypatch.setattr(run_benchmark.importlib, "import_module", lambda _: fake_module)

    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "fake"])
    with pytest.raises(RuntimeError):
        run_benchmark.run()
