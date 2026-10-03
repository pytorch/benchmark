"""Maestro benchmarking package."""

import sys
from pathlib import Path

# Internal modules still import `core`, `blocks`, and `ops_presets` as top-level
# names. Put this package directory on sys.path so `python -m maestro` works.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

_PYPROJECT = _PKG_DIR.parent / "pyproject.toml"


def _load_version() -> str:
    if _PYPROJECT.is_file():
        for line in _PYPROJECT.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("version"):
                return stripped.split("=", 1)[1].strip().strip("\"'")
    return "0.0.0+unknown"


__version__ = _load_version()

from .main import _run_benchmark as run_benchmark

__all__ = ["__version__", "run_benchmark"]
