"""Maestro benchmarking package."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re


def _load_version() -> str:
    try:
        return version("maestro")
    except PackageNotFoundError:
        # Source checkouts need not have installed distribution metadata.
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject.is_file():
            match = re.search(r'^version\s*=\s*[\"\']([^\"\']+)', pyproject.read_text(), re.MULTILINE)
            if match:
                return match.group(1)
        return "0.0.0+unknown"


__version__ = _load_version()

from .main import _run_benchmark as run_benchmark

__all__ = ["__version__", "run_benchmark"]
