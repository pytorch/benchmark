"""Install Maestro and its dependencies for the TorchBench userbenchmark."""

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", str(Path(__file__).resolve().parent)]
    )
