"""
Detect if the regression happens over nightly runs.
"""

from typing import Optional

def detect_regression(a, b) -> Optional[str]:
    """Detect if result b regresses from result a. If so, return a userbenchmark arg str for bisection. Otherwise, return None."""
    return None
