from typing import Optional
from ..regression_detector import TorchBenchABTestResult, TorchBenchABTestMetric


def run(control_metrics: str, treatment_metrics: str) -> Optional[TorchBenchABTestResult]:
    return ""