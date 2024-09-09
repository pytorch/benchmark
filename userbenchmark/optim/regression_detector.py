from typing import Optional

from ..utils import TorchBenchABTestMetric, TorchBenchABTestResult

DEFAULT_REGRESSION_DELTA_THRESHOLD = 0.3
COMPILE_TIME_REGRESSION_DELTA_THRESHOLD = 2.0


def run(control, treatment) -> Optional[TorchBenchABTestResult]:
    control_env = control["environ"]
    treatment_env = treatment["environ"]
    control_metrics = control["metrics"]
    treatment_metrics = treatment["metrics"]
    details = {}
    for control_metric_name, control_metric in control_metrics.items():
        if control_metric_name in treatment_metrics:
            regression_threshold = (
                COMPILE_TIME_REGRESSION_DELTA_THRESHOLD
                if "compile_time" in control_metric_name
                else DEFAULT_REGRESSION_DELTA_THRESHOLD
            )
            treatment_metric = treatment_metrics[control_metric_name]
            delta = (treatment_metric - control_metric) / control_metric
            # Trigger on BOTH slowdowns and speedups
            if abs(delta) > regression_threshold:
                details[control_metric_name] = TorchBenchABTestMetric(
                    control=control_metric, treatment=treatment_metric, delta=delta
                )
    # control_only_metrics/treatment_only_metrics will be filled in later by the main regression detector
    return TorchBenchABTestResult(
        name=control["name"],
        control_env=control_env,
        treatment_env=treatment_env,
        details=details,
    )
