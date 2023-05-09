from typing import Optional
from ..utils import TorchBenchABTestResult, TorchBenchABTestMetric

DEFAULT_REGRESSION_DELTA_THRESHOLD = 0.07

def run(control, treatment) -> Optional[TorchBenchABTestResult]:
    control_env = control["environ"]
    treatment_env = treatment["environ"]
    control_metrics = control["metrics"]
    treatment_metrics = treatment["metrics"]
    details = {}
    for control_metric_name, control_metric in control_metrics.items():
        if control_metric_name in treatment_metrics:
            treatment_metric = treatment_metrics[control_metric_name]
            delta = (treatment_metric - control_metric) / control_metric
            # Trigger on BOTH slowdowns and speedups
            if abs(delta) > DEFAULT_REGRESSION_DELTA_THRESHOLD:
                details[control_metric_name] = TorchBenchABTestMetric(control=control_metric, treatment=treatment_metric, delta=delta)
    return TorchBenchABTestResult(control_env=control_env, \
                                  treatment_env=treatment_env, \
                                  details=details, \
                                  bisection=None)
