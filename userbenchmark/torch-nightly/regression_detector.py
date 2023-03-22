from typing import Optional
from ..utils import TorchBenchABTestResult, TorchBenchABTestMetric

def run(control, treatment) -> Optional[TorchBenchABTestResult]:
    control_env = control["environ"]
    treatment_env = treatment["environ"]
    control_metrics = control["metrics"]
    treatment_metrics = control["metrics"]
    details = {}
    for metric_names in control_metrics.keys():
        control_metric = control_metrics[metric_names]
        treatment_metric = treatment_metrics[metric_names]
        delta = (treatment_metric - control_metric) / control_metric
        details[metric_names] = TorchBenchABTestMetric(control=control_metric, treatment=treatment_metric, delta=delta)
    return TorchBenchABTestResult(control_env=control_env, \
                                  treatment_env=treatment_env, \
                                  details=details, \
                                  bisection=None)
