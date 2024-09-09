from ..utils import TorchBenchABTestMetric, TorchBenchABTestResult
from . import BM_NAME

DEFAULT_REGRESSION_DELTA_THRESHOLD = 0.07


def run(control, treatment) -> TorchBenchABTestResult:
    control_env = control["environ"]
    control_env["git_commit_hash"] = control["environ"]["pytorch_git_version"]
    control_metrics = control["metrics"]
    treatment_env = treatment["environ"]
    treatment_env["git_commit_hash"] = treatment["environ"]["pytorch_git_version"]
    treatment_metrics = treatment["metrics"]
    details = {}
    for metric_names in control_metrics.keys():
        control_metric = control_metrics[metric_names]
        treatment_metric = treatment_metrics[metric_names]
        if isinstance(control_metric, str) or isinstance(treatment_metric, str):
            if (
                control_metric == "skip_by_dryrun"
                or not control_metric == treatment_metric
            ):
                delta = f"{control_metric} -> {treatment_metric}"
                details[metric_names] = TorchBenchABTestMetric(
                    control=control_metric, treatment=treatment_metric, delta=delta
                )
        else:
            delta = (treatment_metric - control_metric) / control_metric
            if abs(delta) > DEFAULT_REGRESSION_DELTA_THRESHOLD:
                details[metric_names] = TorchBenchABTestMetric(
                    control=control_metric, treatment=treatment_metric, delta=delta
                )
    return TorchBenchABTestResult(
        name=BM_NAME,
        control_env=control_env,
        treatment_env=treatment_env,
        details=details,
        control_only_metrics={},
        treatment_only_metrics={},
        bisection="pytorch",
    )
