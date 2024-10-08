import os
import shutil
import sys
from typing import List

"""
A dictionary mapping short metric names to their corresponding NVIDIA Nsight Compute
(NCU) metric names. Don't directly use the NCU metric names in the code, use these short
names instead.
"""
short_ncu_metric_name = {
    "inst_executed_ffma_peak": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "inst_executed_dfma_peak": "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",
    "inst_executed_fadd": "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
    "inst_executed_fmul": "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
    "inst_executed_ffma": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed",
    "inst_executed_dadd": "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed",
    "inst_executed_dmul": "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed",
    "inst_executed_dfma": "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed",
    "dram_bytes_write": "dram__bytes_write.sum",
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_per_second": "dram__bytes.sum.per_second",
    "sm_freq": "smsp__cycles_elapsed.avg.per_second",
    "dram_bandwidth": "dram__bytes.sum.per_second",
}
# A dictionary mapping benchmark metric names to their corresponding short NCU metric names.
bench_metric_to_short_ncu_metric = {
    "memory_traffic": ["dram_bytes_write", "dram_bytes_read"],
    "arithmetic_intensity": [
        "inst_executed_ffma_peak",
        "inst_executed_dfma_peak",
        "inst_executed_fadd",
        "inst_executed_fmul",
        "inst_executed_ffma",
        "inst_executed_dadd",
        "inst_executed_dmul",
        "inst_executed_dfma",
        "dram_bytes_write",
        "dram_bytes_read",
        "sm_freq",
        "dram_bandwidth",
    ],
}


def import_ncu_python_path():
    """
    This function modifies the Python path to include the NVIDIA Nsight Compute (NCU) Python modules.
    It searches for the 'ncu' command in the system PATH, determines its location, and appends the
    'extras/python' directory to the Python path.

    Raises:
        FileNotFoundError: If the 'ncu' command is not found in the system PATH.
        FileNotFoundError: If the 'extras/python' directory does not exist in the determined NCU path.
    """
    ncu_path = shutil.which("ncu")
    if not ncu_path:
        raise FileNotFoundError("Could not find 'ncu' command in PATH.")
    ncu_path = os.path.dirname(ncu_path)
    if not os.path.exists(os.path.join(ncu_path, "extras/python")):
        raise FileNotFoundError(
            f"'extras/python' does not exist in the provided ncu_path: {ncu_path}"
        )
    sys.path.append(os.path.join(ncu_path, "extras/python"))


def get_mem_traffic(kernel):
    return (
        kernel.metric_by_name(short_ncu_metric_name["dram_bytes_read"]).value(),
        kernel.metric_by_name(short_ncu_metric_name["dram_bytes_write"]).value(),
    )


def get_arithmetic_intensity(kernel):
    fp32_add_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_fadd"]
    ).value()
    fp32_mul_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_fmul"]
    ).value()
    fp32_fma_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_ffma"]
    ).value()
    fp32_achieved = fp32_add_achieved + fp32_mul_achieved + 2 * fp32_fma_achieved
    fp64_add_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_dadd"]
    ).value()
    fp64_mul_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_dmul"]
    ).value()
    fp64_fma_achieved = kernel.metric_by_name(
        short_ncu_metric_name["inst_executed_dfma"]
    ).value()
    fp64_achieved = fp64_add_achieved + fp64_mul_achieved + 2 * fp64_fma_achieved
    sm_freq = kernel.metric_by_name(short_ncu_metric_name["sm_freq"]).value()
    fp32_flops = fp32_achieved * sm_freq
    fp64_flops = fp64_achieved * sm_freq
    dram_bandwidth = kernel.metric_by_name(
        short_ncu_metric_name["dram_bandwidth"]
    ).value()
    fp32_arithmetic_intensity = fp32_flops / dram_bandwidth
    fp64_arithmetic_intensity = fp64_flops / dram_bandwidth
    return fp32_arithmetic_intensity, fp64_arithmetic_intensity


def read_ncu_report(report_path: str, required_metrics: List[str]):
    assert os.path.exists(
        report_path
    ), f"NCU report at {report_path} does not exist. Make sure you add --metrics ncu_rep to your benchmark run."
    import_ncu_python_path()
    import ncu_report

    results = {}
    test_report = ncu_report.load_report(report_path)
    assert (
        test_report.num_ranges() > 0
    ), f"No profile data found in NCU report at {report_path}"
    default_range = test_report.range_by_idx(0)
    assert (
        default_range.num_actions() > 0
    ), f"No profile data found in the default range of NCU report at {report_path}"
    for i in range(default_range.num_actions()):
        kernel = default_range.action_by_idx(i)
        if "memory_traffic" in required_metrics:
            mem_traffic = get_mem_traffic(kernel)
            results["memory_traffic"] = mem_traffic
        if "arithmetic_intensity" in required_metrics:
            results["arithmetic_intensity"] = get_arithmetic_intensity(kernel)
    return results
