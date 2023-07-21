"""
Utilities to measure metrics of a model.
"""
import torch
import time
import dataclasses
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.util.experiment.instantiator import TorchBenchModelConfig
from torchbenchmark import ModelTask
from typing import List, Union, Tuple, Optional

WARMUP_ROUNDS = 10
BENCHMARK_ITERS = 15
MEMPROF_ITER = 2
NANOSECONDS_PER_MILLISECONDS = 1_000_000.0

@dataclasses.dataclass
class TorchBenchModelMetrics:
    latencies: List[float]
    throughputs: List[float]
    cpu_peak_mem: Optional[float]
    gpu_peak_mem: Optional[float]
    pt2_compilation_time: Optional[float]
    model_flops: Optional[float]

def get_latencies(func, device: str, nwarmup=WARMUP_ROUNDS, num_iter=BENCHMARK_ITERS) -> List[float]:
    "Run one step of the model, and return the latency in milliseconds."
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        func()
    result_summary = []
    for _i in range(num_iter):
        if device == "cuda":
            torch.cuda.synchronize()
            # Collect time_ns() instead of time() which does not provide better precision than 1
            # second according to https://docs.python.org/3/library/time.html#time.time.
            t0 = time.time_ns()
            func()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            t1 = time.time_ns()
        else:
            t0 = time.time_ns()
            func()
            t1 = time.time_ns()
        result_summary.append((t1 - t0) / NANOSECONDS_PER_MILLISECONDS)
    return result_summary


def get_peak_memory(func, device: str, num_iter=MEMPROF_ITER, export_metrics_file='', metrics_needed=[], metrics_gpu_backend='dcgm', cpu_monitored_pid=None) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    "Run one step of the model, and return the peak memory in MB."
    from components.model_analyzer.TorchBenchAnalyzer import ModelAnalyzer
    new_metrics_needed = [_ for _ in metrics_needed if _ in ['cpu_peak_mem', 'gpu_peak_mem']]
    if not new_metrics_needed:
        raise ValueError(f"Expected metrics_needed to be non-empty, get: {metrics_needed}")
    mem_model_analyzer = ModelAnalyzer(export_metrics_file, new_metrics_needed, metrics_gpu_backend, cpu_monitored_pid)
    continue_num_iter = BENCHMARK_ITERS - num_iter
    def work_func():
        if device == "cuda":
            torch.cuda.synchronize()
            func()
            torch.cuda.synchronize()
        else:
            func()

    t0 = time.time_ns()
    work_func()
    t1 = time.time_ns()
    # if total execution time is less than 15ms, we run the model for BENCHMARK_ITERS times
    #  to get more accurate peak memory
    if (t1 - t0) < 15 * NANOSECONDS_PER_MILLISECONDS:
        num_iter = BENCHMARK_ITERS
    else:
        num_iter = MEMPROF_ITER
    mem_model_analyzer.start_monitor()

    for _i in range(num_iter):
        work_func()
    mem_model_analyzer.stop_monitor()
    mem_model_analyzer.aggregate()
    device_id = None
    gpu_peak_mem = None
    cpu_peak_mem = None
    if 'gpu_peak_mem' in metrics_needed:
        device_id, gpu_peak_mem = mem_model_analyzer.calculate_gpu_peak_mem()
    if 'cpu_peak_mem' in metrics_needed:
        cpu_peak_mem = mem_model_analyzer.calculate_cpu_peak_mem()
    if export_metrics_file:
        mem_model_analyzer.update_export_name("_peak_memory")
        mem_model_analyzer.export_all_records_to_csv()
    return cpu_peak_mem, device_id, gpu_peak_mem

def get_pt2_compilation_time(model: Union[BenchmarkModel, ModelTask]):
    if isinstance(model, ModelTask):
        return model.get_model_attribute('pt2_compilation_time')
    else:
        return model.pt2_compilation_time

def get_model_flops(model: Union[BenchmarkModel, ModelTask]) -> float:
    "Run one step of the model, and return the model total flops."
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode()
    def work_func():
        if model.device == "cuda":
            torch.cuda.synchronize()
            model.invoke()
            torch.cuda.synchronize()
        else:
            model.invoke()
    with flop_counter:
        work_func()
    total_flops = sum([v for _, v in flop_counter.flop_counts["Global"].items()])
    return total_flops

def get_model_test_metrics(model: Union[BenchmarkModel, ModelTask], metrics=[], export_metrics_file=False, metrics_gpu_backend='nvml', nwarmup=WARMUP_ROUNDS, num_iter=BENCHMARK_ITERS) -> TorchBenchModelMetrics:
    import os
    latencies = None
    throughputs = None
    cpu_peak_mem = None
    gpu_peak_mem = None
    pt2_compilation_time = None
    model_flops = None
    if not (isinstance(model, BenchmarkModel) or isinstance(model, ModelTask)):
        raise ValueError(f"Expected BenchmarkModel or ModelTask, get type: {type(model)}")
    model_pid = os.getpid() if isinstance(model, BenchmarkModel) else model.worker.proc_pid()
    device = model.device if isinstance(model, BenchmarkModel) else model.get_model_attribute("device")
    if 'latencies' in metrics or 'throughputs' in metrics:
        latencies = get_latencies(model.invoke, device, nwarmup=nwarmup, num_iter=num_iter)
    if 'cpu_peak_mem' in metrics or 'gpu_peak_mem' in metrics:
        cpu_peak_mem, _device_id, gpu_peak_mem = get_peak_memory(model.invoke, device, export_metrics_file=export_metrics_file, metrics_needed=metrics, metrics_gpu_backend=metrics_gpu_backend, cpu_monitored_pid=model_pid)
    if 'throughputs' in metrics:
        throughputs = [model.batch_size * 1000 / latency for latency in latencies]
    if 'pt2_compilation_time' in metrics:
        pt2_compilation_time = get_pt2_compilation_time(model)
    if 'model_flops' in metrics:
        model_flops = get_model_flops(model)
    return TorchBenchModelMetrics(latencies, throughputs, cpu_peak_mem, gpu_peak_mem, pt2_compilation_time, model_flops)

def get_model_accuracy(model_config: TorchBenchModelConfig, isolated: bool=True) -> str:
    import copy
    from torchbenchmark.util.experiment.instantiator import load_model_isolated, load_model
    # Try load minimal batch size, if fail, load the default batch size
    accuracy_model_config = copy.deepcopy(model_config)
    if not "--accuracy" in accuracy_model_config.extra_args:
        accuracy_model_config.extra_args = ["--accuracy"] + accuracy_model_config.extra_args
    if isolated:
        model = load_model_isolated(accuracy_model_config)
        accuracy = model.get_model_attribute("accuracy")
        del model
        return accuracy
    else:
        model = load_model(model_config)
        return model.accuracy
