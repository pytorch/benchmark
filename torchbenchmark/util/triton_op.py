import argparse
import copy
import functools
import gc
import random
import time
import warnings
from dataclasses import asdict, dataclass, fields, make_dataclass
from enum import Enum
from numbers import Number
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy
import tabulate
import torch
import triton
from torchbenchmark.util.env_check import fresh_triton_cache, set_random_seed
from torchbenchmark.util.experiment.metrics import get_peak_memory
from torchbenchmark.util.extra_args import apply_decoration_args, parse_decoration_args
from torchbenchmark.util.input import input_cast

DEFAULT_WARMUP = 25
DEFAULT_RUN_ITERS = 100
DEFAULT_QUANTILES = [0.5, 0.1, 0.9]
REGISTERED_BENCHMARKS: Dict[str, List[str]] = {}
REGISTERED_METRICS: Dict[str, List[str]] = {}
BASELINE_BENCHMARKS: Dict[str, str] = {}
BUILTIN_METRICS = [
    "latency",
    "tflops",
    "speedup",
    "accuracy",
    "compile_time",
    "ncu_trace",
    "cpu_peak_mem",
    "gpu_peak_mem",
    "hw_roofline",
]
BASELINE_SKIP_METRICS = set(["speedup", "accuracy"])
X_ONLY_METRICS = set(["hw_roofline"])
PRECISION_DTYPE_MAPPING = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Mode(Enum):
    FWD = "fwd"
    BWD = "bwd"
    FWD_BWD = "fwd_bwd"
    FWD_NO_GRAD = "fwd_no_grad"


def do_bench_walltime(fn, warmup=25, rep=100):
    fn()
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    estimate_ms = (end_time - start_time) * 1e3 / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    wall_time_ms = (end_time - start_time) * 1e3 / n_repeat
    return wall_time_ms

def _find_param_loc(l, key: str) -> int:
    try:
        return l.index(key)
    except ValueError:
        return -1
def _remove_params(l, loc):
    if loc == -1:
        return l
    return l[:loc] + l[loc+2:]

@dataclass
class BenchmarkOperatorMetrics:
    # latency in ms
    latency: Optional[List[float]]
    # tflops
    tflops: Optional[List[float]]
    # speedup over baseline
    speedup: Optional[float]
    # accuracy over baseline
    accuracy: Optional[bool]
    # wall time
    walltime: Optional[float]
    # compile time
    compile_time: Optional[float]
    # ncu trace file
    ncu_trace: Optional[str]
    # cpu peak memory
    cpu_peak_mem: Optional[float]
    # gpu peak memory
    gpu_peak_mem: Optional[float]
    # error message
    error_msg: Optional[str]
    # hw roofline
    hw_roofline: Optional[float]
    # extra metrics
    extra_metrics: Dict[str, float]


@dataclass
class BenchmarkOperatorResult:
    # Print the result in a table format
    op_name: str
    metrics: List[str]
    result: List[Tuple[Number, Dict[str, BenchmarkOperatorMetrics]]]
    _result_dict: Optional[Dict[Number, Dict[str, BenchmarkOperatorMetrics]]] = None

    def _table(self):
        table = []
        # generate headers
        headers = ["x_val"]
        y_val = self.result[0][1]
        y_val_keys = list(y_val.keys())
        # move the baseline benchmark to the front of the list if exists
        if BASELINE_BENCHMARKS[self.op_name] in y_val_keys:
            y_val_keys.insert(
                0, y_val_keys.pop(y_val_keys.index(BASELINE_BENCHMARKS[self.op_name]))
            )
        key_metrics = {}
        # Add header for x_only_metrics
        x_only_metrics = sorted([metric for metric in self.metrics if metric in X_ONLY_METRICS ])
        headers.extend(x_only_metrics)
        for k in y_val_keys:
            metrics = (
                [x for x in self.metrics if x not in BASELINE_SKIP_METRICS]
                if k == BASELINE_BENCHMARKS[self.op_name]
                else self.metrics
            )
            # Exclude x_only_metrics for impl-specific metrics
            metrics = [ metric for metric in metrics if not metric in x_only_metrics ]
            key_metrics[k] = sorted(metrics)
            for metric in metrics:
                # add extra metrics
                headers.append(f"{k}_{metric}")
        # generate rows
        for x_val, y_val in self.result:
            row = []
            row.append(x_val)
            # Append x_val_only metrics
            for x_only_metric in x_only_metrics:
                x_only_metric_dict = asdict(y_val[y_val_keys[0]])
                row.append(x_only_metric_dict[x_only_metric])
            for k in y_val_keys:
                metrics_dict = asdict(y_val[k])
                for metric in key_metrics[k]:
                    if metrics_dict["error_msg"]:
                        row.append(metrics_dict["error_msg"])
                        continue
                    _metrics_dict = (
                        metrics_dict["extra_metrics"]
                        if metric in metrics_dict["extra_metrics"]
                        else metrics_dict
                    )
                    if isinstance(_metrics_dict[metric], list):
                        row.append(numpy.median(_metrics_dict[metric]))
                    elif isinstance(_metrics_dict[metric], bool):
                        row.append(1.0 if _metrics_dict[metric] else 0.0)
                    else:
                        row.append(_metrics_dict[metric])
            table.append(row)
        return headers, table

    @property
    def csv(self):
        headers, table = self._table()
        headers = "; ".join(headers)
        table = "\n".join(["; ".join([str(v) for v in row]) for row in table])
        return f"{headers}\n{table}"

    @property
    def x_vals(self):
        return sorted(self._get_result_dict().keys())

    def get_y_vals(self, x_val, provider, metric_name: str):
        if provider in X_ONLY_METRICS:
            maybe_baseline = REGISTERED_BENCHMARKS[self.op_name][0]
            metrics_dict = asdict(self._get_result_dict()[x_val][maybe_baseline])
            metric_name = provider
        else:
            y_vals = self._get_result_dict()[x_val][provider]
            metrics_dict = asdict(y_vals)
        if metric_name in metrics_dict:
            return metrics_dict[metric_name]
        assert (
            metric_name in metrics_dict["extra_metrics"]
        ), f"Metric {metric_name} could not be found."
        return metrics_dict["extra_metrics"][metric_name]

    def _get_result_dict(self):
        if not self._result_dict:
            self._result_dict = {}
            for x_val, y_val in self.result:
                self._result_dict[x_val] = y_val
        return self._result_dict

    def __str__(self):
        headers, table = self._table()
        table = tabulate.tabulate(table, headers=headers, stralign="right")
        return table


def register_benchmark(baseline: bool = False, enabled: bool = True):
    def decorator(function):
        if enabled:
            operator_name = function.__module__.split(".")[-1]
            if not operator_name in REGISTERED_BENCHMARKS:
                REGISTERED_BENCHMARKS[operator_name] = []
            REGISTERED_BENCHMARKS[operator_name].append(function.__name__)
            if baseline:
                BASELINE_BENCHMARKS[operator_name] = function.__name__

        def _inner(self, *args, **kwargs):
            return function(self, *args, **kwargs)
        return _inner

    return decorator


def register_metric(
    # Metrics that only apply to non-baseline impls
    # E.g., accuracy, speedup
    skip_baseline: bool = False,
    # Metrics that are the same across all impls
    # E.g., x_shape, hw_roofline
    x_only: bool = False
):
    def decorator(func):
        operator_name = func.__module__.split(".")[-1]
        if not operator_name in REGISTERED_METRICS:
            REGISTERED_METRICS[operator_name] = []
        REGISTERED_METRICS[operator_name].append(func.__name__)
        if skip_baseline:
            BASELINE_SKIP_METRICS.add(func.__name__)
        if x_only:
            X_ONLY_METRICS.add(func.__name__)

        def _inner(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return _inner

    return decorator


def parse_args(
    default_metrics: List[str], args: List[str]
) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default=",".join(default_metrics),
        help="Metrics to collect, split with comma. E.g., --metrics latency,tflops,speedup.",
    )
    parser.add_argument("--only", default=None, help="Run only the specific benchmark.")
    parser.add_argument("--batch-id", type=int, default=None, help="Run only the specific batch id.")
    return parser.parse_known_args(args)


class BenchmarkOperator:
    mode: Mode = Mode.FWD
    test: str = "eval"
    device: str = "cuda"
    _input_iter: Optional[Generator] = None
    extra_args: List[str] = []
    example_inputs: Any = None

    # By default, only collect latency metrics
    # Each operator can override to define their own default metrics
    DEFAULT_METRICS = ["latency"]
    # By default, generate 100 data points
    DEFAULT_NUM_BATCH = 100

    """
    A base class for adding operators to torch benchmark.
    """

    def __init__(self, mode: str, device: str, extra_args: List[str] = []):
        relative_path = self.__class__.__module__.split(".")
        set_random_seed()
        self.name = relative_path[-1]
        self._raw_extra_args = copy.deepcopy(extra_args)
        # we accept both "fwd" and "eval"
        if mode == "fwd":
            self.mode = Mode.FWD
        elif mode == "fwd_bwd":
            self.mode = Mode.FWD_BWD
        else:
            assert (
                mode == "bwd"
            ), f"We only accept 3 test modes: fwd(eval), fwd_bwd(train), or bwd."
            self.mode = Mode.BWD
        self.dargs, unprocessed_args = parse_decoration_args(self, extra_args)
        # This will be changed by the time we apply the decoration args
        self.dtype = PRECISION_DTYPE_MAPPING.get(self.dargs.precision, None)
        if self.dargs.num_batch is None:
            self.dargs.num_batch = self.DEFAULT_NUM_BATCH
        self.DEFAULT_METRICS.extend(REGISTERED_METRICS.get(self.name, []))
        self.DEFAULT_METRICS = list(set(self.DEFAULT_METRICS))
        self.tb_args, self.extra_args = parse_args(
            self.DEFAULT_METRICS, unprocessed_args
        )
        self.required_metrics = list(set(self.tb_args.metrics.split(",")))
        self._only = self.tb_args.only
        self._batch_id = self.tb_args.batch_id

    def _get_bm_func(self, bm_func_name: str):
        fwd_fn_lambda = getattr(self, bm_func_name, None)
        assert (
            fwd_fn_lambda
        ), f"Could not find benchmark {bm_func_name} registered in {self.name}. " \
            f"Available benchmarks: {REGISTERED_BENCHMARKS[self.name]}. "
        if isinstance(self.example_inputs, dict):
            fwd_fn = fwd_fn_lambda(**self.example_inputs)
        else:
            fwd_fn = fwd_fn_lambda(*self.example_inputs)
        if self.mode == Mode.FWD:
            setattr(fwd_fn, "_name", bm_func_name)
            return fwd_fn
        elif self.mode == Mode.BWD:
            bwd_fn = self.get_bwd_fn(fwd_fn)
            setattr(bwd_fn, "_name", bm_func_name)
            return bwd_fn
        elif self.mode == Mode.FWD_BWD:
            bwd_fn = self.get_bwd_fn(fwd_fn)
            fwd_bwd_fn = lambda: (fwd_fn(), bwd_fn())
            setattr(fwd_bwd_fn, "_name", bm_func_name)
            return fwd_bwd_fn

    def run(
        self, warmup=DEFAULT_WARMUP, rep=DEFAULT_RUN_ITERS, quantiles=DEFAULT_QUANTILES
    ) -> BenchmarkOperatorResult:
        """Benchmarking the operator and returning its metrics."""
        metrics = []
        if self._batch_id is not None:
            # Run only the user-specific batch id
            batch_range = range(self._batch_id + 1)
        else:
            batch_range = range(self.dargs.num_batch)
        for batch_id in batch_range:
            if self._batch_id and batch_id < self._batch_id:
                continue
            self.example_inputs = self.get_example_inputs()
            if self.example_inputs is None:
                warnings.warn(
                    UserWarning(
                        f"The input generator get_input_iter() has depleted. Maximum input batches {batch_id}."
                    )
                )
                break
            # Move inputs to the device
            self.example_inputs = input_cast(
                lambda x: isinstance(x, torch.Tensor),
                lambda x: x.to(self.device),
                self.example_inputs,
            )
            self.baseline_fn = None
            self.baseline_metrics = None
            self._op_flops = {}
            # Cast the input precisions
            apply_decoration_args(self, self.dargs)
            x_val = self.get_x_val(self.example_inputs)
            benchmarks = [ bm for bm in REGISTERED_BENCHMARKS[self.name] ] \
                if self.name in REGISTERED_BENCHMARKS else []
            # Run the baseline first, if baseline exists
            baseline_name = BASELINE_BENCHMARKS[self.name] \
                     if self.name in BASELINE_BENCHMARKS else None
            if baseline_name and baseline_name in benchmarks:
                benchmarks.remove(baseline_name)
                benchmarks.insert(0, baseline_name)
            if self._only:
                benchmarks = [self._only]
            # get metrics for for each registered benchmark
            def _reduce_benchmarks(acc, bm_name: str):
                baseline = bm_name == BASELINE_BENCHMARKS[self.name] \
                     if self.name in BASELINE_BENCHMARKS else False
                acc[bm_name] = self._do_bench(
                    batch_id=batch_id,
                    fn_name=bm_name,
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                    baseline=baseline,
                )
                if baseline:
                    self.baseline_metrics = acc[bm_name]
                return acc

            y_vals: Dict[str, BenchmarkOperatorMetrics] = functools.reduce(
                _reduce_benchmarks, benchmarks, {}
            )
            metrics.append((x_val, y_vals))
            del self.example_inputs
            gc.collect()
        self.output = BenchmarkOperatorResult(
            op_name=self.name,
            metrics=self.required_metrics,
            result=metrics,
        )
        return self.output

    def get_x_val(self, example_inputs) -> float:
        raise NotImplementedError(
            "Each operator must implement its own input to x_val mapping."
        )

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        raise NotImplementedError(
            "Each operator must implement its own backward function."
        )

    def get_input_iter(self) -> Generator:
        """Return the dynamic input iterator for the model."""
        raise NotImplementedError(
            "Each operator must implement its own input iterator."
        )

    def get_grad_to_none(self, args):
        return None

    def plot(self):
        """Plot the comparison between different operator implementations."""
        raise NotImplementedError(
            "Each operator must implement its own plotting logic."
        )

    def enable_bf16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.to(torch.bfloat16)
        self.dtype = torch.bfloat16
        self.example_inputs = input_cast(
            tensor_cond, tensor_action, self.example_inputs
        )

    def enable_fp16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.half()
        self.dtype = torch.float16
        self.example_inputs = input_cast(
            tensor_cond, tensor_action, self.example_inputs
        )

    # a function copied from https://fburl.com/code/hdypvhjw, which generate offsets
    # for jagged tensors with the given load_factor
    def generate_offsets(
        self,
        batch_size: int,
        max_seq_len: int,
        load_factor: float,
        offsets_dtype: torch.dtype,
    ) -> torch.Tensor:
        total_length = int(batch_size * max_seq_len * load_factor)
        avg_length = total_length // batch_size
        std = avg_length // 3  # rather arbitrary, but likely reasonable
        lengths = [random.gauss(avg_length, std) for _ in range(batch_size)]
        lengths = [int(min(max_seq_len, max(L, 0))) for L in lengths]

        if load_factor == 1.0:
            lengths = [max_seq_len] * batch_size

        diff = sum(lengths) - total_length
        idx_and_lengths = list(enumerate(lengths))
        random.shuffle(idx_and_lengths)

        for i, length in idx_and_lengths:
            if diff == 0:
                break
            elif diff > 0:
                delta = min(length, diff)
                lengths[i] -= delta
                diff -= delta
            else:
                delta = min(max_seq_len - length, -diff)
                lengths[i] += delta
                diff += delta

        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        return torch.tensor(
            offsets,
            dtype=offsets_dtype,
        )

    def enable_channels_last(self):
        tensor_cond = lambda x: x.dim() == 4
        tensor_action = lambda x: x.to(memory_format=torch.channels_last)
        self.example_inputs = input_cast(
            tensor_cond, tensor_action, self.example_inputs
        )

    def get_example_inputs(self):
        if self._input_iter is None:
            self._input_iter = self.get_input_iter()
        try:
            return next(self._input_iter)
        except StopIteration:
            return None

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        accuracy = True
        try:
            if self.mode == Mode.FWD:
                torch.testing.assert_close(output, baseline_output)
            elif self.mode == Mode.BWD:
                torch.testing.assert_close(output.grad, baseline_output.grad)
            else:
                fwd_output, loss = output
                baseline_fwd_output, baseline_loss = baseline_output
                torch.testing.assert_close(fwd_output, baseline_fwd_output)
                torch.testing.assert_close(loss.grad, baseline_loss.grad)
        except Exception:
            # either the output tensor or the loss grad tensor does not match
            accuracy = False
        finally:
            return accuracy

    def _do_bench(
        self,
        batch_id: int,
        fn_name: str,
        warmup=DEFAULT_WARMUP,
        rep=DEFAULT_RUN_ITERS,
        quantiles=DEFAULT_QUANTILES,
        baseline: bool = False,
    ) -> BenchmarkOperatorMetrics:
        latency = []
        speedup = None
        accuracy = None
        walltime = None
        cpu_peak_mem = None
        gpu_peak_mem = None
        error_msg = None
        hw_roofline = None
        try:
            fn = self._get_bm_func(fn_name)
            if baseline:
                self.baseline_fn = fn
            if set(["latency", "tflops", "speedup", "compile_time"]) & set(self.required_metrics):
                latency = triton.testing.do_bench(
                    fn,
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                )
            if "walltime" in self.required_metrics:
                walltime = do_bench_walltime(
                    fn,
                    warmup=warmup,
                    rep=rep,
                )
            if "speedup" in self.required_metrics:
                speedup = (
                    numpy.median(self.baseline_metrics.latency) / numpy.median(latency)
                    if self.baseline_metrics and self.baseline_metrics.latency
                    else None
                )
                error_msg = (
                    self.baseline_metrics.error_msg
                    if self.baseline_metrics and self.baseline_metrics.error_msg
                    else None
                )
            if "cpu_peak_mem" in self.required_metrics or "gpu_peak_mem" in self.required_metrics:
                cpu_peak_mem, _device_id, gpu_peak_mem = self.get_peak_mem(fn)
            if not baseline and "accuracy" in self.required_metrics:
                accuracy = (
                    self._get_accuracy(fn, self.baseline_fn)
                    if self.baseline_fn
                    else None
                )
            if "hw_roofline" in self.required_metrics:
                hw_roofline = self.hw_roofline()
            metric = BenchmarkOperatorMetrics(
                latency=latency,
                tflops=None,
                speedup=speedup,
                accuracy=accuracy,
                walltime=walltime,
                compile_time=None,
                ncu_trace=None,
                cpu_peak_mem=cpu_peak_mem,
                gpu_peak_mem=gpu_peak_mem,
                hw_roofline=hw_roofline,
                error_msg=error_msg,
                extra_metrics={},
            )
            if "tflops" in self.required_metrics:
                metric.tflops = self.tflops(fn_name, self.example_inputs, metric)
            if "compile_time" in self.required_metrics:
                metric.compile_time = self.compile_time(batch_id, fn_name, metric)
            if "ncu_trace" in self.required_metrics:
                metric.ncu_trace = self.ncu_trace(batch_id, fn_name)
            extra_metrics = {}
            # run the hidden metric "_compile_time_in_task"
            # to get the compile time in parent process
            if "_compile_time_in_task" in self.required_metrics:
                assert self.required_metrics == ["_compile_time_in_task"] and self._only and (self._batch_id is not None), \
                    "_compile_time_in_task must be measured by itself. " \
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _batch_id: {self._batch_id}"
                extra_metrics["_compile_time_in_task"] = self._compile_time_in_task(fn)
            if "_ncu_trace_in_task" in self.required_metrics:
                assert self.required_metrics == ["_ncu_trace_in_task"] and self._only and (self._batch_id is not None), \
                    "_ncu_trace_in_task must be measured by itself. " \
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _batch_id: {self._batch_id}"
                from torchbenchmark._components.ncu import do_bench_ncu_in_task
                do_bench_ncu_in_task(fn=fn, warmup=warmup, grad_to_none=self.get_grad_to_none(self.example_inputs))
                extra_metrics["_ncu_trace_in_task"] = "success"
            # generate customized metrics
            if self.name in REGISTERED_METRICS:
                for metric_name in REGISTERED_METRICS[self.name]:
                    if metric_name in BUILTIN_METRICS:
                        continue
                    if not metric_name in self.required_metrics:
                        continue
                    func = getattr(self, metric_name)
                    extra_metrics[metric_name] = func(fn, self.example_inputs, metric)
                metric.extra_metrics = extra_metrics
        except torch.cuda.OutOfMemoryError:
            metric = BenchmarkOperatorMetrics(
                latency=None,
                tflops=None,
                speedup=None,
                accuracy=None,
                walltime=None,
                compile_time=None,
                ncu_trace=None,
                hw_roofline=self.hw_roofline(),
                cpu_peak_mem=None,
                gpu_peak_mem=None,
                error_msg="CUDA OOM",
                extra_metrics={},
            )
        return metric

    def get_peak_mem(self, fn: Callable) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        return get_peak_memory(
            func=fn,
            device=self.device,
            metrics_needed=["gpu_peak_mem", "cpu_peak_mem"],
            metrics_gpu_backend="nvml",
        )

    @register_metric()
    def ncu_trace(self, batch_id: int, fn_name: str) -> str:
        # collect the ncu trace
        import sys
        import subprocess
        from pathlib import Path
        op_task_args = copy.deepcopy(sys.argv)
        for override_option in ["--only", "--batch-id", "--metrics"]:
            op_task_args = _remove_params(op_task_args, _find_param_loc(op_task_args, override_option))
        op_task_args.extend(["--only", fn_name, "--batch-id", str(batch_id), "--metrics", "_ncu_trace_in_task"])
        # Disable DCGM
        try:
            disable_dcgm = ["sudo", "dyno", "dcgm_profiling", "--mute=true", "--duration=1000_s"]
            subprocess.run(disable_dcgm, check=True)
        except subprocess.SubprocessError:
            warnings.warn("Cannot find dyno to disable DCGM. Proceed to collect NCU Trace.")
        ncu_output_dir = Path(f"/tmp/tritonbench_{self.name}_{fn_name}_{batch_id}")
        ncu_output_dir.mkdir(parents=True, exist_ok=True)
        ncu_output_file = ncu_output_dir.joinpath("ncu_output.csv").resolve()
        ncu_args = ["ncu", "--set", "full", "--replay-mode", "range", "--target-processes", "all", \
                    "--csv", "-f", "--log-file", str(ncu_output_file.resolve())]
        ncu_args.extend(op_task_args)
        subprocess.check_call(ncu_args)
        return str(ncu_output_file.resolve())


    @register_metric()
    def compile_time(self, batch_id: int, fn_name: str, metrics: BenchmarkOperatorMetrics) -> float:
        # We need to spawn a subprocess when user wants to measure the compile time
        # of multiple batches and backends.
        from torchbenchmark.operators.op_task import OpTask
        op_task_args = copy.deepcopy(self._raw_extra_args)
        for override_option in ["--only", "--batch-id", "--metrics"]:
            op_task_args = _remove_params(op_task_args, _find_param_loc(op_task_args, override_option))
        op_task_args.extend(["--only", fn_name, "--batch-id", str(batch_id), "--metrics", "_compile_time_in_task"])
        op_task = OpTask(name=self.name)
        op_task.make_operator_instance(mode=self.mode.value, device=self.device, extra_args=op_task_args)
        op_task.run()
        latency_with_compile = op_task.get_attribute("_latency_with_compile_in_task")
        del op_task
        latency_without_compile = numpy.median(metrics.latency)
        return latency_with_compile - latency_without_compile

    @register_metric(x_only=True)
    def hw_roofline(self) -> float:
        """Hardware roofline in tflops."""
        from torchbenchmark.util.hardware import HW_ROOFLINE_SPECS
        device_name = torch.cuda.get_device_name()
        assert device_name in HW_ROOFLINE_SPECS, f"{device_name} is not supported in HW roofline specs."
        assert self.dargs.precision in HW_ROOFLINE_SPECS[device_name], f"{self.precision} is not supported for {device_name}."
        return HW_ROOFLINE_SPECS[device_name][self.dargs.precision]


    def _compile_time_in_task(
        self,
        fn: Callable,
    ) -> float:
        with fresh_triton_cache():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
        latency_with_compile = start_event.elapsed_time(end_event)
        self._latency_with_compile_in_task = latency_with_compile
        return latency_with_compile

    @register_metric()
    def tflops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> List[float]:
        def _get_flops(self, func: Callable) -> float:
            """By default, use the torch.__dispatch__ based flops counter."""
            from torch.utils.flop_counter import FlopCounterMode

            flop_counter = FlopCounterMode()

            def work_func():
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    func()
                    torch.cuda.synchronize()
                else:
                    func()

            with flop_counter:
                work_func()
            total_flops = sum(
                [v for _, v in flop_counter.flop_counts["Global"].items()]
            )
            return total_flops

        fn = self._get_bm_func(fn_name)
        if not fn in self._op_flops:
            self._op_flops[fn] = _get_flops(self, fn)
        op_flops = self._op_flops[fn]
        return list(map(lambda x: op_flops / x / 1e12 * 1e3, metrics.latency))
