import argparse
import copy
import functools
import gc
import json
import os
import random
import time
import warnings
from collections import OrderedDict
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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DEFAULT_WARMUP = 25
DEFAULT_RUN_ITERS = 100
DEFAULT_QUANTILES = [0.5, 0.1, 0.9]
REGISTERED_BENCHMARKS: Dict[str, OrderedDict[str, str]] = {}
ENABLED_BENCHMARKS: Dict[str, List[str]] = {}
REGISTERED_METRICS: Dict[str, List[str]] = {}
REGISTERED_X_VALS: Dict[str, str] = {}
BASELINE_BENCHMARKS: Dict[str, str] = {}
BUILTIN_METRICS = [
    "latency",
    "tflops",
    "speedup",
    "accuracy",
    "compile_time",
    "ncu_trace",
    "ncu_rep",
    "kineto_trace",
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
    return l[:loc] + l[loc + 2 :]

def _split_params_by_comma(params: Optional[str]) -> List[str]:
    if params == None:
        return []
    return [x.strip() for x in params.split(",")] if "," in params else [params]

def _find_op_name_from_module_path(module_path: str) -> str:
    PATH_PREFIX = "torchbenchmark.operators."
    assert PATH_PREFIX in module_path, \
        f"We rely on module path prefix to identify operator name. Expected {PATH_PREFIX}<operator_name>, get {module_path}."
    suffix = module_path.partition(PATH_PREFIX)[2]
    if suffix.startswith("fb."):
        return suffix.split(".")[1]
    return suffix.split(".")[0]

def dump_autotuner_best_config(kernel: triton.runtime.Autotuner) -> str:
    if not hasattr(kernel, "best_config"):
        return ""
    # pyre-ignore: Undefined attribute [16]
    bconfig = kernel.best_config
    kwargs = copy.deepcopy(bconfig.kwargs)
    kwargs["num_stages"] = bconfig.num_stages
    kwargs["num_warps"] = bconfig.num_warps
    dumped_str = json.dumps(kwargs)
    return dumped_str


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
    # ncu replay file
    ncu_rep: Optional[str]
    # kineto trace file
    kineto_trace: Optional[str]
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
        headers = [REGISTERED_X_VALS[self.op_name]]
        y_val = self.result[0][1]
        y_val_keys = list(y_val.keys())
        # move the baseline benchmark to the front of the list if exists
        if self.op_name in BASELINE_BENCHMARKS and BASELINE_BENCHMARKS[self.op_name] in y_val_keys:
            y_val_keys.insert(
                0, y_val_keys.pop(y_val_keys.index(BASELINE_BENCHMARKS[self.op_name]))
            )
        y_val_keys = [(x, REGISTERED_BENCHMARKS[self.op_name][x]) for x in y_val_keys]
        key_metrics = {}
        # Add header for x_only_metrics
        x_only_metrics = sorted(
            [metric for metric in self.metrics if metric in X_ONLY_METRICS]
        )
        headers.extend(x_only_metrics)
        for k, label in y_val_keys:
            def select_metric(m):
                if m in x_only_metrics:
                    return False
                if m in BASELINE_SKIP_METRICS and k == BASELINE_BENCHMARKS[self.op_name]:
                    return False
                return True

            key_metrics[k] = sorted(filter(select_metric, self.metrics))
            for metric in key_metrics[k]:
                # add extra metrics
                headers.append(f"{label}-{metric}")
        # generate rows
        for x_val, y_val in self.result:
            row = []
            row.append(x_val)
            # Append x_val_only metrics
            for x_only_metric in x_only_metrics:
                x_only_metric_dict = asdict(y_val[y_val_keys[0]])
                if "extra_metrics" in x_only_metric_dict and x_only_metric in x_only_metric_dict["extra_metrics"]:
                    row.append(x_only_metric_dict["extra_metrics"][x_only_metric])
                else:
                    row.append(x_only_metric_dict[x_only_metric])
            for k, _label in y_val_keys:
                metrics_dict = asdict(y_val[k])
                if metrics_dict["error_msg"]:
                    row.append(metrics_dict["error_msg"])
                    row.extend([None] * (len(key_metrics[k]) - 1))
                    continue
                for metric in key_metrics[k]:
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

    def write_csv_to_file(self, fileobj):
        import csv

        headers, table = self._table()
        writer = csv.writer(fileobj, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(table)

    def write_csv(self, dir_path):
        import tempfile

        # This is just a way to create a unique filename. It's not actually a
        # temporary file (since delete=False).
        with tempfile.NamedTemporaryFile(
            mode='w',
            prefix=os.path.join(dir_path, f"op_{self.op_name}_"),
            suffix=".csv",
            newline="",
            delete=False,
        ) as fileobj:
            self.write_csv_to_file(fileobj)
            return fileobj.name

    @property
    def x_vals(self):
        return sorted(self._get_result_dict().keys())

    @property
    def userbenchmark_dict(self) -> Dict[str, Any]:
        # Userbenchmark Metric key format:
        # tritonbench_{op_name}[{x_val}-{provider}-{metric}]
        userbenchmark_metrics_dict = {}
        headers, table = self._table()
        for row in table:
            x_val = row[0]
            for ind, value in enumerate(row[1:]):
                header = headers[ind+1]
                provider, _dash, metrics = header.partition("-")
                metric_name = f"tritonbench_{self.op_name}[x_{x_val}-{provider}]_{metrics}"
                userbenchmark_metrics_dict[metric_name] = value
        return userbenchmark_metrics_dict

    def get_y_vals(self, x_val, provider, metric_name: str):
        if provider in X_ONLY_METRICS:
            maybe_baseline = list(REGISTERED_BENCHMARKS[self.op_name].keys())[0]
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

def register_x_val(label: str="x_val"):
    def decorator(function):
        operator_name = _find_op_name_from_module_path(function.__module__)
        REGISTERED_X_VALS[operator_name] = label
        def _inner(self, *args, **kwargs):
            return function(self, *args, **kwargs)
        return _inner
    return decorator

def register_benchmark(baseline: bool = False, enabled: bool = True, label: Optional[str] = None):
    def decorator(function):
        operator_name = _find_op_name_from_module_path(function.__module__)
        if not operator_name in REGISTERED_BENCHMARKS:
            REGISTERED_BENCHMARKS[operator_name] = OrderedDict()
        REGISTERED_BENCHMARKS[operator_name][function.__name__] = function.__name__ if not label else label
        if baseline:
            BASELINE_BENCHMARKS[operator_name] = function.__name__
        if enabled:
            if not operator_name in ENABLED_BENCHMARKS:
                ENABLED_BENCHMARKS[operator_name] = []
            ENABLED_BENCHMARKS[operator_name].append(function.__name__)

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
    x_only: bool = False,
):
    def decorator(func):
        metric_name = func.__name__
        if not metric_name in BUILTIN_METRICS:
            operator_name = _find_op_name_from_module_path(func.__module__)
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
    default_metrics: List[str],
    args: List[str],
) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--metrics",
        default=",".join(default_metrics),
        help="Metrics to collect, split with comma. E.g., --metrics latency,tflops,speedup.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Specify one or multiple operator implementations to run."
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        help="Number of example inputs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
    )
    parser.add_argument(
        "--input-id",
        type=int,
        default=0,
        help="Specify the start input id to run. " \
            "For example, --input-id 0 runs only the first available input sample." \
            "When used together like --input-id <X> --num-inputs <Y>, start from the input id <X> " \
            "and run <Y> different inputs."
    )
    return parser.parse_known_args(args)

class PostInitProcessor(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj

class BenchmarkOperator(metaclass=PostInitProcessor):
    mode: Mode = Mode.FWD
    test: str = "eval"
    device: str = "cuda"
    _input_iter: Optional[Generator] = None
    extra_args: List[str] = []
    example_inputs: Any = None

    # By default, only collect latency metrics
    # Each operator can override to define their own default metrics
    DEFAULT_METRICS = ["latency"]

    """
    A base class for adding operators to torch benchmark.
    """

    def __init__(self, mode: str, device: str, extra_args: Optional[List[str]]=None):
        set_random_seed()
        self.name = _find_op_name_from_module_path(self.__class__.__module__)
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
        if self.name not in REGISTERED_X_VALS:
            REGISTERED_X_VALS[self.name] = "x_val"
        # This will be changed by the time we apply the decoration args
        self.dtype = PRECISION_DTYPE_MAPPING.get(self.dargs.precision, None)
        self.DEFAULT_METRICS.extend(
            [x for x in REGISTERED_METRICS.get(self.name, []) if x not in BUILTIN_METRICS]
        )
        self.DEFAULT_METRICS = list(set(self.DEFAULT_METRICS))
        self.tb_args, self.extra_args = parse_args(
            self.DEFAULT_METRICS,
            unprocessed_args
        )
        self.required_metrics = list(set(self.tb_args.metrics.split(",")))
        self._only = _split_params_by_comma(self.tb_args.only)
        self._input_id = self.tb_args.input_id
        self._num_inputs = self.tb_args.num_inputs

    # Run the post initialization
    def __post__init__(self):
        self._available_num_inputs = self.count_example_inputs()
        if self._num_inputs is None:
            self._num_inputs = self._available_num_inputs - self._input_id

    def _get_bm_func(self, bm_func_name: str):
        fwd_fn_lambda = getattr(self, bm_func_name, None)
        assert fwd_fn_lambda, (
            f"Could not find benchmark {bm_func_name} registered in {self.name}. "
            f"Available benchmarks: {REGISTERED_BENCHMARKS[self.name].keys()}. "
        )
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
    ) -> None:
        """Benchmarking the operator and returning its metrics."""
        metrics = []
        try:
            input_id_range = range(self._input_id, self._input_id + self._num_inputs)
            if tqdm is not None:
                input_id_range = tqdm(input_id_range)
            if self._input_id:
                for _dryrun_input_id in range(self._input_id):
                    self.example_inputs = self.get_example_inputs()
            for input_id in input_id_range:
                self.example_inputs = self.get_example_inputs()
                if self.example_inputs is None:
                    warnings.warn(
                        f"The input generator get_input_iter() has depleted at id {input_id}. Available number of "
                        f"inputs: {self._available_num_inputs}.",
                        stacklevel=1
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
                if self._only:
                    benchmarks = self._only
                else:
                    benchmarks = (
                        [bm for bm in ENABLED_BENCHMARKS[self.name]]
                        if self.name in ENABLED_BENCHMARKS
                        else []
                    )
                    # Run the baseline first, if baseline exists
                    baseline_name = (
                        BASELINE_BENCHMARKS[self.name]
                        if self.name in BASELINE_BENCHMARKS
                        else None
                    )
                    if baseline_name and baseline_name in benchmarks:
                        benchmarks.remove(baseline_name)
                        benchmarks.insert(0, baseline_name)

                # get metrics for for each registered benchmark
                def _reduce_benchmarks(acc, bm_name: str):
                    baseline = (
                        bm_name == BASELINE_BENCHMARKS[self.name]
                        if self.name in BASELINE_BENCHMARKS
                        else False
                    )
                    acc[bm_name] = self._do_bench(
                        input_id=input_id,
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
        except (KeyboardInterrupt, Exception):
            warnings.warn("Caught exception, terminating early with partial results", stacklevel=1)
            raise
        finally:
            self.output = BenchmarkOperatorResult(
                op_name=self.name,
                metrics=self.required_metrics,
                result=metrics,
            )

    def get_x_val(self, example_inputs) -> Any:
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

    def count_example_inputs(self):
        return sum(1 for _ in  self.get_input_iter())

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
        input_id: int,
        fn_name: str,
        warmup=DEFAULT_WARMUP,
        rep=DEFAULT_RUN_ITERS,
        quantiles=DEFAULT_QUANTILES,
        baseline: bool = False,
    ) -> BenchmarkOperatorMetrics:
        def _init_extra_metrics() -> Dict[str, Any]:
            extra_metrics = {}
            if self.name in REGISTERED_METRICS:
                for metric_name in REGISTERED_METRICS[self.name]:
                    if metric_name in BUILTIN_METRICS:
                        continue
                    if metric_name not in self.required_metrics:
                        continue
                    extra_metrics[metric_name] = None
            return extra_metrics
        metrics = BenchmarkOperatorMetrics(
            latency=None,
            tflops=None,
            speedup=None,
            accuracy=None,
            walltime=None,
            compile_time=None,
            ncu_trace=None,
            ncu_rep=None,
            hw_roofline=self.hw_roofline() if "hw_roofline" in self.required_metrics else None,
            kineto_trace=None,
            cpu_peak_mem=None,
            gpu_peak_mem=None,
            error_msg="",
            extra_metrics=_init_extra_metrics(),
        )
        try:
            fn = self._get_bm_func(fn_name)
            if baseline:
                self.baseline_fn = fn
            if set(["latency", "tflops", "speedup", "compile_time"]) & set(
                self.required_metrics
            ):
                metrics.latency = triton.testing.do_bench(
                    fn,
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                )
            if "walltime" in self.required_metrics:
                metrics.walltime = do_bench_walltime(
                    fn,
                    warmup=warmup,
                    rep=rep,
                )
            if "speedup" in self.required_metrics:
                metrics.speedup = (
                    numpy.median(self.baseline_metrics.latency) / numpy.median(metrics.latency)
                    if self.baseline_metrics and self.baseline_metrics.latency
                    else None
                )
                metrics.error_msg = (
                    self.baseline_metrics.error_msg
                    if self.baseline_metrics and self.baseline_metrics.error_msg
                    else None
                )
            if (
                "cpu_peak_mem" in self.required_metrics
                or "gpu_peak_mem" in self.required_metrics
            ):
                metrics.cpu_peak_mem, _device_id, metrics.gpu_peak_mem = self.get_peak_mem(fn)
            if not baseline and "accuracy" in self.required_metrics:
                metrics.accuracy = (
                    self._get_accuracy(fn, self.baseline_fn)
                    if self.baseline_fn
                    else None
                )
            if "hw_roofline" in self.required_metrics:
                metrics.hw_roofline = self.hw_roofline()
            if "tflops" in self.required_metrics:
                metrics.tflops = self.tflops(fn_name, self.example_inputs, metrics)
            if "compile_time" in self.required_metrics:
                metrics.compile_time = self.compile_time(input_id, fn_name, metrics)
            if "ncu_trace" in self.required_metrics:
                metrics.ncu_trace = self.ncu_trace(input_id, fn_name)
            if "ncu_rep" in self.required_metrics:
                metrics.ncu_rep = self.ncu_trace(input_id, fn_name, replay=True)
            if "kineto_trace" in self.required_metrics:
                metrics.kineto_trace = self.kineto_trace(input_id, fn)
            # run the hidden metric "_compile_time_in_task"
            # to get the compile time in parent process
            if "_compile_time_in_task" in self.required_metrics:
                assert (
                    self.required_metrics == ["_compile_time_in_task"]
                    and len(self._only) == 1
                    and (self._input_id is not None)
                ), (
                    "_compile_time_in_task must be measured by itself. "
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _input_id: {self._input_id}"
                )
                metrics.extra_metrics["_compile_time_in_task"] = self._compile_time_in_task(fn)
            if "_ncu_trace_in_task" in self.required_metrics:
                assert (
                    self.required_metrics == ["_ncu_trace_in_task"]
                    and len(self._only) == 1
                    and (self._input_id is not None)
                ), (
                    "_ncu_trace_in_task must be measured by itself. "
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _input_id: {self._input_id}"
                )
                from torchbenchmark._components.ncu import do_bench_ncu_in_task

                do_bench_ncu_in_task(
                    fn=fn,
                    warmup=warmup,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                )
                metrics.extra_metrics["_ncu_trace_in_task"] = "success"
            # generate customized metrics
            if self.name in REGISTERED_METRICS:
                for metric_name in REGISTERED_METRICS[self.name]:
                    if metric_name in BUILTIN_METRICS:
                        continue
                    if not metric_name in self.required_metrics:
                        continue
                    func = getattr(self, metric_name)
                    metrics.extra_metrics[metric_name] = func(fn, self.example_inputs, metrics)
        except torch.cuda.OutOfMemoryError:
            metrics.error_msg = "CUDA OOM"
        except Exception as e:
            if not self.tb_args.keep_going:
                raise
            metrics.error_msg = str(e)
        return metrics

    def get_peak_mem(
        self, fn: Callable
    ) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        return get_peak_memory(
            func=fn,
            device=self.device,
            metrics_needed=["gpu_peak_mem", "cpu_peak_mem"],
            metrics_gpu_backend="nvml",
        )

    def ncu_trace(self, input_id: int, fn_name: str, replay: bool=False) -> str:
        # collect the ncu trace
        import sys
        import subprocess
        from pathlib import Path

        op_task_args = copy.deepcopy(sys.argv)
        for override_option in ["--only", "--input-id", "--num-inputs", "--metrics"]:
            op_task_args = _remove_params(
                op_task_args, _find_param_loc(op_task_args, override_option)
            )
        op_task_args.extend(
            [
                "--only",
                fn_name,
                "--num-inputs",
                str(1),
                "--input-id",
                str(input_id),
                "--metrics",
                "_ncu_trace_in_task",
            ]
        )
        # Disable DCGM
        try:
            disable_dcgm = [
                "sudo",
                "dyno",
                "dcgm_profiling",
                "--mute=true",
                "--duration=1000_s",
            ]
            subprocess.run(disable_dcgm, check=True)
        except subprocess.SubprocessError:
            warnings.warn(
                "Cannot find dyno to disable DCGM. Proceed to collect NCU Trace."
            )
        ncu_output_dir = Path(f"/tmp/tritonbench_{self.name}_{fn_name}_{input_id}")
        ncu_output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".csv" if not replay else ".ncu-rep"
        ncu_output_file = ncu_output_dir.joinpath(f"ncu_output{ext}").resolve()
        ncu_args = [
            "ncu",
            "--set",
            "full",
            "--replay-mode",
            "kernel",
            "--target-processes",
            "all",
            "--csv",
            "-f",
            "--log-file",
            str(ncu_output_file.resolve()),
        ] if not replay else [
            "ncu",
            "--set",
            "full",
            "--replay-mode",
            "kernel",
            "--target-processes",
            "all",
            "-f",
            "-o",
            str(ncu_output_file.resolve()),
        ]
        ncu_args.extend(op_task_args)
        subprocess.check_call(ncu_args)
        return str(ncu_output_file.resolve())

    def kineto_trace(self, input_id: int, fn: Callable) -> str:
        from pathlib import Path
        from torchbenchmark._components.kineto import do_bench_kineto

        kineto_output_dir = Path(f"/tmp/tritonbench_{self.name}_{fn._name}_{input_id}")
        kineto_output_dir.mkdir(parents=True, exist_ok=True)
        return do_bench_kineto(
            fn=fn,
            grad_to_none=self.get_grad_to_none(self.example_inputs),
            output_dir=kineto_output_dir,
        )

    def compile_time(
        self, input_id: int, fn_name: str, metrics: BenchmarkOperatorMetrics
    ) -> float:
        # We need to spawn a subprocess when user wants to measure the compile time
        # of multiple sample inputs and backends.
        from torchbenchmark.operators.op_task import OpTask

        op_task_args = copy.deepcopy(self._raw_extra_args)
        for override_option in ["--only", "--input-id", "--num-inputs", "--metrics"]:
            op_task_args = _remove_params(
                op_task_args, _find_param_loc(op_task_args, override_option)
            )
        op_task_args.extend(
            [
                "--only",
                fn_name,
                "--num-inputs",
                str(1),
                "--input-id",
                str(input_id),
                "--metrics",
                "_compile_time_in_task",
            ]
        )
        op_task = OpTask(name=self.name)
        op_task.make_operator_instance(
            mode=self.mode.value, device=self.device, extra_args=op_task_args
        )
        op_task.run()
        latency_with_compile = op_task.get_attribute("_latency_with_compile_in_task")
        del op_task
        latency_without_compile = numpy.median(metrics.latency)
        return latency_with_compile - latency_without_compile

    def hw_roofline(self) -> float:
        """Hardware roofline in tflops."""
        from torchbenchmark.util.hardware import HW_ROOFLINE_SPECS

        device_name = torch.cuda.get_device_name()
        assert (
            device_name in HW_ROOFLINE_SPECS
        ), f"{device_name} is not supported in HW roofline specs."
        assert (
            self.dargs.precision in HW_ROOFLINE_SPECS[device_name]
        ), f"{self.precision} is not supported for {device_name}."
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
