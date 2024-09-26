import argparse
import copy
import functools
import gc
import json
import logging
import os
import random
import shlex
import tempfile
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, make_dataclass
from enum import Enum
from itertools import product
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

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

logger = logging.getLogger(__name__)

IS_FBCODE = not hasattr(torch.version, "git_version")
DEFAULT_WARMUP = 25
DEFAULT_RUN_ITERS = 100
DEFAULT_QUANTILES = [0.5, 0.1, 0.9]
REGISTERED_BENCHMARKS: Dict[str, OrderedDict[str, str]] = {}
ENABLED_BENCHMARKS: Dict[str, List[str]] = {}
REGISTERED_METRICS: Dict[str, List[str]] = {}
REGISTERED_X_VALS: Dict[str, str] = {}
BASELINE_BENCHMARKS: Dict[str, str] = {}
BASELINE_SKIP_METRICS = set(["speedup", "accuracy"])
X_ONLY_METRICS = set(["hw_roofline"])
PRECISION_DTYPE_MAPPING = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_RANGE_NAME = "tritonbench_range"


class Mode(Enum):
    FWD = "fwd"
    BWD = "bwd"
    FWD_BWD = "fwd_bwd"
    FWD_NO_GRAD = "fwd_no_grad"


class TimerContext:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.elapsed_ms = None

    def __enter__(self):
        if self.enabled:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        if self.enabled:
            end_time = time.perf_counter()
            self.elapsed_ms = (end_time - self._start_time) * 1e3


def do_bench_walltime(fn, warmup=25, rep=100):
    fn()
    torch.cuda.synchronize()

    with TimerContext() as timer:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    estimate_ms = timer.elapsed_ms / 5

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


def llama_shapes():
    # batch sizes * seq lengths
    BS = [2**i for i in range(0, 17)]
    # attn: wqkv, wo; ffn: w13, w2
    KN = [
        (4096, 12288),
        (4096, 4096),
        (4096, 22016),
        (11008, 4096),
        (8192, 1280),
        (1024, 8192),
        (8192, 7168),
        (3584, 8192),
        (16384, 2304),
        (2048, 16384),
        (16384, 13312),
        (6656, 16384),
    ]
    return [(bs, n, k, None) for bs, (k, n) in product(BS, KN)]


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
    assert (
        PATH_PREFIX in module_path
    ), f"We rely on module path prefix to identify operator name. Expected {PATH_PREFIX}<operator_name>, get {module_path}."
    suffix = module_path.partition(PATH_PREFIX)[2]
    if suffix.startswith("fb."):
        return suffix.split(".")[1]
    return suffix.split(".")[0]


@dataclass
class BenchmarkOperatorMetrics:
    # latency in ms
    latency: Optional[float] = None
    # tflops
    tflops: Optional[float] = None
    # speedup over baseline
    speedup: Optional[float] = None
    # accuracy over baseline
    accuracy: Optional[bool] = None
    # wall time
    walltime: Optional[float] = None
    # compile time
    compile_time: Optional[float] = None
    # ncu trace file
    ncu_trace: Optional[str] = None
    # ncu replay file
    ncu_rep: Optional[str] = None
    # ncu replay file with TTGIR line numbers
    ncu_rep_ir: Optional[str] = None
    # nsys replay file
    nsys_rep: Optional[str] = None
    # kineto trace file
    kineto_trace: Optional[str] = None
    # cpu peak memory
    cpu_peak_mem: Optional[float] = None
    # gpu peak memory
    gpu_peak_mem: Optional[float] = None
    # error message
    error_msg: Optional[str] = None
    # hw roofline
    hw_roofline: Optional[float] = None
    # best config
    best_config: Optional[Dict[str, Any]] = None
    # extra metrics
    extra_metrics: Optional[Dict[str, float]] = None


BUILTIN_METRICS = {x.name for x in fields(BenchmarkOperatorMetrics)} - {"extra_metrics"}


@dataclass
class BenchmarkOperatorResult:
    # Print the result in a table format
    op_name: str
    op_mode: str
    metrics: List[str]
    result: List[Tuple[Any, Dict[str, BenchmarkOperatorMetrics]]]
    _result_dict: Optional[Dict[Number, Dict[str, BenchmarkOperatorMetrics]]] = None

    def _table(self):
        table = []
        # generate headers
        headers = [REGISTERED_X_VALS[self.op_name]]
        if len(self.result) == 0:
            return headers, table
        y_val = self.result[0][1]
        y_val_keys = list(y_val.keys())
        # move the baseline benchmark to the front of the list if exists
        if (
            self.op_name in BASELINE_BENCHMARKS
            and BASELINE_BENCHMARKS[self.op_name] in y_val_keys
        ):
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
                if (
                    m in BASELINE_SKIP_METRICS
                    and k == BASELINE_BENCHMARKS[self.op_name]
                ):
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
                x_only_metric_dict = asdict(
                    y_val[y_val_keys[0][0]]
                )  # retrieve canonical name for metric function, where y_val_keys[0] = (canonical name, customized label name)
                if (
                    "extra_metrics" in x_only_metric_dict
                    and x_only_metric in x_only_metric_dict["extra_metrics"]
                ):
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
            mode="w",
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
        # tritonbench_{op_name}_{op_mode}[{x_val}-{provider}-{metric}]
        userbenchmark_metrics_dict = {}
        headers, table = self._table()
        for row in table:
            x_val = row[0]
            for ind, value in enumerate(row[1:]):
                header = headers[ind + 1]
                provider, _dash, metrics = header.partition("-")
                metric_name = f"tritonbench_{self.op_name}_{self.op_mode}[x_{x_val}-{provider}]_{metrics}"
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


def register_x_val(label: str = "x_val"):
    def decorator(function):
        operator_name = _find_op_name_from_module_path(function.__module__)
        REGISTERED_X_VALS[operator_name] = label

        def _inner(self, *args, **kwargs):
            return function(self, *args, **kwargs)

        return _inner

    return decorator


def register_benchmark(
    baseline: bool = False, enabled: bool = True, label: Optional[str] = None
):
    def decorator(function):
        operator_name = _find_op_name_from_module_path(function.__module__)
        if not operator_name in REGISTERED_BENCHMARKS:
            REGISTERED_BENCHMARKS[operator_name] = OrderedDict()
        REGISTERED_BENCHMARKS[operator_name][function.__name__] = (
            function.__name__ if not label else label
        )
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


class PostInitProcessor(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj


class BenchmarkOperator(metaclass=PostInitProcessor):
    mode: Mode = Mode.FWD
    test: str = "eval"
    device: str = "cuda"
    # By default, only collect latency metrics
    # Each operator can override to define their own default metrics
    DEFAULT_METRICS = ["latency"]
    required_metrics: List[str]
    _cur_input_id: Optional[int] = None
    _input_iter: Optional[Generator] = None
    extra_args: List[str] = []
    example_inputs: Any = None
    use_cuda_graphs: bool = True

    """
    A base class for adding operators to torch benchmark.
    """

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        set_random_seed()
        self.name = _find_op_name_from_module_path(self.__class__.__module__)
        self._raw_extra_args = copy.deepcopy(extra_args)
        self.tb_args = tb_args
        # we accept both "fwd" and "eval"
        if self.tb_args.mode == "fwd":
            self.mode = Mode.FWD
        elif self.tb_args.mode == "fwd_bwd":
            self.mode = Mode.FWD_BWD
        elif self.tb_args.mode == "fwd_no_grad":
            self.mode = Mode.FWD_NO_GRAD
        else:
            assert (
                self.tb_args.mode == "bwd"
            ), f"We only accept 3 test modes: fwd(eval), fwd_bwd(train), or bwd."
            self.mode = Mode.BWD
        self.device = tb_args.device
        self.required_metrics = (
            list(set(tb_args.metrics.split(",")))
            if tb_args.metrics
            else self.DEFAULT_METRICS
        )
        self.dargs, self.extra_args = parse_decoration_args(self, extra_args)
        if self.name not in REGISTERED_X_VALS:
            REGISTERED_X_VALS[self.name] = "x_val"
        # This will be changed by the time we apply the decoration args
        self.dtype = PRECISION_DTYPE_MAPPING.get(self.dargs.precision, None)
        self.DEFAULT_METRICS.extend(
            [
                x
                for x in REGISTERED_METRICS.get(self.name, [])
                if x not in BUILTIN_METRICS
            ]
        )
        self.DEFAULT_METRICS = list(set(self.DEFAULT_METRICS))
        if self.tb_args.baseline:
            BASELINE_BENCHMARKS[self.name] = self.tb_args.baseline
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
        with TimerContext(enabled=logger.level <= logging.INFO) as timer:
            if isinstance(self.example_inputs, dict):
                fwd_fn = fwd_fn_lambda(**self.example_inputs)
            else:
                fwd_fn = fwd_fn_lambda(*self.example_inputs)
        logger.info(
            "Took %.02fms to get benchmark function for %s",
            timer.elapsed_ms,
            bm_func_name,
        )

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
        elif self.mode == Mode.FWD_NO_GRAD:

            def fwd_no_grad_fn():
                with torch.no_grad():
                    fwd_fn()

            setattr(fwd_no_grad_fn, "_name", bm_func_name)
            return fwd_no_grad_fn

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
                self._cur_input_id = input_id
                self.example_inputs = self.get_example_inputs()
                if self.example_inputs is None:
                    warnings.warn(
                        f"The input generator get_input_iter() has depleted at id {input_id}. Available number of "
                        f"inputs: {self._available_num_inputs}.",
                        stacklevel=1,
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
            logger.warning(
                "Caught exception, terminating early with partial results",
                exc_info=True,
            )
            raise
        finally:
            self.output = BenchmarkOperatorResult(
                op_name=self.name,
                op_mode=self.mode.value,
                metrics=self.required_metrics,
                result=metrics,
            )

    def get_x_val(self, example_inputs) -> Any:
        return self._cur_input_id

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

    def best_config(self, fn):
        from unittest import mock

        from triton.runtime import Autotuner

        original_run = Autotuner.run
        autotuner = None

        def run_and_capture(self, *args, **kwargs):
            nonlocal autotuner
            autotuner = self
            original_run(self, *args, **kwargs)

        with mock.patch.object(Autotuner, "run", run_and_capture):
            fn()

        if autotuner is not None:
            return autotuner.best_config.all_kwargs()
        return None

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
        if self._num_inputs is not None:
            return self._num_inputs
        return sum(1 for _ in self.get_input_iter())

    def get_example_inputs(self):
        if self._input_iter is None:
            self._input_iter = self.get_input_iter()
        try:
            return next(self._input_iter)
        except StopIteration:
            return None

    def get_temp_path(self, path: Union[str, Path]) -> Path:
        return Path(tempfile.gettempdir()) / "tritonbench" / self.name / Path(path)

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
            hw_roofline=(
                self.hw_roofline() if "hw_roofline" in self.required_metrics else None
            ),
            extra_metrics=_init_extra_metrics(),
        )
        try:
            fn = self._get_bm_func(fn_name)
            if baseline:
                self.baseline_fn = fn
            if set(["latency", "tflops", "speedup", "compile_time"]) & set(
                self.required_metrics
            ):
                if self.use_cuda_graphs:
                    with torch.cuda.stream(torch.cuda.Stream()):
                        metrics.latency = triton.testing.do_bench_cudagraph(
                            fn,
                            rep=rep,
                            return_mode="median",
                            grad_to_none=self.get_grad_to_none(self.example_inputs),
                        )
                else:
                    metrics.latency = triton.testing.do_bench(
                        fn,
                        warmup=warmup,
                        rep=rep,
                        return_mode="median",
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
                    self.baseline_metrics.latency / metrics.latency
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
                metrics.cpu_peak_mem, _device_id, metrics.gpu_peak_mem = (
                    self.get_peak_mem(fn)
                )
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
            if "ncu_rep_ir" in self.required_metrics:
                metrics.ncu_rep_ir = self.ncu_trace(
                    input_id, fn_name, replay=True, profile_ir=True
                )
            if "nsys_rep" in self.required_metrics:
                metrics.nsys_rep = self.nsys_rep(input_id, fn_name)
            if "kineto_trace" in self.required_metrics:
                metrics.kineto_trace = self.kineto_trace(input_id, fn)
            if "best_config" in self.required_metrics:
                metrics.best_config = self.best_config(fn)
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
                metrics.extra_metrics["_compile_time_in_task"] = (
                    self._compile_time_in_task(fn)
                )
            if "_ncu_trace_in_task" in self.required_metrics:
                assert (
                    self.required_metrics == ["_ncu_trace_in_task"]
                    and len(self._only) == 1
                    and (self._input_id is not None)
                ), (
                    "_ncu_trace_in_task must be measured by itself. "
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _input_id: {self._input_id}"
                )
                from torchbenchmark._components.ncu import do_bench_in_task

                do_bench_in_task(
                    fn=fn,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                    range_name=_RANGE_NAME,
                )
                metrics.extra_metrics["_ncu_trace_in_task"] = "success"
            if "_nsys_rep_in_task" in self.required_metrics:
                assert (
                    self.required_metrics == ["_nsys_rep_in_task"]
                    and len(self._only) == 1
                    and (self._input_id is not None)
                ), (
                    "_nsys_rep_in_task must be measured by itself. "
                    f"required_metrics: {self.required_metrics}, _only: {self._only}, _input_id: {self._input_id}"
                )
                from torchbenchmark._components.ncu import do_bench_in_task

                do_bench_in_task(
                    fn=fn,
                    grad_to_none=self.get_grad_to_none(self.example_inputs),
                    range_name=_RANGE_NAME,
                    warmup=True,
                    use_cuda_profiler_range=True,
                )
                metrics.extra_metrics["_nsys_rep_in_task"] = "success"
            # generate customized metrics
            if self.name in REGISTERED_METRICS:
                for metric_name in REGISTERED_METRICS[self.name]:
                    if metric_name in BUILTIN_METRICS:
                        continue
                    if not metric_name in self.required_metrics:
                        continue
                    func = getattr(self, metric_name)
                    metrics.extra_metrics[metric_name] = func(
                        fn, self.example_inputs, metrics
                    )
            if self.tb_args.dump_ir:
                self.dump_ir(input_id, fn)
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

    def nsys_rep(self, input_id: int, fn_name: str) -> str:
        import subprocess
        import sys

        op_task_args = [] if IS_FBCODE else [sys.executable]
        op_task_args.extend(copy.deepcopy(sys.argv))
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
                "_nsys_rep_in_task",
            ]
        )
        nsys_output_dir = self.get_temp_path(f"nsys_traces/{fn_name}_{input_id}")
        nsys_output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".nsys-rep"
        nsys_output_file = nsys_output_dir.joinpath(f"nsys_output{ext}").resolve()
        nsys_trace_cmd = [
            "nsys",
            "profile",
            "-c",
            "cudaProfilerApi",
            "-t",
            "nvtx,osrt,cuda,cudnn,cublas",
            "-w",
            "true",
            "-f",
            "true",
            "-o",
            nsys_output_file,
        ]
        nsys_trace_cmd.extend(op_task_args)
        try:
            subprocess.check_call(nsys_trace_cmd)
        except subprocess.CalledProcessError:
            # FIXME: calling nsys on Tritonbench will throw SIGTERM with error code 143
            pass
        return str(nsys_output_file.resolve())

    def ncu_trace(
        self, input_id: int, fn_name: str, replay: bool = False, profile_ir=False
    ) -> str:
        import subprocess

        # collect the ncu trace
        import sys

        op_task_args = [] if IS_FBCODE else [sys.executable]
        op_task_args.extend(copy.deepcopy(sys.argv))
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
        disable_dyno_dcgm = [
            "sudo",
            "dyno",
            "dcgm_profiling",
            "--mute=true",
            "--duration=100000_s",
        ]
        disable_dcgm_service = [
            "sudo",
            "systemctl",
            "stop",
            "nvidia-dcgm",
        ]
        if (
            subprocess.run(disable_dyno_dcgm).returncode != 0
            and subprocess.run(disable_dcgm_service).returncode != 0
        ):
            warnings.warn(
                "DCGM may not have been successfully disabled. Proceeding to collect NCU trace anyway..."
            )
        ncu_output_dir = self.get_temp_path(f"ncu_traces/{fn_name}_{input_id}")
        ncu_output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".csv" if not replay else ".ncu-rep"
        ncu_output_file = ncu_output_dir.joinpath(
            f"ncu_output{'_ir' if profile_ir else ''}{ext}"
        ).resolve()
        ncu_args = [
            "ncu",
            "--set",
            "full",
            "--nvtx",
            "--nvtx-include",
            f"{_RANGE_NAME}/",
            "--target-processes",
            "all",
            "--import-source",
            "yes",
        ]
        if replay:
            ncu_args.extend(
                [
                    "-f",
                    "-o",
                    str(ncu_output_file.resolve()),
                ]
            )
        else:
            ncu_args.extend(
                [
                    "--csv",
                    "-f",
                    "--log-file",
                    str(ncu_output_file.resolve()),
                ]
            )
        ncu_args.extend(op_task_args)
        logger.info("Running NCU: %s", shlex.join(ncu_args))
        # Sometimes, `ncu --target-processes all` will fail with the message "Failed to connect to process". Setting
        # CUDA_INJECTION64_PATH=none seems to fix this issue.
        env = {**os.environ, "CUDA_INJECTION64_PATH": "none"}
        if profile_ir:
            env["USE_TTGIR_LOC"] = "1"
        subprocess.check_call(ncu_args, env=env)
        return str(ncu_output_file.resolve())

    def kineto_trace(self, input_id: int, fn: Callable) -> str:
        from pathlib import Path

        from torchbenchmark._components.kineto import do_bench_kineto

        kineto_output_dir = self.get_temp_path(f"kineto_traces/{fn._name}_{input_id}")
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
        latency_without_compile = metrics.latency
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
    ) -> float:
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
        return op_flops / metrics.latency / 1e12 * 1e3

    def dump_ir(self, input_id, fn):
        from unittest import mock

        from triton.runtime.jit import JITFunction

        original_run = JITFunction.run
        compiled_kernels = []

        # There isn't really a great way to get the compiled kernels without monkeypatching
        def run_and_capture(self, *args, **kwargs):
            compiled_kernel = original_run(self, *args, **kwargs)
            compiled_kernels.append(compiled_kernel)
            return compiled_kernel

        with mock.patch.object(JITFunction, "run", run_and_capture):
            fn()

        if len(compiled_kernels) > 0:
            ir_dir = self.get_temp_path("ir")
            ir_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Writing Triton IR to %s", ir_dir)

        for kernel in compiled_kernels:
            for ir in ["ttir", "ttgir", "llir", "ptx", "amdgcn"]:
                if ir in kernel.asm:
                    with open(
                        ir_dir / f"{fn._name}_{kernel.name}_{input_id}.{ir}", "w"
                    ) as f:
                        f.write(kernel.asm[ir])
            if "cubin" in kernel.asm:
                from triton.tools.disasm import get_sass

                sass = get_sass(kernel.asm["cubin"])
                with open(
                    ir_dir / f"{fn._name}_{kernel.name}_{input_id}.sass", "w"
                ) as f:
                    f.write(sass)
