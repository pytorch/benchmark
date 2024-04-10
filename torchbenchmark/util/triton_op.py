import argparse
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
from torchbenchmark.util.env_check import set_random_seed
from torchbenchmark.util.extra_args import apply_decoration_args, parse_decoration_args
from torchbenchmark.util.input import input_cast

DEFAULT_WARMUP = 25
DEFAULT_RUN_ITERS = 100
DEFAULT_QUANTILES = [0.5, 0.1, 0.9]
REGISTERED_BENCHMARKS: Dict[str, List[str]] = {}
REGISTERED_METRICS: Dict[str, List[str]] = {}
BASELINE_BENCHMARKS: Dict[str, str] = {}
BUILTIN_METRICS = ["latency", "tflops", "speedup", "accuracy"]
BASELINE_SKIP_METRICS = ["speedup", "accuracy"]
PRECISION_DTYPE_MAPPING = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Mode(Enum):
    FWD = 1
    BWD = 2
    FWD_BWD = 3
    FWD_NO_GRAD = 4


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
    # error message
    error_msg: Optional[str]
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
        for k in y_val_keys:
            metrics = (
                [x for x in self.metrics if x not in BASELINE_SKIP_METRICS]
                if k == BASELINE_BENCHMARKS[self.op_name]
                else self.metrics
            )
            key_metrics[k] = metrics
            for metric in metrics:
                # add extra metrics
                headers.append(f"{k}_{metric}")
        # generate rows
        for x_val, y_val in self.result:
            row = []
            row.append(x_val)
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


def register_metric(skip_baseline: bool = False):
    def decorator(func):
        operator_name = func.__module__.split(".")[-1]
        if not operator_name in REGISTERED_METRICS:
            REGISTERED_METRICS[operator_name] = []
        REGISTERED_METRICS[operator_name].append(func.__name__)
        if skip_baseline:
            BASELINE_SKIP_METRICS.append(func.__name__)

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
        if self.dargs.num_batch == None:
            self.dargs.num_batch = self.DEFAULT_NUM_BATCH
        self.DEFAULT_METRICS.extend(REGISTERED_METRICS.get(self.name, []))
        self.DEFAULT_METRICS = list(set(self.DEFAULT_METRICS))
        self.tb_args, self.extra_args = parse_args(
            self.DEFAULT_METRICS, unprocessed_args
        )
        self.required_metrics = list(set(self.tb_args.metrics.split(",")))

    def _get_bm_func(self, bm_func_name: str):
        fwd_fn_lambda = getattr(self, bm_func_name, None)
        assert (
            fwd_fn_lambda
        ), f"Could not find benchmark {bm_func_name} registered in {self.name}. Please report a bug."
        if isinstance(self.example_inputs, dict):
            fwd_fn = fwd_fn_lambda(**self.example_inputs)
        else:
            fwd_fn = fwd_fn_lambda(*self.example_inputs)
        if self.mode == Mode.FWD:
            return fwd_fn
        elif self.mode == Mode.BWD:
            return self.get_bwd_fn(fwd_fn)
        elif self.mode == Mode.FWD_BWD:
            bwd_fn = self.get_bwd_fn(fwd_fn)
            return lambda: (fwd_fn(), bwd_fn())

    def run(
        self, warmup=DEFAULT_WARMUP, rep=DEFAULT_RUN_ITERS, quantiles=DEFAULT_QUANTILES
    ) -> BenchmarkOperatorResult:
        """Benchmarking the operator and returning its metrics."""
        metrics = []
        for _dp in range(self.dargs.num_batch):
            self.example_inputs = self.get_example_inputs()
            if self.example_inputs == None:
                warnings.warn(
                    UserWarning(
                        f"The input generator get_input_iter() has depleted. Maximum input batches {_dp}."
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
            # Run the baseline first
            if self.name in BASELINE_BENCHMARKS:
                self.baseline_metrics = self._do_bench(
                    fn_name=BASELINE_BENCHMARKS[self.name],
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                    baseline=True,
                )
            benchmarks = (
                [
                    bm
                    for bm in REGISTERED_BENCHMARKS[self.name]
                    if not bm == BASELINE_BENCHMARKS.get(self.name, None)
                ]
                if self.name in REGISTERED_BENCHMARKS
                else []
            )

            # get metrics for for each registered benchmark
            def _reduce_benchmarks(acc, bm_name: str):
                acc[bm_name] = self._do_bench(
                    fn_name=bm_name,
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                    baseline=False,
                )
                return acc

            y_vals: Dict[str, BenchmarkOperatorMetrics] = functools.reduce(
                _reduce_benchmarks, benchmarks, {}
            )
            if self.baseline_metrics:
                y_vals[BASELINE_BENCHMARKS[self.name]] = self.baseline_metrics
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
        if self._input_iter == None:
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
        fn_name: str,
        warmup=DEFAULT_WARMUP,
        rep=DEFAULT_RUN_ITERS,
        quantiles=DEFAULT_QUANTILES,
        baseline: bool = False,
    ) -> BenchmarkOperatorMetrics:
        latency = []
        tflops = []
        speedup = None
        accuracy = None
        walltime = None
        error_msg = None
        try:
            fn = self._get_bm_func(fn_name)
            if baseline:
                self.baseline_fn = fn
            if set(["latency", "tflops", "speedup"]) & set(self.required_metrics):
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
            if not baseline and "accuracy" in self.required_metrics:
                accuracy = (
                    self._get_accuracy(fn, self.baseline_fn)
                    if self.baseline_fn
                    else None
                )
            metric = BenchmarkOperatorMetrics(
                latency=latency,
                tflops=None,
                speedup=speedup,
                accuracy=accuracy,
                walltime=walltime,
                error_msg=error_msg,
                extra_metrics={},
            )
            if "tflops" in self.required_metrics:
                tflops = self.tflops(fn, self.example_inputs, metric)
                metric.tflops = tflops
            # generate customized metrics
            extra_metrics = {}
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
                error_msg="CUDA OOM",
                extra_metrics={},
            )
        return metric

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
