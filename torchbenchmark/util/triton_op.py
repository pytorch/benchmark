import functools
import numpy
import argparse
import triton
import torch
import gc
import tabulate
import warnings
from dataclasses import dataclass, fields, asdict, make_dataclass
from typing import List, Dict, Generator, Optional, Callable, Tuple, Any
from numbers import Number
from torchbenchmark.util.input import input_cast
from torchbenchmark.util.env_check import set_random_seed
from torchbenchmark.util.extra_args import parse_decoration_args, apply_decoration_args

DEFAULT_WARMUP = 25
DEFAULT_RUN_ITERS = 100
DEFAULT_QUANTILES = [0.5, 0.1, 0.9]
REGISTERED_BENCHMARKS: Dict[str, List[str]] = {}
REGISTERED_METRICS: Dict[str, List[str]] = {}
BASELINE_BENCHMARKS: Dict[str, str] = {}
BUILTIN_METRICS = ["latency", "tflops", "speedup", "accuracy"]
BASELINE_SKIP_METRICS = ["speedup", "accuracy"]
DEFAULT_METRICS = "latency"

@dataclass
class BenchmarkOperatorMetrics:
    # latency in ms
    latency: Optional[List[float]]
    tflops: Optional[List[float]]
    # speedup over baseline
    speedup: Optional[float]
    # accuracy over baseline
    accuracy: Optional[bool]
    # error message
    error_msg: Optional[str]
    # extra metrics
    extra_metrics: Dict[str, float]

@dataclass
class BenchmarkOperatorResult:
    # Print the result in a table format
    op_name: str
    metrics: List[str]
    result: Dict[Number, Dict[str, BenchmarkOperatorMetrics]]

    def _table(self):
        table = []
        # generate headers
        headers = ["x_val"]
        y_val = self.result[list(self.result.keys())[0]]
        y_val_keys = list(y_val.keys())
        # move the baseline benchmark to the front of the list if exists
        if BASELINE_BENCHMARKS[self.op_name] in y_val_keys:
            y_val_keys.insert(0,
                y_val_keys.pop(
                    y_val_keys.index(BASELINE_BENCHMARKS[self.op_name])))
        key_metrics = {}
        for k in y_val_keys:
            metrics = [ x for x in self.metrics if x not in  BASELINE_SKIP_METRICS ] \
                    if k == BASELINE_BENCHMARKS[self.op_name] else self.metrics
            key_metrics[k] = metrics
            for metric in metrics:
                # add extra metrics
                headers.append(f"{k}_{metric}")
        # generate rows
        x_val_list = sorted(self.result.keys())
        for x_val in x_val_list:
            row = []
            row.append(x_val)
            y_val = self.result[x_val]
            for k in y_val_keys:
                metrics_dict = asdict(y_val[k])
                for metric in key_metrics[k]:
                    if metrics_dict["error_msg"]:
                        row.append(metrics_dict["error_msg"])
                    elif metric in metrics_dict:
                        if isinstance(metrics_dict[metric], list):
                            row.append(numpy.median(metrics_dict[metric]))
                        elif isinstance(metrics_dict[metric], bool):
                            row.append(1.0 if metrics_dict[metric] else 0.0)
                        else:
                            row.append(metrics_dict[metric])
                    elif metric in metrics_dict["extra_metrics"]:
                        row.append(metrics_dict["extra_metrics"][metric])
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
        return sorted(self.result.keys())

    def get_y_vals(self, x_val, provider, metric_name: str):
        y_vals = self.result[x_val][provider]
        metrics_dict = asdict(y_vals)
        if metric_name in metrics_dict:
            return metrics_dict[metric_name]
        assert metric_name in metrics_dict["extra_metrics"], \
            f"Metric {metric_name} could not be found."
        return metrics_dict["extra_metrics"][metric_name]

    def __str__(self):
        headers, table = self._table()
        table = tabulate.tabulate(table, headers=headers)
        return table

def register_benchmark(baseline: bool=False, enabled: bool=True, preprocess: Optional[Callable]=None):
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
        if preprocess:
            setattr(_inner, "_preprocess", preprocess)
        return _inner
    return decorator

def register_metric(func):
    operator_name = func.__module__.split(".")[-1]
    if not operator_name in REGISTERED_METRICS:
        REGISTERED_METRICS[operator_name] = []
    REGISTERED_METRICS[operator_name].append(func.__name__)
    def _inner(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    return _inner

def parse_args(op_name: str, args: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser()
    default_metrics = DEFAULT_METRICS + "," + ",".join(REGISTERED_METRICS.get(op_name, []))
    parser.add_argument("--metrics", default=default_metrics)
    return parser.parse_known_args(args)

class BenchmarkOperator():
    test: str = "eval"
    device: str = "cuda"
    _input_iter: Optional[Generator] = None
    extra_args: List[str] = []
    example_inputs: Any = None

    # By default, generate 100 data points
    DEFAULT_NUM_BATCH = 100

    """
    A base class for adding operators to torch benchmark.
    """
    def __init__(self, test: str, device: str, extra_args: List[str]=[]):
        relative_path = self.__class__.__module__.split(".")
        set_random_seed()
        self.name = relative_path[-1]
        self.is_training = self.test == "train"
        self.dargs, unprocessed_args = parse_decoration_args(self, extra_args)
        if self.dargs.num_batch == None:
            self.dargs.num_batch = self.DEFAULT_NUM_BATCH
        self.tb_args, self.extra_args = parse_args(self.name, unprocessed_args)
        self.required_metrics = list(set(self.tb_args.metrics.split(",")))

    def _preprocess_inputs(self, fn: Callable, example_inputs):
        if hasattr(fn, "_preprocess"):
            return (getattr(fn, "_preprocess")(*([self] + list(example_inputs))), )
        else:
            return example_inputs

    def run(self,
            warmup=DEFAULT_WARMUP,
            rep=DEFAULT_RUN_ITERS,
            quantiles=DEFAULT_QUANTILES) -> BenchmarkOperatorResult:
        """Benchmarking the operator and returning its metrics."""
        metrics = {}
        for _dp in range(self.dargs.num_batch):
            self.example_inputs =  self.get_example_inputs()
            if self.example_inputs == None:
                warnings.warn(
                    UserWarning(f"The input generator get_input_iter() has depleted. Maximum input batches {_dp}.")
                )
                break
            # Move inputs to the device
            self.example_inputs = input_cast(
                lambda x: isinstance(x, torch.Tensor),
                lambda x: x.to(self.device),
                self.example_inputs,
            )
            self.baseline_fn = None
            self._op_flops = {}
            # Cast the input precisions
            apply_decoration_args(self, self.dargs)
            x_val = self.get_x_val(self.example_inputs)
            # Run the baseline first
            baseline_metrics = None
            if self.name in BASELINE_BENCHMARKS:
                inputs = self._preprocess_inputs(self.baseline_fn, self.example_inputs)
                fn = lambda: getattr(self, BASELINE_BENCHMARKS[self.name])(*inputs)
                self.baseline_fn = fn
                baseline_metrics = self._do_bench(fn=fn,warmup=warmup, rep=rep, quantiles=quantiles)
            benchmarks = [ bm for bm in REGISTERED_BENCHMARKS[self.name] if not bm == BASELINE_BENCHMARKS[self.name] ] \
                if self.name in REGISTERED_BENCHMARKS else []
            # get metrics for for each registered benchmark
            def _reduce_benchmarks(acc, bm_name):
                bm_func = getattr(self, bm_name, None)
                assert bm_func, f"Could not find benchmark {bm_name} registered in {self.name}. Please report a bug."
                inputs = self._preprocess_inputs(bm_func, self.example_inputs)
                fn = lambda: bm_func(*inputs)
                acc[bm_name] = self._do_bench(
                    fn=fn,
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                    baseline_fn=self.baseline_fn,
                    baseline_metrics=baseline_metrics,
                )
                return acc
            y_vals: Dict[str, BenchmarkOperatorMetrics] = functools.reduce(_reduce_benchmarks, benchmarks, {})
            if baseline_metrics:
                y_vals[BASELINE_BENCHMARKS[self.name]] = baseline_metrics
            metrics[x_val] = y_vals
            del self.example_inputs
            gc.collect()
        self.output = BenchmarkOperatorResult(
            op_name=self.name,
            metrics=self.required_metrics,
            result=metrics,
        )
        return self.output


    def get_x_val(self, example_inputs) -> float:
        raise NotImplementedError("Each operator must implement its own input to x_val mapping.")


    def get_input_iter(self) -> Generator:
        """Return the dynamic input iterator for the model."""
        raise NotImplementedError("Each operator must implement its own input iterator.")


    def plot(self):
        """Plot the comparison between different operator implementations."""
        raise NotImplementedError("Each operator must implement its own plotting logic.")


    def enable_bf16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.to(torch.bfloat16)
        self.example_inputs = input_cast(tensor_cond, tensor_action, self.example_inputs)


    def enable_fp16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.half()
        self.example_inputs = input_cast(tensor_cond, tensor_action, self.example_inputs)


    def enable_channels_last(self):
        tensor_cond = lambda x: x.dim() == 4
        tensor_action = lambda x: x.to(memory_format=torch.channels_last)
        self.example_inputs = input_cast(tensor_cond, tensor_action, self.example_inputs)


    def get_example_inputs(self):
        if self._input_iter == None:
            self._input_iter = self.get_input_iter()
        try:
            return next(self._input_iter)
        except StopIteration:
            return None


    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output, loss = fn()
        baseline_output, baseline_loss = baseline_fn()
        accuracy = True
        try:
            torch.testing.assert_close(output, baseline_output)
            if not (loss == None and baseline_loss == None):
                torch.testing.assert_close(loss.grad, baseline_loss.grad)
        except AssertionError:
            # either the output tensor or the loss grad tensor does not match
            accuracy = False
        finally:
            return accuracy


    def _do_bench(self,
                  fn: Callable,
                  warmup=DEFAULT_WARMUP,
                  rep=DEFAULT_RUN_ITERS,
                  quantiles=DEFAULT_QUANTILES,
                  baseline_fn: Optional[Callable]=None,
                  baseline_metrics: Optional[BenchmarkOperatorMetrics]=None) -> BenchmarkOperatorMetrics:
        latency = []
        tflops = []
        speedup = None
        accuracy = None
        try:
            if set(["latency", "tflops", "speedup"]) & set(self.required_metrics):
                latency = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
            if "tflops" in self.required_metrics:
                tflops = self.tflops(latency, fn)
            if "speedup" in self.required_metrics:
                speedup = numpy.median(baseline_metrics.latency) / numpy.median(latency) if baseline_metrics else None
            if "accuracy" in self.required_metrics:
                accuracy = self._get_accuracy(fn, baseline_fn) if baseline_fn else None
            metric = BenchmarkOperatorMetrics(
                latency=latency,
                tflops=tflops,
                speedup=speedup,
                accuracy=accuracy,
                error_msg=None,
                extra_metrics={},
            )
            # generate customized metrics
            extra_metrics = {}
            if self.name in REGISTERED_METRICS:
                for metric_name in REGISTERED_METRICS[self.name]:
                    if metric_name in BUILTIN_METRICS:
                        continue
                    func = getattr(self, metric_name)
                    extra_metrics[metric_name] = func(self.example_inputs, metric)
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

    @register_metric
    def tflops(self, latency: List[float], func: Optional[Callable]=None) -> List[float]:
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
            total_flops = sum([v for _, v in flop_counter.flop_counts["Global"].items()])
            return total_flops
        if not func in self._op_flops:
            self._op_flops[func] = _get_flops(self, func)
        op_flops = self._op_flops[func]
        return list(map(lambda x: op_flops / x / 1e12 * 1e3, latency))
