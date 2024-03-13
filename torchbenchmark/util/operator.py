import functools
import numpy
import triton
import torch
import tabulate
import warnings
from dataclasses import dataclass, fields, asdict, make_dataclass
from typing import List, Dict, Generator, Optional, Callable, Tuple
from numbers import Number
from torchbenchmark.util.input import input_cast
from torchbenchmark.util.extra_args import parse_decoration_args, apply_decoration_args

DEFAULT_WARMUP = 25
DEFAULT_RUN_ITERS = 100
DEFAULT_QUANTILES = [0.5, 0.1, 0.9]
REGISTERED_BENCHMARKS: Dict[str, List[str]] = {}
REGISTERED_METRICS: Dict[str, List[str]] = {}
BASELINE_BENCHMARKS: Dict[str, str] = {}
DEFAULT_METRICS = ["latency", "speedup", "accuracy"]

@dataclass
class BenchmarkOperatorMetrics:
    # latency in ms
    latency: List[float]
    tflops: List[float]
    # speedup over baseline
    speedup: Optional[float]
    # accuracy over baseline
    accuracy: Optional[bool]
    # extra metrics
    extra_metrics: Dict[str, float]

@dataclass
class BenchmarkOperatorResult:
    # Print the result in a table format
    op_name: str
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
        for k in y_val_keys:
            metrics_dict = asdict(y_val[k])
            metrics_dict_keys = list(metrics_dict.keys())
            metrics_dict_keys.remove("extra_metrics")
            for metric in metrics_dict_keys:
                if metrics_dict[metric]:
                    headers.append(f"{k}_{metric}")
            # add extra metrics
            for extra_metric in metrics_dict["extra_metrics"].keys():
                headers.append(f"{k}_{extra_metric}")
        # generate rows
        x_val_list = sorted(self.result.keys())
        for x_val in x_val_list:
            row = []
            row.append(x_val)
            y_val = self.result[x_val]
            for k in y_val_keys:
                metrics_dict = asdict(y_val[k])
                for metric in metrics_dict_keys:
                    if metrics_dict[metric] is not None:
                        if isinstance(metrics_dict[metric], list):
                            row.append(numpy.median(metrics_dict[metric]))
                        elif isinstance(metrics_dict[metric], bool):
                            row.append(1.0 if metrics_dict[metric] else 0.0)
                        else:
                            row.append(metrics_dict[metric])
                for extra_metric in metrics_dict["extra_metrics"].keys():
                    row.append(metrics_dict["extra_metrics"][extra_metric])
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


def register_benchmark(baseline: bool=False):
    def decorator(function):
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

def register_metric(func):
    # register the metric to dict
    operator_name = func.__module__.split(".")[-1]
    if not operator_name in REGISTERED_METRICS:
        REGISTERED_METRICS[operator_name] = []
    REGISTERED_METRICS[operator_name].append(func.__name__)
    def _inner(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    return _inner

class BenchmarkOperator():
    test: str
    device: str
    _input_iter: Optional[Generator] = None

    # By default, generate 100 data points
    DEFAULT_NUM_BATCH = 100

    """
    A base class for adding operators to torch benchmark.
    """
    def __init__(self, test: str, device: str, metrics: List[str]=DEFAULT_METRICS, extra_args: List[str]=[]):
        self.test = test
        self.device = device
        relative_path = self.__class__.__module__.split(".")
        self.name = relative_path[-1]
        self.is_training = self.test == "train"
        self.required_metrics = metrics

        self.dargs, self.extra_args = parse_decoration_args(self, extra_args)
        if self.dargs.num_batch == None:
            self.dargs.num_batch = self.DEFAULT_NUM_BATCH

    def run(self,
            warmup=DEFAULT_WARMUP,
            rep=DEFAULT_RUN_ITERS,
            quantiles=DEFAULT_QUANTILES) -> BenchmarkOperatorResult:
        """Benchmarking the operator and returning its metrics."""
        metrics = {}
        for _dp in range(self.dargs.num_batch):
            example_inputs =  self.get_example_inputs()
            if example_inputs == None:
                warnings.warn(
                    UserWarning(f"The input generated by get_input_iter() has depleted. Maximum input batches {_dp}.")
                )
                break
            # Move the inputs to the device
            example_inputs = input_cast(lambda x: True, lambda x: x.to(self.device), example_inputs)
            self.op_flops = None
            self.example_inputs = example_inputs
            self.baseline_fn = None
            # Cast the input precisions
            apply_decoration_args(self, self.dargs)
            x_val = self.get_x_val(example_inputs)
            # Run the baseline first
            baseline_metrics = None
            if self.name in BASELINE_BENCHMARKS:
                fn = lambda: getattr(self, BASELINE_BENCHMARKS[self.name])(*example_inputs)
                if self.op_flops == None and "tflops" in self.required_metrics:
                    self.op_flops = self.get_flops(fn)
                self.baseline_fn = fn
                baseline_metrics = self._do_bench(fn=fn,warmup=warmup, rep=rep, quantiles=quantiles)
            benchmarks = [ bm for bm in REGISTERED_BENCHMARKS[self.name] if not bm == BASELINE_BENCHMARKS[self.name] ] \
                if self.name in REGISTERED_BENCHMARKS else []
            # get metrics for for each registered benchmark
            def _reduce_benchmarks(acc, bm_name):
                bm_func = getattr(self, bm_name, None)
                assert bm_func, f"Could not find benchmark {bm_name} registered in {self.name}. Please report a bug."
                fn = lambda: bm_func(*example_inputs)
                if self.op_flops == None and "tflops" in self.required_metrics:
                    self.op_flops = self.get_flops(fn)
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
        self.output = BenchmarkOperatorResult(
            op_name=self.name,
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

    def _cast_to(self, cond, action):
        return input_cast(cond, action, self.example_inputs)

    def enable_bf16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.to(torch.bfloat16)
        self._cast_to(tensor_cond, tensor_action)

    def enable_fp16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.half()
        self._cast_to(tensor_cond, tensor_action)

    def enable_channels_last(self):
        tensor_cond = lambda x: x.dim() == 4
        tensor_action = lambda x: x.to(memory_format=torch.channels_last)
        self._cast_to(tensor_cond, tensor_action)

    def get_flops(self, func: Callable) -> float:
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
        latency = None
        tflops = None
        speedup = None
        accuracy = None
        if "latency" in self.required_metrics:
            latency = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
        if "tflops" in self.required_metrics:
            tflops = list(map(lambda x: self.op_flops / x / 1e12 * 1e3, latency))
        if "speedup" in self.required_metrics:
            speedup = numpy.median(baseline_metrics.latency) / numpy.median(latency) if baseline_metrics else None
        if "accuracy" in self.required_metrics:
            accuracy = self._get_accuracy(fn, baseline_fn) if baseline_fn else None
        metric = BenchmarkOperatorMetrics(
            latency=latency,
            tflops=tflops,
            speedup=speedup,
            accuracy=accuracy,
            extra_metrics={},
        )
        # generate customized metrics
        extra_metrics = {}
        if self.name in REGISTERED_METRICS:
            for metric_name in REGISTERED_METRICS[self.name]:
                func = getattr(self, metric_name)
                extra_metrics[metric_name] = func(self.example_inputs, metric)
            metric.extra_metrics = extra_metrics
        return metric
