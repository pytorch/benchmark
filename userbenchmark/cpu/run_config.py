"""
Run PyTorch cpu benchmarking.
"""

import argparse
import os
from pathlib import Path

from typing import Dict, List, Optional

import numpy

from cpu_utils import (
    add_path,
    dump_output,
    get_output_dir,
    get_output_json,
    list_metrics,
    parse_str_to_list,
    REPO_PATH,
    validate,
)

with add_path(str(REPO_PATH)):
    from torchbenchmark.util.experiment.instantiator import (
        list_devices,
        list_models,
        list_tests,
        load_model,
        load_model_isolated,
        TorchBenchModelConfig,
    )
    from torchbenchmark.util.experiment.metrics import (
        get_model_test_metrics,
        TorchBenchModelMetrics,
    )

    BM_NAME = "cpu"
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    # output_iter_metrics is True only when '--output-iter-metrics' is given,
    # otherwise it is False by default.
    def result_to_output_metrics(
        metrics: List[str],
        metrics_res: TorchBenchModelMetrics,
        output_iter_metrics: bool,
    ) -> Dict[str, float]:
        result_metrics = {}
        if metrics_res:
            if "latencies" in metrics and metrics_res.latencies:
                latency_metric = "latency"
                median_latency = numpy.median(metrics_res.latencies)
                assert median_latency, f"Run failed for metric {latency_metric}"
                result_metrics[latency_metric] = median_latency
                if output_iter_metrics:
                    iter_latencies_metric = "iter_latencies"
                    result_metrics[iter_latencies_metric] = list(metrics_res.latencies)
            if "throughputs" in metrics and metrics_res.throughputs:
                throughput_metric = "throughput"
                median_throughput = numpy.median(metrics_res.throughputs)
                assert median_throughput, f"Run failed for metric {throughput_metric}"
                result_metrics[throughput_metric] = median_throughput
                if output_iter_metrics:
                    iter_throughputs_metric = "iter_throughputs"
                    result_metrics[iter_throughputs_metric] = list(
                        metrics_res.throughputs
                    )
            if "cpu_peak_mem" in metrics and metrics_res.cpu_peak_mem:
                cpu_peak_mem = "cpu_peak_mem"
                result_metrics[cpu_peak_mem] = metrics_res.cpu_peak_mem
        return result_metrics

    def dump_result_to_json(metrics, output_dir):
        result = get_output_json(BM_NAME, metrics)
        dump_output(BM_NAME, result, output_dir)

    def run_config(
        config: TorchBenchModelConfig,
        metrics: List[str],
        nwarmup: int,
        niter: int,
        dryrun: bool = False,
    ) -> Optional[TorchBenchModelMetrics]:
        """This function only handles NotImplementedError, all other errors will fail."""
        print(f"Running {config} ...", end="")
        if dryrun:
            return None
        # We do not allow RuntimeError in this test
        try:
            if "cpu_peak_mem" in metrics:
                # load the model instance within separate subprocess
                model = load_model_isolated(config)
            else:
                # load the model instance within current process
                model = load_model(config)
            # get the model test metrics
            result: TorchBenchModelMetrics = get_model_test_metrics(
                model, metrics=metrics, nwarmup=nwarmup, num_iter=niter
            )
        except NotImplementedError as e:
            print(" [NotImplemented]")
            return None
        print(" [Done]")
        return result

    def run(args: List[str], extra_args: List[str]):
        device = validate(args.device, list_devices())
        test = validate(args.test, list_tests())
        model = validate(args.model, list_models())
        metrics = validate(parse_str_to_list(args.metrics), list_metrics())
        config = TorchBenchModelConfig(
            name=model,
            device=device,
            test=test,
            batch_size=args.batch_size,
            extra_args=extra_args,
            extra_env=None,
        )
        try:
            metrics_res = run_config(
                config,
                metrics,
                nwarmup=int(args.nwarmup),
                niter=int(args.niter),
                dryrun=args.dryrun,
            )
        except KeyboardInterrupt:
            print("User keyboard interrupted!")
        if not args.dryrun:
            args.output = args.output if args.output else get_output_dir(BM_NAME)
            target_dir = Path(args.output).joinpath(f"{config.name}-{config.test}")
            target_dir.mkdir(exist_ok=True, parents=True)
            metrics_dict = result_to_output_metrics(
                metrics, metrics_res, args.output_iter_metrics
            )
            dump_result_to_json(metrics_dict, target_dir)

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", "-d", default="cpu", help="Devices to run.")
        parser.add_argument("--test", "-t", default="eval", help="Tests to run.")
        parser.add_argument(
            "--model",
            "-m",
            default=None,
            type=str,
            help="Only run the specifice model.",
        )
        parser.add_argument(
            "--batch-size",
            "-b",
            default=None,
            type=int,
            help="Run the specifice batch size.",
        )
        parser.add_argument("--output", "-o", default=None, help="Output dir.")
        parser.add_argument(
            "--metrics", default="latencies", help="Benchmark metrics, split by comma."
        )
        parser.add_argument(
            "--output-iter-metrics",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable per-iteration benchmark metrics",
        )
        parser.add_argument(
            "--nwarmup", default=20, help="Benchmark warmup iteration number."
        )
        parser.add_argument("--niter", default=50, help="Benchmark iteration number.")
        parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
        args, extra_args = parser.parse_known_args()
        run(args, extra_args)
