
"""
Run PyTorch cpu benchmarking.
"""
import argparse
import os
import numpy

from typing import List, Dict, Optional
from pathlib import Path

from cpu_utils import add_path, REPO_PATH, validate, parse_str_to_list, list_metrics, get_output_dir, get_output_json, dump_output
with add_path(str(REPO_PATH)):
    from torchbenchmark.util.experiment.instantiator import (list_models, load_model, load_model_isolated, TorchBenchModelConfig,
                                                            list_devices, list_tests)
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

    BM_NAME = 'cpu'
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    def get_output_subdir(config: TorchBenchModelConfig) -> str:
        mode = "jit" if config.jit else "eager"
        subdir = f"{config.name}-{config.test}-{mode}"
        return subdir

    def result_to_output_metrics(metrics: TorchBenchModelMetrics) -> Dict[str, float]:
        result_metrics = {}
        if metrics:
            if metrics.latencies:
                latency_metric = "latency"
                median_latency = numpy.median(metrics.latencies)
                assert median_latency, f"Run failed for metric {latency_metric}"
                result_metrics[latency_metric] = median_latency
            if metrics.cpu_peak_mem:
                cpu_peak_mem = "cpu_peak_mem"
                result_metrics[cpu_peak_mem] = metrics.cpu_peak_mem
        return result_metrics

    def dump_result_to_json(metrics, output_dir):
        result = get_output_json(BM_NAME, metrics)
        dump_output(BM_NAME, result, output_dir)

    def run_config(config: TorchBenchModelConfig, metrics: List[str], dryrun: bool=False) -> Optional[TorchBenchModelMetrics]:
        """This function only handles NotImplementedError, all other errors will fail."""
        print(f"Running {config} ...", end='')
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
            result: TorchBenchModelMetrics = get_model_test_metrics(model, metrics=metrics)
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
            jit=args.jit,
            extra_args=extra_args,
            extra_env=None)
        try:
            metrics = run_config(config, metrics, dryrun=args.dryrun)
        except KeyboardInterrupt:
            print("User keyboard interrupted!")
        if not args.dryrun:
            args.output = args.output if args.output else get_output_dir(BM_NAME)
            target_dir = Path(args.output).joinpath(get_output_subdir(config))
            target_dir.mkdir(exist_ok=True, parents=True)
            metrics_dict = result_to_output_metrics(metrics)
            dump_result_to_json(metrics_dict, target_dir)

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", "-d", default="cpu", help="Devices to run.")
        parser.add_argument("--test", "-t", default="eval", help="Tests to run.")
        parser.add_argument("--model", "-m", default=None, type=str, help="Only run the specifice model.")
        parser.add_argument("--batch-size", "-b", default=None, type=int, help="Run the specifice batch size.")
        parser.add_argument("--jit", action="store_true", help="Convert the models to jit mode.")
        parser.add_argument("--output", "-o", default=None, help="Output dir.")
        parser.add_argument("--metrics", default="latencies", help="Benchmark metrics, split by comma.")
        parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
        args, extra_args = parser.parse_known_args()
        run(args, extra_args)
