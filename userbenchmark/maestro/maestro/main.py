import click
from pathlib import Path
import yaml
import pprint
import sys
from core.config_parser import parse_config
from core.orchestrator import Orchestrator
from core.axis import TorchAxis 
from core.utils.logging import get_logger, get_root_rank_logger



logger = get_logger(__name__)
root_logger = get_root_rank_logger()

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    """
    Maestro CLI - A benchmarking framework for parallel communication/compute operations.
    
    Maestro enables accurate performance measurement of AI workloads by benchmarking
    parallel operations on the same GPU, rather than isolated micro-benchmarks.
    
    Use 'python -m maestro <command> --help' for more information on a specific command.
    """
    return

@cli.command(short_help="List all available blocks.")
def list_blocks():
    """
    List all available blocks registered in the BlockRegistry.
    
    Blocks are the fundamental building units in Maestro patterns. Each block
    represents a discrete operation (e.g., NCCL collective, GEMM, custom kernel)
    that can be composed into workload patterns.
    
    Example:
        $ python -m maestro list-blocks
    """
    from core.block import BlockRegistry
    print("Available blocks:")
    for block in BlockRegistry.get_all_blocks():
        print(f"- {block.registry_name}")

@cli.command(short_help="List all available operation presets.")
def list_op_presets():
    """
    List all available operation presets registered in the OpPresetRegistry.
    
    Op presets define pre-configured operation parameters that can be reused
    across different patterns. They simplify configuration by providing
    commonly-used operation settings.
    
    Example:
        $ python -m maestro list-op-presets
    """
    from core.op_preset import OpPresetRegistry
    print("Available op presets:")
    for op_preset in OpPresetRegistry.get_all_op_presets():
        print(f"- {op_preset.name}")

@cli.command(short_help="Run benchmarks with a config file.")
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Path to pattern/config file.')
def run_benchmark(config: Path):
    """
    Run benchmarks for distributed patterns defined in a configuration file.
    
    This command parses the provided YAML configuration file, initializes the
    specified backends (axis and profiler), and executes all defined patterns
    through the Orchestrator. Results are collected and optionally saved to
    a file.
    
    The configuration file should define:
    
    \b
    - patterns: List of workload patterns to benchmark
    - axes: Axis configuration for distributed execution
    - backends: Backend settings (axis type, profiler)
    - warmup_iters: Number of warmup iterations before measurement
    - iters: Number of measurement iterations
    - save_results_to: Optional output file path (csv, json, xlsx)
    
    Supported output formats: .csv, .json, .xlsx

    Returns: Results DataFrame if rank is root, otherwise None
    Example:
        $ python -m maestro run-benchmark -c configs/allreduce_gemm.yaml
    """
    return _run_benchmark(config)

def _run_benchmark(config: Path):
    if not config.exists():
        raise ValueError(f"Config file {config} does not exist")

    try:
        config = parse_config(config)
    except (ValueError, KeyError, TypeError) as e:
        logger.exception(f"Error parsing config: {e}")
        logger.info("Usage: python -m maestro run-benchmark -c <config_file>")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error parsing config: {e}")
        raise


    ########################## Load backends ##########################
    axis_backend = config["backends"]["axis"]["name"]
    if axis_backend == "torch":
        axis_cls = TorchAxis
    else:
        raise ValueError(f"Invalid axis backend: {axis_backend}")
    
    profiler_backend = config["backends"]["profiler"]["name"]
    if profiler_backend == "cupti":
        from core.profilers import CuptiProfiler 
        profiler = CuptiProfiler(output_dir=config["backends"]["profiler"]["output_dir"])
    else:
        raise ValueError(f"Invalid profiler backend: {profiler_backend}")
    ##############################################################

    root_logger.info(f"Running with config:\n{pprint.pformat(config)}")

    # Launch orchestrator
    orchestrator = Orchestrator(
        patterns=config["patterns"], 
        axes_cfg=config["axes"],
        profiler=profiler, 
        axis_cls=axis_cls, 
        warmup_iters=config["warmup_iters"], 
        iters=config["iters"],
        latency_precision=config["latency_precision"],
        register_buffers=config["register_buffers"],
        stop_on_error=config["stop_on_error"]
    )
    results_df = orchestrator.run_all_patterns()
    if results_df is None:
        return

    save_results_to = config["save_results_to"]
    if save_results_to:
        extension = save_results_to.suffix.lower().lstrip(".")
        if extension == "csv":
            results_df.to_csv(save_results_to, index=False)
        elif extension == "json":
            results_df.to_json(save_results_to, index=False)
        elif extension == "xlsx":
            results_df.to_excel(save_results_to, index=False)
        else:
            results_df.to_csv(save_results_to, index=False)
            logger.error(f"Invalid extension: {extension}, supported extensions are: csv, json, saving as csv.")

        root_logger.info(f"Saved results to {config['save_results_to'].absolute()}")

    orchestrator.destroy()

    return results_df



if __name__ == '__main__':
    cli()
