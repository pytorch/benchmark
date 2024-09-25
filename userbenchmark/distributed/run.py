from typing import List

import torch
from torchbenchmark.util.distributed.submit import (
    get_init_file,
    parse_args,
    TrainerWrapper,
)

from ..utils import dump_output

BM_NAME = "distributed"


def gen_metrics_from_result(result):
    assert isinstance(result, List), "The result should be a list."
    metrics = {}
    for result_id, r in enumerate(result):
        for metric_name in r:
            metrics[f"{result_id}-{metric_name}"] = r[metric_name]
    return metrics


def run(args: List[str]):
    args, model_args = parse_args(args)

    if args.scheduler == "slurm":
        result = slurm_run(args, model_args)
    elif args.scheduler == "local":
        result = local_run(args, model_args)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    version = (
        torch.version.git_version
        if hasattr(torch.version, "git_verison")
        else "Internal"
    )

    # dump the output file
    output = {
        "name": BM_NAME,
        "environ": {"pytorch_git_version": version},
        "args": vars(args),
        "metrics": gen_metrics_from_result(result),
    }
    dump_output(BM_NAME, output)


def local_run(args, model_args):
    # TODO: Currently this does nothing but to support the path for "--scheduler local"
    print(
        "Current local run is not implemented, use '--scheduler slurm'. Skipping local run."
    )
    return []


def slurm_run(args, model_args):
    import submitit

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(
        folder=args.job_dir, cluster=args.cluster, slurm_max_num_timeout=3000
    )

    executor.update_parameters(
        gpus_per_node=args.ngpus,
        # one task per GPU
        tasks_per_node=args.ngpus,
        cpus_per_task=10,
        nodes=args.nodes,
        timeout_min=args.timeout,
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_exclude=args.exclude,
    )

    executor.update_parameters(
        name="distbench", slurm_array_parallelism=1, timeout_min=1000
    )

    args.dist_url = get_init_file(args).as_uri()
    args.output_dir = args.job_dir
    args.extra_args = []
    if model_args:
        args.extra_args = model_args

    job = executor.submit(TrainerWrapper(args, model_args))

    # waits for completion and returns output
    result = job.results()
    return result
