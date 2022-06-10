from typing import List
import submitit
import torch
from torchbenchmark.util.distributed.submit import parse_args, get_init_file, TrainerWrapper
from ..utils import dump_output

BM_NAME = "distributed"

def gen_metrics_from_result(result):
    assert isinstance(result, List), "The result of submitit should be a list."
    metrics = {}
    for result_id, r in enumerate(result):
        for metric_name in r:
            metrics[f"{result_id}-{metric_name}"] = r[metric_name]
    return metrics

def run(args: List[str]):
    args = parse_args(args)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=3000)

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
    )

    executor.update_parameters(name="distbench", slurm_array_parallelism=1, timeout_min=1000)

    args.dist_url = get_init_file(args).as_uri()
    args.output_dir = args.job_dir

    job = executor.submit(TrainerWrapper(args))

    # waits for completion and returns output
    result = job.results()
    # dump the output file
    output = {
        "name": BM_NAME,
        "environ": {"pytorch_git_version": torch.version.git_version},
        "metrics": gen_metrics_from_result(result),
    }
    dump_output(BM_NAME, output)
