from typing import List
import submitit
from torchbenchmark.util.distributed.submit import parse_args, get_init_file, TrainerWrapper

def run(args: List[str]):
    args = parse_args()

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
    # print ID of the Slurm job
    print(job.job_id)

    # waits for completion and returns output
    print(job.results())