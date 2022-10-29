import argparse
import importlib
import os
import copy
import csv
import dataclasses
import io
import json
import multiprocessing
import queue
import submitit
import time
from datetime import datetime, timedelta
import sys
import torch
import uuid

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def output_csv(filename, headers, row):
    assert filename
    existed = os.path.exists(filename)
    output = csv.writer(
        io.TextIOWrapper(
            open(filename, "ab", buffering=0),
            "utf-8",
            write_through=True,
        ),
        lineterminator="\n",
    )
    if not existed:
        output.writerow(headers)
    output.writerow([(f"{x:.4f}" if isinstance(x, float) else x) for x in row])



def parse_args(args: List[str]=None):
    parser = argparse.ArgumentParser(description='Submitit for PyTorch Distributed Benchmark', add_help=False)

    parser.add_argument(
        "--ngpus",
        default=8,
        type=int,
        help="Number of gpus to request on each node"
    )

    parser.add_argument(
        "--nodes",
        default=1,
        type=int,
        help="Number of nodes to request"
    )

    parser.add_argument(
        "--timeout",
        default=120,
        type=int,
        help="Duration of the job"
    )

    parser.add_argument(
        "--profiler",
        default=False,
        type=bool,
        help="Measure with PyTorch Profiler. Disabled by default, as it crashes on AWS"
    )

    parser.add_argument(
        "--partition",
        default="train",
        type=str,
        help="The Slurm partition to submit to"
    )

    parser.add_argument(
        "--distributed",
        default="ddp_no_static_graph",
        type=str,
        help="the distributed runner to use"
    )

    parser.add_argument(
        "--job_dir",
        default=os.getcwd(),
        type=str,
        help="A shared folder across all worker processes"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="torchbenchmark.models.hf_Bert.Model",
        help="specify the model to experiment with, by default uses e2e_models.hf_bert"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="specify the batch size of the input"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="torchbenchmark.util.distributed.core_model.trainer.Trainer",
        help="training paradigm, by default using DDP"
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=f"ddp_experiments_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        help="training paradigm, by default using DDP"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="comma-separated list of nodes to exclude from the slurm allocation",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="number of times to repeat the experiments",
    )


    try:
        if args:
            return parser.parse_args(args)
        else:
            return parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(args.job_dir, exist_ok=True)
    init_file = Path(args.job_dir) / f"{uuid.uuid4().hex}_init"
    print(init_file)
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


# This implements a barrier function, where all processes wait until they all
# reach the barrier() call.
# rank: there should be one
class FileBarrier:
    def __init__(self, rank, world_size, sync_file, timeout: Optional[timedelta] = None):
        self.rank = rank
        self.world_size = world_size
        self.sync_file = sync_file
        self.store = torch.distributed.FileStore(sync_file, world_size)
        if timeout is None:
            timeout = timedelta(minutes=30)
        self.store.set_timeout(timeout)
        self.call_idx = 0
        self.barrier()

    def barrier(self):
        self.call_idx += 1
        my_key = f"barrier{self.call_idx}.{self.rank}"
        self.store.add(my_key, 1)
        wait_for = []
        for i in range(self.world_size):
            key = f"barrier{self.call_idx}.{i}"
            wait_for.append(key)
        self.store.wait(wait_for)


@dataclasses.dataclass
class JobConfig:
    outer_sync_path: str


class TrainerWrapper(object):
    # per_experiment_args is a list of expriments.
    # Each experiment should be a tuple of (config dict, args, model_args).
    # config: configuration data to attach to the result dict.
    # args & model_args: arguments for core_model.Trainer.
    def __init__(self, job_config: JobConfig, per_experiment_args: List[Tuple[Dict, Any, Any]]):
        self.job_config = job_config
        self.per_experiment_args = per_experiment_args
        self.timeout = timedelta(45)

    # this is called within a multiprocessing.Process.
    def run_once(self, args, model_args, q):
        print("run_once")
        self._setup_gpu_args(args)

        pos = args.model.rfind(".")
        module = importlib.import_module(args.model[:pos])
        model_class = getattr(module, args.model[(pos+1):])

        pos = args.trainer.rfind(".")
        module = importlib.import_module(args.trainer[:pos])
        trainer_class = getattr(module, args.trainer[(pos+1):])

        trainer = trainer_class(args, model_class, model_args=model_args)
        result = trainer.measure()
        print(f"result {result}")
        q.put(result)
        trainer.teardown()

    def __call__(self):
        results = []
        job_env = submitit.JobEnvironment()
        barrier = self._get_barrier()
        print(f"This is node {job_env.node}")
        for (config, args, model_args) in self.per_experiment_args:
            try:
                if job_env.node >= args.nodes:
                    continue
                result_dict = {**config}
                q = multiprocessing.Queue()
                proc = multiprocessing.Process(target=self.run_once, args=(args, model_args, q))
                proc.start()

                # wait for 3 minutes less than timeout, to give some buffer time so that
                # the barrier doesn't time out
                timeout_seconds = (self.timeout - timedelta(minutes=3)).total_seconds()

                # Wait in a loop because:
                # - the queue has a limited buffer size, so we need to call q.get() before proc.join()
                #   in case the queue blocks when the worker process tries to put into the queue
                # - if the worker process errors out, nothing will get put into the queue when it
                #   exits early and then we end up waiting until the timeout finishes
                # So we wait in a loop and wait until either finishes
                got_result = False
                got_exit = False
                exit_code = None
                result = None
                start_time = time.time()
                while time.time() < start_time + timeout_seconds and not got_exit:
                    proc.join(timeout=5)
                    if proc.exitcode is not None:
                        got_exit = True
                        exit_code = proc.exitcode

                    if not got_result:
                        try:
                            result = q.get(timeout=5)
                            got_result = True
                        except queue.Empty:
                            pass
                if not got_exit:
                    proc.kill()
                    proc.join(timeout=60)

                proc.close()

                if isinstance(result, dict) and 'latency_median' in result:
                    result_dict['result'] = result
                else:
                    result_dict['result'] = None
                print(f"exit code: {exit_code} and result: {result_dict}")
                assert 'result' in result_dict
                # wrap in <RESULT></RESULT> so we can parse partial results in the stdout logs
                print(f"<RESULT>{json.dumps(result_dict)}</RESULT>")
                results.append(result_dict)
            finally:
                barrier.barrier()

        return results

    def checkpoint(self):
        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args, self.model_args)
        empty_trainer = type(self)(self.args, self.model_args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _get_barrier(self):
        job_env = submitit.JobEnvironment()
        rank = job_env.global_rank
        world_size = job_env.num_tasks
        return FileBarrier(
            rank=rank,
            world_size=world_size,
            sync_file=self.job_config.outer_sync_path,
            timeout=self.timeout
        )

    def _setup_gpu_args(self, args):
        job_env = submitit.JobEnvironment()
        args.output_dir = Path(str(args.output_dir).replace("%j", str(job_env.job_id)))
        args.gpu = job_env.local_rank
        args.rank = job_env.global_rank
        args.world_size = args.ngpus * args.nodes
        print(f"Process group: {args.world_size} tasks, rank: {args.rank}")

        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["GPUS_PER_NODE"] = str(job_env.num_tasks//job_env.num_nodes)
        # os.environ["NCCL_IB_DISABLE"] = str(1)
        os.environ["NCCL_DEBUG"] = 'INFO'
        os.environ["NCCL_DEBUG_SUBSYS"] = 'INIT,ENV,NET'
        os.environ['NCCL_SOCKET_IFNAME'] = 'ens'
        # os.environ["NCCL_ALGO"] = 'ring'
        os.environ["FI_PROVIDER"] = 'efa'
        os.environ["FI_EFA_USE_DEVICE_RDMA"]= str(1)
        os.environ["NET_TYPE"] = 'efa'
        os.environ["ADAM_CAPTURABLE"] = str(1)


def main():
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
        slurm_partition=args.partition if args.nodes < 16 else 'scavenge',
        slurm_signal_delay_s=120,
        slurm_exclude=args.exclude,
    )

    executor.update_parameters(name="distbench", slurm_array_parallelism=1, timeout_min=1000)


    # args.dist_url = get_init_file(args).as_uri()
    # args.output_dir = args.job_dir
    # job = executor.submit(TrainerWrapper(args))
    #     # print ID of the Slurm job
    # print(job.job_id)

    # # waits for completion and returns output
    # print(job.results())

    models = [
        # 'torchbenchmark.models.hf_Bert.Model',
        # # 'torchbenchmark.models.hf_BertLarge.Model',
        # 'torchbenchmark.models.hf_GPT2_large.Model',
        # 'torchbenchmark.models.hf_T5_large.Model',
        # 'torchbenchmark.models.timm_vision_transformer_large.Model',
        # # 'torchbenchmark.models.hf_GPT2.Model',
        # 'torchbenchmark.models.hf_T5.Model',
        'torchbenchmark.models.resnet50.Model',
    ]

    model_batch_size = {
        'torchbenchmark.models.hf_Bert.Model': 32,
        'torchbenchmark.models.hf_BertLarge.Model': 16,
        'torchbenchmark.models.hf_GPT2_large.Model': 4,
        'torchbenchmark.models.hf_T5_large.Model': 4,
        'torchbenchmark.models.timm_vision_transformer_large.Model': 16,
        'torchbenchmark.models.hf_GPT2.Model': 24,
        'torchbenchmark.models.hf_T5.Model': 12,
        'torchbenchmark.models.resnet50.Model': 32,
    }
    model_args_configs = [
        [],  # no args = pure eager baseline
        # ["--torchdynamo", "eager"],  # runs dynamo without a backend
        # ["--torchdynamo", "aot_nvfuser"],
        # ["--torchdynamo", "inductor"],
    ]
    # node_list = [1, 2, 4, 8, 12, 16, 20, 24]
    node_list = [1]

    def get_backend_name(model_args):
        if "--torchdynamo" in model_args:
            return "torchdynamo_" + model_args[model_args.index("--torchdynamo") + 1]
        return "eager"

    experiments = []
    for i in range(args.repeat):
        for nodes in node_list:
            for model_name in models:
                for model_args in model_args_configs:
                    for has_breaks in [True, False]:
                        backend_name = get_backend_name(model_args)
                        if backend_name == "eager" and has_breaks:
                            continue
                        # copy the model args so we can add more arguments without modifying
                        # the original model_args list.
                        copied_model_args = copy.copy(model_args)
                        breakname = "withbreaks" if has_breaks else "nobreaks"
                        if has_breaks:
                            copied_model_args.append("--optimize_dynamo_ddp")
                        if "inductor" in backend_name:
                            copied_model_args.append("--torchinductor_cudagraph False")
                        batch_size = model_batch_size[model_name]
                        args_copy = copy.deepcopy(args)
                        args_copy.model = model_name
                        args_copy.batch_size = batch_size
                        args_copy.nodes = nodes
                        args_copy.dist_url = get_init_file(args).as_uri()
                        args_copy.output_dir = args.job_dir
                        config = {
                            "nodes": nodes,
                            "model_name": model_name,
                            "backend": backend_name,
                            "has_breaks": has_breaks,
                        }
                        experiments.append((config, args_copy, copied_model_args))

    allocation_nodes = max(node_list)
    executor.update_parameters(
        gpus_per_node=args.ngpus,
        # one task per GPU
        tasks_per_node=args.ngpus,
        cpus_per_task=10,
        nodes=allocation_nodes,
        timeout_min=args.timeout,
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_exclude=args.exclude,
    )
    job_config = JobConfig(
        outer_sync_path=str(get_init_file(args))
    )
    job = executor.submit(TrainerWrapper(job_config, experiments))

    # print ID of the Slurm job
    print(f"{allocation_nodes} nodes: {job.job_id}")
    output_csv(
        args.index_file,
        ("job_id",),
        (job.job_id,),
    )

    # waits for completion and returns output
    print(job.results())


if __name__=="__main__":
    import torch
    if torch.version.debug:
        raise RuntimeError("torch.version.debug == True, which is disallowed because " \
            "NCCL performance is drastically worse when debug is on. Build with " \
            "DEBUG=0 python setup.py [develop|install|bdist_wheel] instead."
        )
    main()
