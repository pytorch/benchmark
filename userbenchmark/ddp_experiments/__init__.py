import argparse
import importlib
import os
import copy
import csv
import dataclasses
import functools
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
import warnings

from pathlib import Path
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import Any, Dict, List, Optional, Tuple

MODEL_PATH_TEMPLATE = "torchbenchmark.models.{}.Model"

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
        default=None,
        type=int,
        action="extend",
        nargs="+",
        help="Number of nodes to request. Provide a list of nodes to test, e.g. `--nodes 8 4 2 1 --next_arg..."
    )
    parser.add_argument(
        "--filter_models",
        default=None,
        type=str,
        action="extend",
        nargs="+",
        help="List of models to test, e.g. --filter hf_T5 hf_T5_large resnet50"
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
    parser.add_argument(
        "--check_correctness_distributed",
        action='store_true',
        help="Do distributed correctness checks. Don't expect to use the same results for performance tests."
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
class ExperimentParams:
    config: Dict
    args: Any  # arguments to the distributed trainer
    model_args: Any  # arguments to the model
    is_reference: bool  # should this experiment be treated as a reference for correctness?


# used for labeling filenames for correctness checks
def serialize_config(config: Dict):
    keys = ["nodes", "model_name", "backend", "has_breaks"]
    return "-".join([f"{k}_{config[k]}" for k in keys if k in config])


@dataclasses.dataclass
class JobConfig:
    outer_sync_path: str


class TrainerWrapper(object):
    # per_experiment_args is a list of expriments.
    # Each experiment should be a tuple of (config dict, args, model_args).
    # config: configuration data to attach to the result dict.
    # args & model_args: arguments for core_model.Trainer.
    def __init__(self, job_config: JobConfig, per_experiment_args: List[ExperimentParams]):
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

        # maps all configs that are expected to have the same output/gradients to the same value.
        # i.e. we should expect that for a given model_name & number of nodes, we should get the same
        #      outputs and gradients, regardless of the backend/has_breaks/etc.
        def reference_key(config):
            return f"{config['model_name']}-{config['nodes']}"
        latest_reference_file = {}

        output_dir = self.per_experiment_args[0].args.output_dir
        base_ref_name = Path(output_dir) / uuid.uuid4().hex

        for experiment_args in self.per_experiment_args:
            config = experiment_args.config
            args = experiment_args.args
            model_args = experiment_args.model_args
            is_reference = experiment_args.is_reference
            try:
                key = reference_key(config)

                if args.check_correctness_distributed:
                    # if this is a reference, dump the gradients into a file for later use.
                    # if this is not a reference, read the dumped gradients and compare.
                    if is_reference:
                        args.check_correctness_distributed = "reference"
                        args.reference_data_path = f"{base_ref_name}-{serialize_config(config)}"
                        latest_reference_file[key] = args.reference_data_path
                    else:
                        args.check_correctness_distributed = "test"
                        args.reference_data_path = latest_reference_file[key] if key in latest_reference_file else None
                else:
                    args.check_correctness_distributed = None


                if job_env.node >= args.nodes:
                    continue
                result_dict = {**config}
                q = multiprocessing.Queue()
                proc = multiprocessing.Process(target=self.run_once, args=(args, model_args, q))
                proc.start()

                # wait for 3 minutes less than timeout, to give some buffer time so that
                # the barrier doesn't time out.
                # 3 minutes chosen based on 3x the 60s timeout for killing & joining jobs
                # that are timing out.
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
                    proc.join(timeout=1)
                    if proc.exitcode is not None:
                        got_exit = True
                        exit_code = proc.exitcode

                    if not got_result:
                        try:
                            result = q.get(timeout=1)
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

    def _global_rank(self):
        job_env = submitit.JobEnvironment()
        return job_env.global_rank

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

def get_node_list(args):
    node_list = args.nodes
    if node_list is None:
        # run the 8-node version first so that all the caches get warmed up at the same time.
        node_list = [8, 4, 2, 1]
    return node_list

# takes `models` as a list of models in shortened form (i.e. not containing MODEL_PATH_TEMPLATE).
def filter_models(args, models: List[str]):
    if args.filter_models is None:
        return models
    final_models = []
    for m in args.filter_models:
        if m in models:
            final_models.append(m)
        else:
            warnings.warn(f"Model {m} was specified but is unsupported.")

    return final_models


def benchmark_ddp(args, executor):
    available_models = [
        'hf_Bert',
        'hf_GPT2_large',
        'hf_T5_large',
        'timm_vision_transformer_large',
        'hf_T5',
        'resnet50',
    ]

    models = [MODEL_PATH_TEMPLATE.format(m) for m in filter_models(args, available_models)]

    model_batch_size = {
        'hf_Bert': 32,
        'hf_GPT2_large': 4,
        'hf_T5_large': 4,
        'timm_vision_transformer_large': 16,
        'hf_T5': 12,
        'resnet50': 128,
    }
    model_batch_size = {MODEL_PATH_TEMPLATE.format(k): v for k, v in model_batch_size.items()}
    # put eager first to ensure it can be used for reference values.
    # try --torchdynamo eager or --torchdynamo aot_eager for debugging
    model_args_configs = [
        [],  # no args = pure eager baseline
        ["--torchdynamo", "inductor"],
    ]
    node_list = get_node_list(args)

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
                        is_reference = (backend_name == "eager")
                        # copy the model args so we can add more arguments without modifying
                        # the original model_args list.
                        copied_model_args = copy.copy(model_args)
                        breakname = "withbreaks" if has_breaks else "nobreaks"
                        if has_breaks:
                            copied_model_args.append("--optimize_dynamo_ddp")
                        if "inductor" in backend_name:
                            copied_model_args.extend(["--torchinductor_cudagraph", "False"])
                        if backend_name != "eager":
                            copied_model_args.extend(["--dynamo_disable_optimizer_step", "True"])

                        # skip non-distributed correctness checks to avoid extra iterations which can
                        # interfere with distributed correctness checks.
                        copied_model_args.append("--skip_correctness")
                        if args.check_correctness_distributed and "inductor" in backend_name:
                            copied_model_args.extend(["--torchinductor_fallback_random", "True"])

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
                        experiments.append(ExperimentParams(config, args_copy, copied_model_args, is_reference))

    allocation_nodes = max(node_list)
    executor.update_parameters(
        nodes=allocation_nodes,
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

def apply_fsdp(model, trainer, auto_wrap_policy):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    assert trainer == "fsdp"
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )
    return fsdp_model

def apply_fsdp_hf_T5_large(model, trainer):
    from transformers.models.t5.modeling_t5 import T5Block
    return apply_fsdp(
        model,
        trainer,
        functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=(T5Block,)),
    )

def apply_fsdp_hf_GPT2_large(model, trainer):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    return apply_fsdp(
        model,
        trainer,
        functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=(GPT2Block,)),
    )

def apply_fsdp_hf_Bert_large(model, trainer):
    from transformers.models.bert.modeling_bert import BertLayer
    return apply_fsdp(
        model,
        trainer,
        functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=(BertLayer,)),
    )

def apply_fsdp_timm_VIT_large(model, trainer):
    from timm.models.vision_transformer import Block
    return apply_fsdp(
        model,
        trainer,
        functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=(Block,)),
    )

def benchmark_fsdp(args, executor):
    def get_backend_name(model_args):
        if "--torchdynamo" in model_args:
            return "torchdynamo_" + model_args[model_args.index("--torchdynamo") + 1]
        return "eager"

    def generic_setup(nodes, model_args):
        backend_name = get_backend_name(model_args)
        copied_model_args = copy.copy(model_args)
        if "inductor" in backend_name:
            copied_model_args.extend(["--torchinductor_cudagraph", "False"])
        if backend_name != "eager":
            copied_model_args.extend(["--dynamo_disable_optimizer_step", "True"])
        copied_model_args.append("--skip_correctness")
        if args.check_correctness_distributed and "inductor" in backend_name:
            copied_model_args.extend(["--torchinductor_fallback_random", "True"])

        args_copy = copy.deepcopy(args)

        args_copy.nodes = nodes
        args_copy.dist_url = get_init_file(args).as_uri()
        args_copy.output_dir = args.job_dir

        return args_copy, copied_model_args

    def fsdp_is_reference(backend_name):
        return backend_name == "eager"

    def get_model_config(
        nodes,
        model_args,
        model_name,
        wrap_fn,
        batch_size_per_nodes,
    ):
        model_path = MODEL_PATH_TEMPLATE.format(model_name)
        args_copy, copied_model_args = generic_setup(nodes, model_args)
        copied_model_args.extend(["--distributed_wrap_fn", wrap_fn])

        assert nodes in batch_size_per_nodes
        args_copy.batch_size = batch_size_per_nodes[nodes]
        args_copy.model = model_path

        backend_name = get_backend_name(model_args)
        config = {
            "nodes": nodes,
            "model_name": model_name,
            "backend": backend_name,
        }
        return ExperimentParams(config, args_copy, copied_model_args, is_reference=fsdp_is_reference(backend_name))

    model_configs = {
        "timm_vision_transformer_large": functools.partial(
            get_model_config,
            model_name="timm_vision_transformer_large",
            wrap_fn="userbenchmark.ddp_experiments.apply_fsdp_timm_VIT_large",
            batch_size_per_nodes={1: 6, 2: 6, 4: 6, 8: 6},
        ),
        "hf_GPT2_large": functools.partial(
            get_model_config,
            model_name="hf_GPT2_large",
            wrap_fn="userbenchmark.ddp_experiments.apply_fsdp_hf_GPT2_large",
            batch_size_per_nodes={1: 6, 2: 6, 4: 6, 8: 6},
        ),
        "hf_Bert_large": functools.partial(
            get_model_config,
            model_name="hf_Bert_large",
            wrap_fn="userbenchmark.ddp_experiments.apply_fsdp_hf_Bert_large",
            batch_size_per_nodes={1: 16, 2: 16, 4: 16, 8: 16},
        ),
        "hf_T5_large": functools.partial(
            get_model_config,
            model_name="hf_T5_large",
            wrap_fn="userbenchmark.ddp_experiments.apply_fsdp_hf_T5_large",
            batch_size_per_nodes={1: 6, 2: 6, 4: 6, 8: 6},
        ),
    }

    selected_models = filter_models(args, [k for k, _ in model_configs.items()])
    model_configs = {k: v for k, v in model_configs.items() if k in selected_models}

    model_args_configs = [
        [],  # no args = pure eager baseline
        ["--torchdynamo", "inductor"],
    ]

    node_list = get_node_list(args)

    experiments = []
    for i in range(args.repeat):
        for nodes in node_list:
            for model_name, config_generator in model_configs.items():
                for model_args in model_args_configs:
                    experiments.append(config_generator(nodes, model_args))

    allocation_nodes = max(node_list)
    executor.update_parameters(
        nodes=allocation_nodes,
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


def main():
    args = parse_args()

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=3000)

    executor.update_parameters(
        gpus_per_node=args.ngpus,
        # one task per GPU
        tasks_per_node=args.ngpus,
        cpus_per_task=12,
        timeout_min=args.timeout,
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_exclude=args.exclude,
    )

    executor.update_parameters(name="distbench", slurm_array_parallelism=1, timeout_min=1000)

    if "ddp" in args.distributed:
        benchmark_ddp(args, executor)
    elif "fsdp" in args.distributed:
        benchmark_fsdp(args, executor)


if __name__=="__main__":
    import torch
    if torch.version.debug:
        raise RuntimeError("torch.version.debug == True, which is disallowed because " \
            "NCCL performance is drastically worse when debug is on. Build with " \
            "DEBUG=0 python setup.py [develop|install|bdist_wheel] instead."
        )
    main()
