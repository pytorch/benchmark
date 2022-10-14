from datetime import datetime
import os
from pathlib import Path
from statistics import stdev

import numpy as np
import torch
from torch.cuda import Event
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torchbenchmark.util.model import BenchmarkModel
import torch.distributed as dist

class Trainer():
    DEFAULT_MEASURE_ITERATIONS = 10

    def __init__(self, args, model_class, mode="SPMD", model_args=None):
        def print_env_var(env_name):
            print(env_name, ":", os.getenv(env_name))

        import socket
        print("MY HOSTNAME:", socket.gethostname())
        print_env_var("FI_PROVIDER")
        print_env_var("LD_LIBRARY_PATH")
        print_env_var("NCCL_DEBUG")
        print_env_var("FI_EFA_USE_DEVICE_RDMA")

        self.args = args
        self.model_args = model_args
        self.model_class = model_class
        self.mode = mode

        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.setup()

        # specify the name of the distributed trainer
        extra_args = [
            "--distributed",
            self.args.distributed,
        ]
        extra_args.extend(model_args)

        # create model instance after Trainer setup, so that
        # visible devices won't be revised in model constructor
        self.benchmark: BenchmarkModel = model_class(test="train", device="cuda", batch_size=None, extra_args=extra_args)

        self.rank = dist.get_rank()


    def setup(self):
        if self.mode == "SPMD":
            # set the visible devices so that each SPMD process only sees one
            # CUDA device
            # N.B.: this has to be done before using any CUDA API from torch
            # N.B.: Remove the following block as HF Accelerator by default puts
            # the model to the device corresponding to LOCAL_RANK. It's better
            # to use CUDA_VISIBLE_DEVICES and cuda:0 if HF Accelerator can avoid
            # using local_rank as the device id.
            """
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"
            assert torch.cuda.device_count() == 1, (
                "SPMD Trainer expects 1 visible device per process, but saw "
                f"{torch.cuda.device_count()} devices."
            )
            """
            torch.cuda.set_device(self.local_rank)

            world_size = int(os.getenv("WORLD_SIZE", -1))
            rank = int(os.getenv("RANK", -1))

            assert self.local_rank != -1 and world_size != -1 and rank != -1, (
                "Failed to retrieve SPMD configurations from environment "
                f"variables. local_rank={self.local_rank}, world_size={world_size}, "
                f"rank={rank}."

            )

            # TODO: hardcode NCCL for now, make this configurable if necessary
            dist.init_process_group("nccl", init_method=self.args.dist_url, rank=rank, world_size=world_size)
        else:
            raise ValueError(f"Unrecognized distributed training mode {self.mode}")

    def measure(self):
        niters = self.DEFAULT_MEASURE_ITERATIONS

        ######################################
        # 1. warming up CUDACachingAllocator #
        ######################################
        for _ in range(self.DEFAULT_MEASURE_ITERATIONS):
            self.benchmark.invoke()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)

        now = datetime.now()
        name = f"{type(self).__name__}_{now.strftime('%Y_%m_%d_%H_%M_%S')}"
        ##################################################################
        # 2. measure raw delays and memory to rule out profiler overhead #
        ##################################################################
        events_pre_train = [Event(enable_timing=True) for _ in range(niters)]
        events_post_train = [Event(enable_timing=True) for _ in range(niters)]
        for i in range(niters):
            events_pre_train[i].record()
            self.benchmark.invoke()
            events_post_train[i].record()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)

        latency_train = [pre.elapsed_time(post) for pre, post in zip(events_pre_train, events_post_train)]
        median_latency = np.median(latency_train)
        stdev_latency = stdev(latency_train)


        if self.args.profiler:
            # N.B.: disable PyTorch Profiler by default due to
            # https://github.com/pytorch/pytorch/issues/75369
            ################################################
            # 3. meausre complete metrics through profiler #
            ################################################
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, # Causes seg fault in export_chrome_trace
                with_stack=True, # Causes seg fault with EFA
                with_flops=True, # Causes seg fault in export_chrome_trace
                on_trace_ready=tensorboard_trace_handler(
                    f"{self.args.job_dir}/tb/{name}",
                    self.rank,
                    use_gzip=True,
                )
            ):
                for i in range(2):
                    self.benchmark.invoke()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)
        # wait for all peers to finish
        dist.barrier(device_ids=[self.local_rank])

        return {
            "latency_median" : median_latency,
            "latency_stdev" : stdev_latency,
        }


    def teardown(self):
        if self.mode == "SPMD":
            dist.destroy_process_group()
