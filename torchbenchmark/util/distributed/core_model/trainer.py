from datetime import datetime
import os
from pathlib import Path
from statistics import stdev
from typing import Optional

import numpy as np
import torch
from torch.cuda import Event
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from torchbenchmark.util.env_check import same
from torchbenchmark.util.model import BenchmarkModel
import torch.distributed as dist

class Trainer():
    DEFAULT_MEASURE_ITERATIONS = 10
    PROFILE_ITERATIONS = 2

    def __init__(self, args, model_class, mode="SPMD", model_args=None):
        self.args = args
        self.model_args = model_args
        self.model_class = model_class
        self.mode = mode

        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.global_rank = int(os.getenv("RANK", -1))
        self.setup()

        # specify the name of the distributed trainer
        extra_args = [
            "--distributed",
            self.args.distributed,
        ]
        extra_args.extend(model_args)

        batch_size = args.batch_size if hasattr(args, "batch_size") else None

        # create model instance after Trainer setup, so that
        # visible devices won't be revised in model constructor
        self.benchmark: BenchmarkModel = model_class(test="train", device="cuda", batch_size=batch_size, extra_args=extra_args)

        # options: "reference" or "test"
        self.check_correctness_distributed : Optional[str] = getattr(args, "check_correctness_distributed", None)
        self.reference_data_path : Optional[str] = getattr(args, "reference_data_path", None)

        # reduce iterations to speed up the tests
        if self.check_correctness_distributed:
            self.DEFAULT_MEASURE_ITERATIONS = 2

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

        correctness = None
        if self.check_correctness_distributed is not None:
            self.benchmark.invoke()
            if self.global_rank == 0:
                grad_params = {}
                for name, param in self.benchmark.model.named_parameters():
                    if param.requires_grad:
                        grad_params[name + ".grad"] = param.grad.cpu()

                if self.check_correctness_distributed == "reference":
                    with open(self.reference_data_path, "wb") as f:
                        torch.save(grad_params, f)
                elif self.check_correctness_distributed == "test":
                    with open(self.reference_data_path, "rb") as f:
                        ref_params = torch.load(f)

                    def do_correctness_check():
                        correctness = True
                        for ref_name, ref_param in ref_params.items():
                            if ref_name not in grad_params:
                                correctness = False
                                print(f"correctness failure: {ref_name} in reference params but not in test params")
                            test_param = grad_params[ref_name]
                            atol = rtol = 1e-4
                            if not same(test_param, ref_param, cos_similarity=False, atol=atol*40, rtol=rtol*40):
                                correctness=False
                                print(f"correctness failure: Test model differs from reference model in parameter: {ref_name}")

                        for test_name, test_param in grad_params.items():
                            if test_name not in ref_params:
                                correctness = False
                                print(f"correctness failure: {test_name} in reference params but not in ref params")
                        return correctness

                    correctness = do_correctness_check()

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
            wait_runs = 2
            warmup_runs = 2
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, # Causes seg fault in export_chrome_trace
                with_stack=True, # Causes seg fault with EFA
                with_flops=True, # Causes seg fault in export_chrome_trace
                on_trace_ready=tensorboard_trace_handler(
                    f"{self.args.job_dir}/tb/{name}",
                    self.rank,
                    use_gzip=True,
                ),
                schedule=schedule(wait=wait_runs, warmup=warmup_runs, active=self.PROFILE_ITERATIONS),
            ) as profiler:
                for i in range(self.PROFILE_ITERATIONS + warmup_runs + wait_runs):
                    self.benchmark.invoke()
                    profiler.step()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)
        # wait for all peers to finish
        dist.barrier(device_ids=[self.local_rank])

        return {
            "latency_median" : median_latency,
            "latency_stdev" : stdev_latency,
            **({"correctness": correctness} if correctness is not None else {}),
        }


    def teardown(self):
        if self.mode == "SPMD":
            dist.destroy_process_group()
