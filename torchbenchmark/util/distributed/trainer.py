import argparse
from datetime import datetime
import os
from pathlib import Path
from statistics import stdev

import torch
from torch.cuda import Event
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torchbenchmark.util.e2emodel import E2EBenchmarkModel, nested
import torch.distributed as dist

def parse_model_args(model_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cuda", help="Which device to use.")
    return parser.parse_known_args(model_args)

class Trainer():
    DEFAULT_MEASURE_ITERATIONS = 10

    def __init__(self, args, model_class, mode="SPMD", model_args=None):
        self.args = args
        known_args, other_model_args = parse_model_args(model_args)
        self.model_args = other_model_args
        self.model_class = model_class
        self.mode = mode

        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.setup()

        extra_args = [
            "--distributed",
            self.args.distributed,
        ]
        extra_args.extend(self.model_args)

        # create model instance after Trainer setup, so that
        # visible devices won't be revised in model constructor
        self.e2e_benchmark: E2EBenchmarkModel = model_class("train", known_args.device, batch_size=None, extra_args=extra_args)

        expected_attrs = ["model", "optimizer", "train_dataloader", "accelerator", "run_contexts"]
        assert all(attr in dir(self.e2e_benchmark) for attr in expected_attrs), (
            "Missing attributes in the input E2EBenchmarkModel implementation: "
            f"{[attr for attr in expected_attrs if attr not in dir(self.e2e_benchmark)]}"
        )

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
        # TODO: using dummy data for now to rule out dataloader delays
        batch = self.e2e_benchmark.next_batch()

        ######################################
        # 1. warming up CUDACachingAllocator #
        ######################################
        for _ in range(self.DEFAULT_MEASURE_ITERATIONS):
            with nested(*self.e2e_benchmark.run_contexts):
                loss = self.e2e_benchmark.run_forward(batch)
                self.e2e_benchmark.run_backward(loss)
                self.e2e_benchmark.run_optimizer_step()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)

        now = datetime.now()
        name = f"{type(self).__name__}_{now.strftime('%Y_%m_%d_%H_%M_%S')}"
        ##################################################################
        # 2. measure raw delays and memory to rule out profiler overhead #
        ##################################################################
        events_pre_fwd = [Event(enable_timing=True) for _ in range(niters)]
        events_pre_bwd = [Event(enable_timing=True) for _ in range(niters)]
        events_pre_opt = [Event(enable_timing=True) for _ in range(niters)]
        events_post_opt = [Event(enable_timing=True) for _ in range(niters)]
        with nested(*self.e2e_benchmark.run_contexts):
            for i in range(niters):
                events_pre_fwd[i].record()
                loss = self.e2e_benchmark.run_forward(batch)

                events_pre_bwd[i].record()
                self.e2e_benchmark.run_backward(loss)

                events_pre_opt[i].record()
                self.e2e_benchmark.run_optimizer_step()

                events_post_opt[i].record()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)

        delays_fwd = [pre.elapsed_time(post) for pre, post in zip(events_pre_fwd, events_pre_bwd)]
        delays_bwd = [pre.elapsed_time(post) for pre, post in zip(events_pre_bwd, events_pre_opt)]
        delays_opt = [pre.elapsed_time(post) for pre, post in zip(events_pre_opt, events_post_opt)]

        mean_fwd = float(sum(delays_fwd)) / len(delays_fwd)
        stdev_fwd = stdev(delays_fwd)
        mean_bwd = float(sum(delays_bwd)) / len(delays_bwd)
        stdev_bwd = stdev(delays_bwd)
        mean_opt = float(sum(delays_opt)) / len(delays_opt)
        stdev_opt = stdev(delays_opt)
        iter_time = events_pre_fwd[0].elapsed_time(events_post_opt[-1]) / niters

        # write results
        delay_dir = f"{self.args.job_dir}/delay"
        Path(delay_dir).mkdir(parents=True, exist_ok=True)
        fout = open(f"{delay_dir}/{name}.log", "w")
        fout.write(
            f"{mean_fwd:.2f}, {stdev_fwd:.2f}, "
            f"{mean_bwd:.2f}, {stdev_bwd:.2f}, "
            f"{mean_opt:.2f}, {stdev_opt:.2f}\n"
        )
        fout.close()

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
                for i in range(niters):
                    loss = self.e2e_benchmark.run_forward(batch)
                    self.e2e_benchmark.run_backward(loss)
                    self.e2e_benchmark.run_optimizer_step()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)
        # wait for all peers to finish
        dist.barrier(device_ids=[self.local_rank])

        return {
            "iter" : iter_time,
            "fwd_mean" : mean_fwd,
            "fwd_stdev" : stdev_fwd,
            "bwd_mean" : mean_bwd,
            "bwd_stdev" : stdev_bwd,
            "opt_mean" : mean_opt,
            "opt_stdev" : stdev_opt,
        }


    def teardown(self):
        if self.mode == "SPMD":
            dist.destroy_process_group()
