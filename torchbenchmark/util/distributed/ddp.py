from datetime import datetime
from statistics import stdev
from pathlib import Path

from .trainer import Trainer

from torch.cuda import Event
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from torchbenchmark.util.e2emodel import E2EBenchmarkModel

import torch
import torch.distributed as dist

class DDPTrainer(Trainer):
    DEFAULT_MEASURE_ITERATIONS = 10
    def __init__(self, args, model_class, batch_size=None, extra_args=[]):
        super().__init__(args, model_class, mode="SPMD")

        self.setup()

        # create model instance after Trainer setup, so that
        # visible devices won't be revised in model constructor
        model: E2EBenchmarkModel = model_class("train", batch_size, extra_args)

        expected_attrs = ["model", "optimizer", "train_dataloader", "accelerator"]
        assert all(attr in dir(model) for attr in expected_attrs), (
            "Missing attributes in the input E2EBenchmarkModel implementation: "
            f"{[attr for attr in expected_attrs if attr not in dir(model)]}"
        )

        self.model = model.model
        self.optimizer = model.optimizer
        self.dataloader = model.train_dataloader
        self.accelerator = model.accelerator

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()


        self.ddp_model = DDP(
            self.model,
            device_ids=[self.local_rank],
            # If buffer broadcast is necessary, specific optimizations might be
            # necessary to optimize performance. Disable it by default.
            broadcast_buffers=False,
            # Set gradient as bucket view to avoid unnecessary copies
            gradient_as_bucket_view=True,
            # TODO: tune bucket_cap_mb
        )

    def measure(self):
        niters = self.DEFAULT_MEASURE_ITERATIONS
        # TODO: using dummy data for now to rule out dataloader delays
        batch = next(iter(self.dataloader))

        ######################################
        # 1. warming up CUDACachingAllocator #
        ######################################
        for _ in range(self.DEFAULT_MEASURE_ITERATIONS):
            loss = self.model(**batch).loss
            self.accelerator.backward(loss)
            self.optimizer.step()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)

        now = datetime.now()
        name = f"ddp_{now.strftime('%Y_%m_%d_%H_%M_%S')}"
        ##################################################################
        # 2. measure raw delays and memory to rule out profiler overhead #
        ##################################################################
        events_pre_fwd = [Event(enable_timing=True) for _ in range(niters)]
        events_pre_bwd = [Event(enable_timing=True) for _ in range(niters)]
        events_pre_opt = [Event(enable_timing=True) for _ in range(niters)]
        events_post_opt = [Event(enable_timing=True) for _ in range(niters)]
        for i in range(niters):
            events_pre_fwd[i].record()
            loss = self.model(**batch).loss

            events_pre_bwd[i].record()
            self.accelerator.backward(loss)

            events_pre_opt[i].record()
            self.optimizer.step()

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

        # write results
        Path("delay").mkdir(parents=True, exist_ok=True)
        fout = open(f"delay/{name}.log", "w")
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
                    f"tb/{name}",
                    self.rank,
                    use_gzip=True,
                )
            ):
                for i in range(niters):
                    loss = self.model(**batch).loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)
        # wait for all peers to finish
        dist.barrier(device_ids=[self.local_rank])

        return {
            "fwd_mean" : mean_fwd,
            "fwd_stdev" : stdev_fwd,
            "bwd_mean" : mean_bwd,
            "bwd_stdev" : stdev_bwd,
            "opt_mean" : mean_opt,
            "opt_stdev" : stdev_opt,
        }


def test():
    from torchbenchmark.e2e_models.hf_bert import Model

    import os

    os.environ["LOCAL_RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    os.environ["RANK"] = str(0)

    trainer = DDPTrainer(Model)

    trainer.measure()

if __name__=="__main__":
    test()

