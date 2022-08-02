from datetime import datetime
from statistics import stdev
from pathlib import Path

import torchbenchmark

from .trainer import Trainer

from torch.cuda import Event
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from torchbenchmark.util.e2emodel import E2EBenchmarkModel
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.models import resnet50, hf_Bert, hf_BertLarge
import torch
import torch.distributed as dist
net_type = 'efa'
class DDPTrainer(Trainer):
    DEFAULT_MEASURE_ITERATIONS = 10
    def __init__(self, args, model_class, batch_size=None, extra_args=[]):
        super().__init__(args, model_class, mode="SPMD")

        self.setup()
        model: BenchmarkModel = model_class("train", self.local_rank, batch_size=batch_size, extra_args=extra_args)
        # create model instance after Trainer setup, so that
        # visible devices won't be revised in model constructor

        expected_attrs = ["model", "optimizer", "example_inputs", "device"]
        assert all(attr in dir(model) for attr in expected_attrs), (
            "Missing attributes in the input BenchmarkModel implementation: "
            f"{[attr for attr in expected_attrs if attr not in dir(model)]}"
        )

        self.model = model.model
        self.optimizer = model.optimizer
        self.model_type = type(model)
        self.batch_size = batch_size
        if(self.model_type == resnet50.Model):
            self.example_outputs = model.example_outputs
            self.loss_fn = model.loss_fn
            self.forward = self.resnet_forward
            self.forward_ddp = self.resnet_forward_ddp
        elif(self.model_type == hf_Bert.Model):
            self.forward = self.bert_forward
            self.forward_ddp = self.bert_forward_ddp
        elif(self.model_type == hf_BertLarge.Model):
            self.forward = self.bert_forward
            self.forward_ddp = self.bert_forward_ddp

        self.example_inputs = model.example_inputs
        

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
            static_graph=True,
        )
        opt_cls = type(self.optimizer)
        self.ddp_optimizer = opt_cls(self.ddp_model.parameters(), lr=0.001)

    def measure(self):
        niters = self.DEFAULT_MEASURE_ITERATIONS
        # TODO: using dummy data for now to rule out dataloader delays

        ######################################
        # 1. warming up CUDACachingAllocator #
        ######################################
        for _ in range(self.DEFAULT_MEASURE_ITERATIONS):
            loss = self.forward()
            loss.backward()
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
            loss = self.forward()

            events_pre_bwd[i].record()
            loss.backward()

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
        delay_dir = f"{self.args.job_dir}/delay"
        Path(delay_dir).mkdir(parents=True, exist_ok=True)
        fout = open(f"{delay_dir}/{name}.log", "w")
        fout.write(
            f"{mean_fwd:.2f}, {stdev_fwd:.2f}, "
            f"{mean_bwd:.2f}, {stdev_bwd:.2f}, "
            f"{mean_opt:.2f}, {stdev_opt:.2f}\n"
        )
        fout.close()


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

    def measure_allreduce(self):
        niters = self.DEFAULT_MEASURE_ITERATIONS
        # TODO: using dummy data for now to rule out dataloader delays

        ######################################
        # 1. warming up CUDACachingAllocator #
        ######################################
        for _ in range(self.DEFAULT_MEASURE_ITERATIONS):
            loss = self.forward()
            loss.backward()
            self.optimizer.step()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)
        current_size = 0
        size = 2**18
        num_tasks = self.world_size
        name = f"all_red_{net_type}_{num_tasks}_{self.gpus_per_node}_{self.rank}"
        delay_dir = f"{self.args.job_dir}/delay"
        Path(delay_dir).mkdir(parents=True, exist_ok=True)
        fout = open(f"{delay_dir}/{name}.data", "w")
        for i in range(145):
            if(i == 100):
                size = 20 * (2**18)
            current_size += size
            size_in_mb = (current_size * 4)// 2**20     

            ##################################################################
            # 2. measure raw delays and memory to rule out profiler overhead #
            ##################################################################
            events_pre_all_reduce = [Event(enable_timing=True) for _ in range(niters)]
            events_post_all_reduce = [Event(enable_timing=True) for _ in range(niters)]
            for i in range(niters):
                loss = self.forward()
                loss.backward()
                data_tensor = torch.randn(current_size, dtype=torch.float32, device=self.local_rank)
                events_pre_all_reduce[i].record()
                dist.all_reduce(data_tensor)
                events_post_all_reduce[i].record()
                self.optimizer.step()


            # wait for all pending CUDA ops to finish
            torch.cuda.synchronize(device=self.local_rank)

            delays_all_reduce = [pre.elapsed_time(post) for pre, post in zip(events_pre_all_reduce, events_post_all_reduce)]

            # write results

            for delay in delays_all_reduce:
                fout.write(
                    f"{size_in_mb}, {delay:.4f}\n"
                )
           

            # wait for all peers to finish
            dist.barrier(device_ids=[self.local_rank])
        fout.close()
        self.teardown()
        return {
            "data_size" : size_in_mb,
        }

    def resnet_forward_ddp(self):
        out = self.ddp_model(*self.example_inputs)
        loss = self.loss_fn(out, self.example_outputs)
        return loss
    def resnet_forward(self):
        out = self.model(*self.example_inputs)
        loss = self.loss_fn(out, self.example_outputs)
        return loss
    def bert_forward_ddp(self):
        loss = self.ddp_model(**self.example_inputs).loss
        return loss
    def bert_forward(self):
        loss = self.model(**self.example_inputs).loss
        return loss

    def measure_ddp(self):
        niters = self.DEFAULT_MEASURE_ITERATIONS
        # TODO: using dummy data for now to rule out dataloader delays

        ######################################
        # 1. warming up CUDACachingAllocator #
        ######################################
        for _ in range(self.DEFAULT_MEASURE_ITERATIONS):
            loss = self.forward_ddp()
            loss.backward()
            self.ddp_optimizer.step()

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=self.local_rank)

        now = datetime.now()
        name = f"ddp_{str(type(self.model).__name__)}_{self.batch_size}"
        ##################################################################
        # 2. measure raw delays and memory to rule out profiler overhead #
        ##################################################################
        events_pre_fwd = [Event(enable_timing=True) for _ in range(niters)]
        events_pre_bwd = [Event(enable_timing=True) for _ in range(niters)]
        events_pre_opt = [Event(enable_timing=True) for _ in range(niters)]
        events_post_opt = [Event(enable_timing=True) for _ in range(niters)]
        for i in range(niters):
            events_pre_fwd[i].record()
            loss = self.forward_ddp()

            events_pre_bwd[i].record()
            loss.backward()

            events_pre_opt[i].record()
            self.ddp_optimizer.step()

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
        delay_dir = f"{self.args.job_dir}/delay"
        Path(delay_dir).mkdir(parents=True, exist_ok=True)
        fout = open(f"{delay_dir}/{name}.log", "w")
        fout.write(
            f"{mean_fwd:.2f}, {stdev_fwd:.2f}, "
            f"{mean_bwd:.2f}, {stdev_bwd:.2f}, "
            f"{mean_opt:.2f}, {stdev_opt:.2f}\n"
        )
        fout.close()


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
    os.environ["GPUS_PER_NODE"] = str(1)
    trainer = DDPTrainer(model_class=Model)

    trainer.measure_allreduce()

if __name__=="__main__":
    test()

