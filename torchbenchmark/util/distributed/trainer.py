import os
import time
import torch
import torch.distributed as dist


class Trainer():
    def __init__(self, args, model_class, mode="SPMD"):
        self.args = args
        self.model_class = model_class
        self.mode = mode

    def setup(self):
        if self.mode == "SPMD":
            local_rank = int(os.getenv("LOCAL_RANK", -1))
            self.local_rank = local_rank
            self.gpus_per_node = int(os.getenv("GPUS_PER_NODE"))
            self.network_type = str(os.getenv("NET_TYPE","efa"))
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
            torch.cuda.set_device(local_rank)

            world_size = int(os.getenv("WORLD_SIZE", -1))
            rank = int(os.getenv("RANK", -1))

            assert local_rank != -1 and world_size != -1 and rank != -1, (
                "Failed to retrieve SPMD configurations from environment "
                f"variables. local_rank={local_rank}, world_size={world_size}, "
                f"rank={rank}."

            )

            # TODO: hardcode NCCL for now, make this configurable if necessary
            dist.init_process_group("nccl", init_method=self.args.dist_url, rank=rank, world_size=world_size)
        else:
            raise ValueError(f"Unrecognized distributed training mode {self.mode}")

    def teardown(self):
        if self.mode == "SPMD":
            dist.destroy_process_group()
