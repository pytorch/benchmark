from io import UnsupportedOperation
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def apply_trainer(model, trainer):
    local_rank = int(os.getenv("LOCAL_RANK", -1))

    if trainer == "ddp":
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            # If buffer broadcast is necessary, specific optimizations might be
            # necessary to optimize performance. Disable it by default.
            broadcast_buffers=False,
            # Set gradient as bucket view to avoid unnecessary copies
            gradient_as_bucket_view=True,
            # TODO: tune bucket_cap_mb
            bucket_cap_mb=4,
            static_graph=False,
        )
        return ddp_model
    elif trainer == "fsdp":
        fsdp_model = FSDP(
            model,
            device_id = torch.cuda.current_device()
        )
        return fsdp_model
    raise UnsupportedOperation(f"Only DDP, FSDP are currently supported, but tried to use {trainer}")
