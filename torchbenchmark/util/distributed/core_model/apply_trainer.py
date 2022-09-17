import os
from torch.nn.parallel import DistributedDataParallel as DDP

def apply_trainer(model, trainer):
    assert trainer == "ddp", f"We only support the DDP trainer at the moment."
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        # If buffer broadcast is necessary, specific optimizations might be
        # necessary to optimize performance. Disable it by default.
        broadcast_buffers=False,
        # Set gradient as bucket view to avoid unnecessary copies
        gradient_as_bucket_view=True,
        # TODO: tune bucket_cap_mb
        static_graph=False,
    ) 
    return ddp_model
