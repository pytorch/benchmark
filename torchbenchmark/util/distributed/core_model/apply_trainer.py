import functools
from io import UnsupportedOperation
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def apply_trainer(model, trainer):
    local_rank = int(os.getenv("LOCAL_RANK", -1))

    if trainer == "ddp" or trainer == "ddp_no_static_graph":
        static_graph = (trainer == "ddp")
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            # If buffer broadcast is necessary, specific optimizations might be
            # necessary to optimize performance. Disable it by default.
            broadcast_buffers=False,
            # Set gradient as bucket view to avoid unnecessary copies
            gradient_as_bucket_view=True,
            # TODO: tune bucket_cap_mb
            static_graph=static_graph,
        )
        return ddp_model
    elif trainer == "fsdp":
        from transformers.models.bert.modeling_bert import BertLayer
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=(BertLayer,)),
            device_id = torch.cuda.current_device(),
            use_orig_params=True,
        )
        print(fsdp_model)
        return fsdp_model
    raise UnsupportedOperation(f"Only DDP, FSDP are currently supported, but tried to use {trainer}")
