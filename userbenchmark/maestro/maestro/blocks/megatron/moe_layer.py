from typing import Optional
import torch

from core.block import MegatronModuleBlock
from core.axis import TorchAxis
from core.utils.logging import get_logger


logger = get_logger(__name__)

# Megatron is optional — its import chain can fail (missing nvidia_resiliency_ext,
# version mismatches, etc.). Defer the failure until something actually tries to
# instantiate the block, instead of breaking the whole package at import time.
try:
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core import parallel_state
    MEGATRON_AVAILABLE = True
except Exception as _e:
    MEGATRON_AVAILABLE = False
    logger.warning(
        f"Megatron failed to initialize, megatron blocks will be unavailable. "
        f"Original error: {type(_e).__name__}: {_e}"
    )

# TODO - move this somewhere else when we have more visibility on how to use megatron blocks.
if MEGATRON_AVAILABLE:
    from core.utils.distributed import nccl_torch_dist_utils
    if nccl_torch_dist_utils is not None:
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(1,1)


class MegatronMoELayerBlock(MegatronModuleBlock):
    """
    Careful: MegatronMoELayerBlock is a blocking block, it will block the other streams, put it at the end of the pattern if possible, or be aware that the blocks that come after won't overlap
    """
    IS_BLOCKING_CPU_BLOCK = True

    registry_name = "megatron_moe_layer"
    api_params_schema = {
        "axis": {},
        "name": {},
        "isl": {},
        "batch_size": {"default": 2},
        'hidden_size': {},
        "topk": {'default': 2},
        'dispatch_type': {'default': 'allgather'}, # can be allgather, alltoall and flex
        'num_experts': {'default': -1}, # If -1, there is one expert per rank
        "operation": {'default': 'forward'}
    }
    
    def __init__(self, axis: TorchAxis, input_tensor: torch.Tensor, hidden_size: int, topk: int = 2, dispatch_type: str = "allgather", num_experts: int = -1, name: str = "", operation: str = "forward"):
        assert MEGATRON_AVAILABLE, "Megatron failed to initialize, cannot use megatron block"
        super().__init__(axis=axis, input_tensor=input_tensor, name=name)

        self.operation = operation

        pg_collection = get_default_pg_collection()
        pg_collection.ep = axis.get_process_group()
        if num_experts == -1:
            num_experts = axis.team_size()

        transformer_config = TransformerConfig(
            num_moe_experts=num_experts,
            moe_token_dispatcher_type=dispatch_type,
            moe_router_topk=topk,
            moe_grouped_gemm=True,
            moe_ffn_hidden_size=hidden_size,
            add_bias_linear=False,
            num_attention_heads=1,
            hidden_size=hidden_size,
            num_layers=1,
            params_dtype=self.input_tensor.dtype,
            pipeline_dtype=self.input_tensor.dtype,
        )

        backend = TESpecProvider()

        expert_module, expert_submodule = backend.grouped_mlp_modules(False, False)
        expert_submodule.activation_func = backend.activation_func()

        experts = ModuleSpec(module=expert_module, submodules=expert_submodule)
        moe_submodules = MoESubmodules(experts=experts)

        self.module = MoELayer(
            transformer_config, 
            moe_submodules,
            pg_collection=pg_collection
        )
    
    def _run(self):
        with self.axis.use_axis():
            if self.operation == "forward":
                self.module.forward(self.input_tensor)
            elif self.operation == "dispatch":
                raise NotImplementedError("Dispatch operation is not implemented")
            else:
                raise ValueError(f"Invalid operation: {self.operation}")

    @classmethod
    def get_tensor_size(cls, api_params: dict) -> int:
        api_params = cls._enrich_api_params_schema(api_params)
        return api_params["batch_size"] * api_params["isl"] * api_params["hidden_size"]
    
    @classmethod
    def from_api_params(cls, axis: TorchAxis, name: str, api_params: dict, cached_tensor: Optional[torch.Tensor] = None):
        api_params = cls._enrich_api_params_schema(api_params)
        if cached_tensor is None:
            cached_tensor = torch.zeros(cls.get_tensor_size(api_params), device="cuda")
        input_tensor = cached_tensor[:cls.get_tensor_size(api_params)]
        # View vector as (batch_size, isl, hidden_size)
        input_tensor = input_tensor.view(api_params["batch_size"], api_params["isl"], api_params["hidden_size"])
        return cls(axis=axis, 
            input_tensor=input_tensor,
            hidden_size=api_params["hidden_size"],
            topk=api_params["topk"],
            dispatch_type=api_params["dispatch_type"],
            num_experts=api_params["num_experts"],
            name=name,
            operation=api_params["operation"]
            )
        

