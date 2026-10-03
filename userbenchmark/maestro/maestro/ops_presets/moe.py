import math 

from core.op_preset import OpPreset

class MOEOpPreset(OpPreset):

    name = "moe"

    params_schema = {
        "axis_name": {
            "required": True,
            "description": "Axis to use for the MOE operation",
        },
        "seq_len": {
            "required": True,
            "description": "Sequence length",
        },
        "k": {
            "description": "Number of experts per token (top-k)",
            "default": 2,
        },
        "d_model": {
            "required": True,
            "description": "Model hidden dimension",
        },
        "expert_dim": {
            "required": True,
            "description": "Expert FFN hidden dimension",
        },
        "batch_size": {
            "description": "Batch size",
            "default": 1,
        },
        "expert_capacity_factor": {
            "description": "Expert capacity factor",
            "default": 1.5,
        }
    }

    @classmethod
    def create_op(cls, params: dict, axes_config: dict) -> tuple[dict, list]:
        cls.validate_params(params)
        axis_name = params["axis_name"]

        axis = axes_config.get(axis_name)
        if axis is None:
            raise ValueError(f"{cls.__name__}'s axis {axis_name} was not defined in the pattern's axes")
        elif len(axis) == 0:
            raise ValueError(f"{cls.__name__}'s axis {axis_name} is empty")
        elif len(axis[0]) == 0:
            raise ValueError(f"{cls.__name__}'s axis {axis_name} has empty teams")

        num_experts = len(axis[0])

        num_tokens = params["batch_size"] * params["seq_len"]
        exp_capacity = math.floor(num_tokens / num_experts)

        ops = [
            # Dispatch
            {
                "block": "torch_alltoall",
                "axis": axis_name,
                "name": "moe_dispatch",
                "vector_size": num_tokens * params["d_model"]
            },
            # Gemm (technically there is more than one gemm (3gemms + hadamar product), but this is for the example)
            {
                "block": "torch_gemm",
                "axis": axis_name,
                "name": "moe_gemm",
                "mat_a_shape": [exp_capacity, params["expert_dim"]],
                "mat_b_shape": [params["expert_dim"], params["d_model"]],
            },
            # Combine
            {
                "block": "torch_alltoall",
                "axis": axis_name,
                "name": "moe_combine",
                "vector_size": num_tokens * params["d_model"],
            },
        ]

        return {}, ops 