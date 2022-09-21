import dataclasses
from typing import List

def cfg_to_str(cfg: dataclasses.dataclass) -> List[str]:
    def rewrite_option(opt: str) -> str:
        new_opt = opt.replace("_", "-")
        return f"--{new_opt}"
    out = []
    for fld in dataclasses.fields(cfg):
        new_option = rewrite_option(fld.name)
        val = getattr(cfg, fld.name)
        if isinstance(val, bool):
            if val:
                out.append(new_option)
        else:
            out.append(new_option)
            out.append(str(getattr(cfg, fld.name)))
    return out

# dummy config location:
# https://github.com/facebookresearch/FAMBench/blob/a0f12ca4fe8973f4cc65d18b51ce3aa94ceec0ac/benchmarks/run_dlrm_ootb_train.sh#L54
# config: A.1dev-embed32-fp32
@dataclasses.dataclass
class FAMBenchTrainConfig:
    mini_batch_size: int = 1024
    test_mini_batch_size: int = 1024
    test_num_workers: int = 0
    data_generation: str = "random"
    arch_mlp_bot:str = "2000-1500-1500-1500-192"
    arch_mlp_top:str = "4000-4000-4000-4000-4000-4000-4000-4000-4000-1"
    arch_sparse_feature_size:int = 192
    arch_embedding_size:str = "965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965-965"
    num_indices_per_lookup:int = 55
    num_indices_per_lookup_fixed:int = 1
    numpy_rand_seed:int = 727
    weighted_pooling: str = "learned"
    # torchbench: run 2 batches only (original 15)
    num_batches:int = 2
    # torchbench: these items in the original config are disabled
    # because they are handled by the framework
    # num_batches:int = 15
    # warmup_step = 5
    # use_gpu: bool = True
    # precache_ml_data: bool = True

# dummy config location:
# https://github.com/facebookresearch/FAMBench/blob/a0f12ca4fe8973f4cc65d18b51ce3aa94ceec0ac/benchmarks/run_dlrm_ootb_infer.sh#L54
# config: A.1dev-embed4-fp16
@dataclasses.dataclass
class FAMBenchEvalConfig:
    mini_batch_size:int = 1024
    test_mini_batch_size:int = 1024
    test_num_workers:int = 0
    data_generation:str = "random"
    arch_mlp_bot:str = "1414-1750-1750-1750-1750-1750-1750-96"
    arch_mlp_top:str = "1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1450-1"
    arch_sparse_feature_size:int = 96
    arch_embedding_size:str = "555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693-555693"
    num_indices_per_lookup:int = 8
    num_indices_per_lookup_fixed:int = 1
    numpy_rand_seed:int = 727
    weighted_pooling: str = "fixed"
    # original number of batches: 15
    num_batches:int = 15
    # torchbench: these items in the original config are disabled
    # because they either handled by the framework
    # or requires extra dependencies that we don't support yet (such as fbgemm and torch2trt_for_mlp)
    # disable warmup
    # warmup_step: int = 5
    # do not support quantize, torch2trt_for_mlp or fbgemm
    # quantize_emb_with_bit: int = 4
    # use_fbgemm_gpu: bool = True
    # use_gpu: bool = True
    # inference_only: bool = True
    # precache_ml_data: bool = True
    # use_torch2trt_for_mlp: bool = True
    # quantize_mlp_with_bit: int = 16
