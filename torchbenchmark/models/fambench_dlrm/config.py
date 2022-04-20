import dataclasses
from typing import List

def cfg_to_str(cfg: dataclasses.dataclass) -> List[str]:
    def rewrite_option(opt: str) -> str:
        new_opt = opt.replace("_", "-")
        return f"--{new_opt}"
    out = []
    for fld in dataclasses.fields(cfg):
        new_option = rewrite_option(fld.name)
        out.append(new_option)
        out.append(str(getattr(cfg, fld.name)))
    return out

# Original config location:
# https://github.com/facebookresearch/FAMBench/blob/a0f12ca4fe8973f4cc65d18b51ce3aa94ceec0ac/benchmarks/run_dlrm_ootb_train.sh#L54
@dataclasses.dataclass
class FAMBenchTrainConfig:
    mini_batch_size: int = 64
    test_mini_batch_size: int = 64
    test_num_workers: int = 0
    data_generation: str = "random"
    arch_mlp_bot:str = "512-512-64"
    arch_mlp_top:str = "1024-1024-1024-1"
    arch_sparse_feature_size:int = 64
    arch_embedding_size:str = "1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
    num_indices_per_lookup:int = 100
    numpy_rand_seed:int = 727
    # not used, overridden by Model.DEFAULT_NUM_BATCHES
    num_batches:int = 200

# Original config location:
# https://github.com/facebookresearch/FAMBench/blob/a0f12ca4fe8973f4cc65d18b51ce3aa94ceec0ac/benchmarks/run_dlrm_ootb_infer.sh#L54
@dataclasses.dataclass
class FAMBenchEvalConfig:
    mini_batch_size:int = 64
    test_mini_batch_size:int = 64
    test_num_workers:int = 0
    data_generation:str = "random"
    arch_mlp_bot:str = "512-512-64"
    arch_mlp_top:str = "1024-1024-1024-1"
    arch_sparse_feature_size:int = 64
    arch_embedding_size:str = "1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
    num_indices_per_lookup:int = 100
    numpy_rand_seed:int = 727
    # not used, overridden by Model.DEFAULT_NUM_BATCHES
    num_batches:int = 200