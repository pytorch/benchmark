from dataclasses import dataclass

# Original config location:
# https://github.com/facebookresearch/FAMBench/blob/a0f12ca4fe8973f4cc65d18b51ce3aa94ceec0ac/benchmarks/run_dlrm_ootb_train.sh#L54
@dataclass
class FAMBenchTrainConfig:
    mini_batch_size = 64
    test_mini_batch_size = 64
    test_num_workers = 0
    data_generation = "random"
    arch_mlp_bot = "512-512-64"
    arch_mlp_top = "1024-1024-1024-1"
    arch_sparse_feature_size = 64
    arch_embedding_size = "1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
    num_indices_per_lookup = 100
    numpy_rand_seed = 727
    # not used, overridden by Model.DEFAULT_NUM_BATCHES
    num_batches = 200

# Original config location:
# https://github.com/facebookresearch/FAMBench/blob/a0f12ca4fe8973f4cc65d18b51ce3aa94ceec0ac/benchmarks/run_dlrm_ootb_infer.sh#L54
@dataclass
class FAMBenchEvalConfig:
    mini_batch_size = 64
    test_mini_batch_size = 64
    test_num_workers = 0
    data_generation = "random"
    arch_mlp_bot = "512-512-64"
    arch_mlp_top = "1024-1024-1024-1"
    arch_sparse_feature_size = 64
    arch_embedding_size = "1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
    num_indices_per_lookup = 100
    numpy_rand_seed = 727
    # not used, overridden by Model.DEFAULT_NUM_BATCHES
    num_batches = 200