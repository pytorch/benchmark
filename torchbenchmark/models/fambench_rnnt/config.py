import dataclasses


# Source of the train config
# https://github.com/facebookresearch/FAMBench/blob/main/benchmarks/rnnt/ootb/train/scripts/train.sh#L28
@dataclasses.dataclass
class FambenchRNNTTrainConfig:
    num_gpus = 1
    val_batch_size = 2
    learning_rate = 0.004
    batch_size = 1024

# Source of the eval config

@dataclasses.dataclass
class FambenchRNNTEvalConfig:
    batch_size = 1