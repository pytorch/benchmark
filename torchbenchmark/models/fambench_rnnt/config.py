import dataclasses
import os
from typing import List
from torchbenchmark import REPO_PATH
RNNT_EVAL_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "rnnt", "ootb", "inference")
DATASET_PATH = os.path.abspath(REPO_PATH, "torchbenchmark", ".data", "librespeech")

def cfg_to_str(cfg: dataclasses.dataclass) -> List[str]:
    def rewrite_option(opt: str) -> str:
        return f"--{opt}"
    out = []
    for fld in dataclasses.fields(cfg):
        new_option = rewrite_option(fld.name)
        out.append(new_option)
        out.append(str(getattr(cfg, fld.name)))
    return out

# Source of the train config
# https://github.com/facebookresearch/FAMBench/blob/main/benchmarks/rnnt/ootb/train/scripts/train.sh#L28
@dataclasses.dataclass
class FambenchRNNTTrainConfig:
    num_gpus: int = 1
    val_batch_size: int = 2
    learning_rate: int = 0.004
    batch_size: int = 1024

# Source of the eval config
# https://github.com/facebookresearch/FAMBench/blob/a0f12ca4fe8973f4cc65d18b51ce3aa94ceec0ac/benchmarks/rnnt/ootb/inference/run.sh
@dataclasses.dataclass
class FambenchRNNTEvalConfig:
    pytorch_config_toml: str = os.path.join(RNNT_EVAL_PATH, "pytorch", "configs", "rnnt.toml")
    manifest: str = os.path.join(DATASET_PATH, "dev-clean", "dev-clean-wav.json")
    scenario: str = "Offline"
    backend: str = "pytorch"
    batch_size: int = 1