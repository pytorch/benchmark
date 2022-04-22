import dataclasses
import os
from typing import List
from torchbenchmark import REPO_PATH

RNNT_TRAIN_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "rnnt", "ootb", "train")
RNNT_EVAL_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "rnnt", "ootb", "inference")
DATASET_PATH = os.path.abspath(REPO_PATH, "torchbenchmark", ".data", "librespeech")

def cfg_to_str(cfg: dataclasses.dataclass) -> List[str]:
    def rewrite_option(opt: str) -> str:
        if opt:
            return f"--{opt}"
        return None
    out = []
    for fld in dataclasses.fields(cfg):
        new_option = rewrite_option(fld.name)
        if new_option:
            out.append(new_option)
        out.append(str(getattr(cfg, fld.name)))
    return out

# Source of the train config
# https://github.com/facebookresearch/FAMBench/blob/main/benchmarks/rnnt/ootb/train/scripts/train.sh#L28
@dataclasses.dataclass
class FambenchRNNTTrainConfig:
    dataset_dir: str = DATASET_PATH
    val_manifests: str = os.path.join(DATASET_PATH, "librispeech-dev-clean-wav.json")
    train_manifests: str = os.path.join(DATASET_PATH, "librispeech-train-clean-100-wav.json")
    model_config: str = os.path.join(RNNT_TRAIN_PATH, "configs", "baseline_v3-1023sp.yaml")
    # don't use output dir
    output_dir: str = ""
    lr: str = "0.004"
    # global batch size, local batch size = global batch size / gpunum
    batch_size: str = "1024"
    val_batch_size: str = "2"
    min_lr: str = "1e-5"
    lr_exp_gamma: str = "0.935"
    # only run 1 epoch
    epochs: str = "1"
    # we do warmup in the outside loop
    warmup_epoch: str = "0"
    hold_epochs: str = "40"
    epochs_this_job: str = "0"
    ema: str = "0.999"
    seed: str = "1"
    weight_decay: str = "1e-3"
    log_frequency: str = "1"
    val_frequency: str = "1"
    grad_accumulation_steps: str = "64"
    dali_device: str = "cpu"
    beta1: str = "0.9"
    beta2: str = "0.999"
    amp: str = "false"
    cudnn_benchmark: str = "true"
    num_buckets: str = "6"
    target: str = ""
    clip_norm: str = "1"
    prediction_frequency: str = "1000"
    keep_milestones: str = ""
    start_clip: str = ""
    hidden_hidden_bias_scale: str = ""
    weights_init_scale: str = "0.5"
    max_symbol_per_sample: str = "300"

# Source of the eval config
# https://github.com/facebookresearch/FAMBench/blob/a0f12ca4fe8973f4cc65d18b51ce3aa94ceec0ac/benchmarks/rnnt/ootb/inference/run.sh
@dataclasses.dataclass
class FambenchRNNTEvalConfig:
    pytorch_config_toml: str = os.path.join(RNNT_EVAL_PATH, "pytorch", "configs", "rnnt.toml")
    manifest: str = os.path.join(DATASET_PATH, "dev-clean", "dev-clean-wav.json")
    scenario: str = "Offline"
    backend: str = "pytorch"
    batch_size: int = 1