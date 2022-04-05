import os
import sys
import torch
import subprocess
from pathlib import Path
from dataclasses import dataclass

from torchbenchmark.util.e2emodel import E2EBenchmarkModel

from typing import Optional, List

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
FAMBENCH_ROOT = CURRENT_DIR.parent.parent.parent.joinpath("submodules", "FAMBench")

def _create_data_dir(data_dir: str):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def _get_fambench_test_root(name: str):
    xlmr_ootb_root = FAMBENCH_ROOT.joinpath("benchmarks")
    assert xlmr_ootb_root.exists(), f"Can't find FAMBench source at {xlmr_ootb_root.absolute()}," \
                                    "please check out the submodules."
    return xlmr_ootb_root

@dataclass
class FAMBenchXLMREvalConfig:
    """
    Original config reference:
    https://github.com/facebookresearch/FAMBench/blob/main/benchmarks/run_xlmr_ootb.sh
    """
    config_name = "default-config"
    nbatches = 10
    batchsize = 16
    seqlength = 16
    vocabsize = 250000
    warmupbatches = 1
    log_dir = os.path.join(CURRENT_DIR, ".data", "logs")
    config_flags=["--inference-only", f"--num-batches={nbatches}", f"--batch-size={batchsize}", \
                 f"--sequence-length={seqlength}", f"--vocab-size={vocabsize}", \
                 f"--famconfig={config_name}", "--half-model", f"--warmup-batches={warmupbatches}", \
                 f"--logdir={log_dir}"]

class Model(E2EBenchmarkModel):
    DEFAULT_EVAL_BSIZE = FAMBenchXLMREvalConfig.batchsize
    def __init__(self, test: str, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        super().__init__(test=test, batch_size=batch_size, extra_args=extra_args)
        if not torch.cuda.is_available():
            raise NotImplementedError("FAMBench only support running on Nvidia GPU.")
        self.device = "cuda"
        self.device_num = torch.cuda.device_count()
        self.name = "xlmr"
        self.implementation = "ootb"
        self.code_root = _get_fambench_test_root(self.name)
        if test == "eval":
            self.config = FAMBenchXLMREvalConfig()
            self.config.batchsize = self.batch_size
            self.num_examples = self.config.nbatches * self.batch_size
            _create_data_dir(self.config.log_dir)

    def train(self):
        raise NotImplementedError("FAMBench XLMR train is not implemented yet.")

    def eval(self):
        prog_args = [sys.executable, f"{self.name}/{self.implementation}/{self.name}.py"]
        prog_args.extend(self.config.config_flags)
        subprocess.check_call(prog_args, cwd=self.code_root)