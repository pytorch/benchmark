import torch
import sys
import os
from torchbenchmark import REPO_PATH
import numpy as np
from typing import Tuple

# Import FAMBench model path
class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass
RNNT_TRAIN_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "rnnt", "ootb", "train")
RNNT_EVAL_PATH = os.path.join(REPO_PATH, "submodules", "FAMBench", "benchmarks", "rnnt", "ootb", "inference")

with add_path(RNNT_TRAIN_PATH):
    from common.data.dali.data_loader import DaliDataLoader

with add_path(RNNT_EVAL_PATH):
    from pytorch.model_separable_rnnt import RNNT

from .config import FambenchRNNTConfig, FambenchRNNTTrainConfig, FambenchRNNTEvalConfig
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH

class Model(BenchmarkModel):
    task = SPEECH.RECOGNITION
    RNNT_TRAIN_CONFIG = FambenchRNNTTrainConfig()
    RNNT_EVAL_CONFIG = FambenchRNNTEvalConfig()

    DEFAULT_TRAIN_BATCH_SIZE = RNNT_TRAIN_CONFIG.batch_size
    DEFAULT_EVAL_BATCH_SIZE = RNNT_EVAL_CONFIG.batch_size
    # run only 1 batch
    DEFAULT_NUM_BATCHES = 1

    def __init__(self):
        if self.test == "train":
            self._init_train()
        elif self.test == "eval":
            self._init_eval()

    # reference: FAMBench/benchmarks/rnnt/ootb/inference/pytorch_SUT.py
    def eval(self) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            feature, feature_length = self.auto_preprocessor.forward(*self.example_inputs)
            assert feature.ndim == 3
            assert feature_length.ndim == 1
            feature = feature.permute(2, 0, 1)
            _, _, transcript = self.greedy_decoder.forward(feature, feature_length)
        return (transcript, )

    def enable_jit(self):
        pass

    def _init_train(self):
        pass

    def _init_eval(self):
        model = RNNT(feature_config=featurizer_config,
                     rnnt=config['rnnt'],
                     num_classes=len(rnnt_vocab))
        model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path), strict=True)
        model.eval()
        self.greedy_decoder = ScriptGreedyDecoder(len(rnn_vocab) - 1, model)
        waveform = self.qsl[0]
        assert waveform.ndim == 1
        waveform_length = np.array(waveform.shape[0], dtype=np.int64)
        waveform = np.expand_dims(waveform, 0)
        waveform_length = np.expand_dims(waveform_length, 0)
        waveform = torch.from_numpy(waveform)
        waveform_length = torch.from_numpy(waveform_length)
        self.example_inputs = (waveform, waveform_length)