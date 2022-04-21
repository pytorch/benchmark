import torch
import sys
import os
import toml
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
    pass

with add_path(RNNT_EVAL_PATH):
    from QSL import AudioQSL, AudioQSLInMemory
    from helpers import add_blank_label
    from preprocessing import AudioPreprocessing
    from model_separable_rnnt import RNNT

from .config import FambenchRNNTTrainConfig, FambenchRNNTEvalConfig, cfg_to_str
from .decoders import ScriptGreedyDecoder
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH
from .args import get_eval_args

class Model(BenchmarkModel):
    task = SPEECH.RECOGNITION
    RNNT_TRAIN_CONFIG = FambenchRNNTTrainConfig()
    RNNT_EVAL_CONFIG = FambenchRNNTEvalConfig()

    # This model doesn't allow customize batch size
    DEFAULT_TRAIN_BATCH_SIZE = RNNT_TRAIN_CONFIG.batch_size
    DEFAULT_EVAL_BATCH_SIZE = RNNT_EVAL_CONFIG.batch_size
    ALLOW_CUSTOMIZE_BSIZE = False
    # run only 1 batch
    DEFAULT_NUM_BATCHES = 1

    def __init__(self):
        if self.test == "train":
            self._init_train()
        elif self.test == "eval":
            self._init_eval()

    # reference: FAMBench/benchmarks/rnnt/ootb/inference/pytorch_SUT.py
    def eval(self) -> Tuple[torch.Tensor]:
        for query_sample in self.query_samples:
            waveform = self.qsl[query_sample.index]
            assert waveform.ndim == 1
            waveform_length = np.array(waveform.shape[0], dtype=np.int64)
            waveform = np.expand_dims(waveform, 0)
            waveform_length = np.expand_dims(waveform_length, 0)
            with torch.no_grad():
                waveform = torch.from_numpy(waveform)
                waveform_length = torch.from_numpy(waveform_length)
                feature, feature_length = self.audio_preprocessor.forward((waveform, waveform_length))
                assert feature.ndim == 3
                assert feature_length.ndim == 1
                feature = feature.permute(2, 0, 1)

                _, _, transcript = self.greedy_decoder.forward(feature, feature_length)

    def enable_jit(self):
        if self.test == "eval":
            self.audio_preprocessor = torch.jit.script(self.audio_preprocessor)
            self.audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.audio_preprocessor._c))
            self.innner_model.encoder = torch.jit.script(self.inner_model.encoder)
            self.inner_model.encoder = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.inner_model.encoder._c))
            self.inner_model.prediction = torch.jit.script(self.inner_model.prediction)
            self.inner_model.prediction = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.inner_model.prediction._c))
            self.inner_model.joint = torch.jit.script(self.inner_model.joint)
            self.inner_model.joint = torch.jit._recursive.wrap_cpp_module(
                torch._C._freeze_module(self.inner_model.joint._c))
            self.inner_model = torch.jit.script(self.inner_model)
            self.greedy_decoder.set_model(self.inner_model)

    def _init_train(self):
        pass

    def _init_eval(self):
        args = cfg_to_str(self.RNNT_EVAL_CONFIG)
        args = get_eval_args(args) 
        assert args.backend == "pytorch", f"Unknown backend: {args.backend}"
        config = toml.load(args.config_toml)

        dataset_vocab = config['labels']['labels']
        rnnt_vocab = add_blank_label(dataset_vocab)
        featurizer_config = config['input_eval']

        self.qsl = AudioQSLInMemory(args.dataset_dir,
                                    args.manifest_filepath,
                                    dataset_vocab,
                                    featurizer_config["sample_rate"],
                                    args.perf_count)
        self.audio_preprocessor = AudioPreprocessing(**featurizer_config)
        self.audio_preprocessor.eval()

        model = RNNT(
            feature_config=featurizer_config,
            rnnt=config['rnnt'],
            num_classes=len(rnnt_vocab)
        )
        # torchbench: we don't support load checkpoint state dict
        # model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path),
        #                       strict=True)
        model.eval()

        self.inner_model = model
        self.greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, self.inner_model)