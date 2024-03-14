import os

from typing import Generator
from torchbenchmark.util.framework.timm.model_factory import Hugging
from userbenchmark.dynamo import DYNAMOBENCH_PATH
from .config import BATCH_SIZE_DIVISORS

# These models contain the models present in huggingface_models_list. It is a
# combination of models supported by HF Fx parser and some manually supplied
# models. For these models, we already know the largest batch size that can fit
# on A100 GPUs - 40 GB.
BATCH_SIZE_KNOWN_MODELS = dict()


# Get the list of models and their batch sizes
MODELS_FILENAME = os.path.join(DYNAMOBENCH_PATH, "huggingface_models_list.txt")
assert os.path.exists(MODELS_FILENAME)
with open(MODELS_FILENAME, "r") as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(",")
        batch_size = int(batch_size)
        BATCH_SIZE_KNOWN_MODELS[model_name] = batch_size
assert len(BATCH_SIZE_KNOWN_MODELS)

class ModelSuite():
    def __init__(self, test, device, extra_args=[]):
        self.suite_name = "huggingface_models"
        self.model_ctor = TimmModel
        self.test = test
        self.device = device
        self.extra_args = extra_args
        self.is_training = self.test == "train"

    def create_model(self, model_name: str) -> TimmModel:
        # Determine the batch size
        recorded_batch_size = BATCH_SIZE_KNOWN_MODELS[model_name]
        if model_name in BATCH_SIZE_DIVISORS:
            recorded_batch_size = max(
                int(recorded_batch_size / BATCH_SIZE_DIVISORS[model_name]), 1
            )
        batch_size = batch_size or recorded_batch_size
        model = self.model_ctor(model_name,
                                self.test,
                                self.device,
                                batch_size,
                                self.extra_args)
        return model

    def iter_model_names(self) -> Generator:
        model_names = sorted(BATCH_SIZE_KNOWN_MODELS.keys())
        for _index, model_name in enumerate(model_names):
            yield model_name
