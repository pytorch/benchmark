import os
import re

from typing import List
from torchbenchmark.util.framework.timm.model_factory import TimmModel
from userbenchmark.dynamo.timm_models import BATCH_SIZE_DIVISORS, DYNAMO_BENCH_ROOT

TIMM_MODELS = dict()

class ModelSuite():
    def __init__(self, test, device, extra_args=[]):
        self.suite_name = "timm_models"
        self.model_ctor = TimmModel
        self.test = test
        self.device = device
        self.extra_args = extra_args
        self.is_training = self.test == "train"
        filename = os.path.join(DYNAMO_BENCH_ROOT, "timm_models_list.txt")
        with open(filename) as fh:
            lines = fh.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                model_name, batch_size = line.split(" ")
                TIMM_MODELS[model_name] = int(batch_size)

    def create_model(self, model_name: str) -> TimmModel:
        # Determine the batch size
        recorded_batch_size = TIMM_MODELS[model_name]
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

    def iter_model_names(self) -> List[str]:
        model_names = sorted(TIMM_MODELS.keys())
        for _index, model_name in enumerate(model_names):
            yield model_name
