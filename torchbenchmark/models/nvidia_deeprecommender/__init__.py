# Benchmark created from NVidia DeepRecommender github project:
#   https://github.com/NVIDIA/DeepRecommender
#   a32a8a5c23092c551616acf6fac5b32e1155d18b
# Test supports eval and train modes for cpu and cuda targets.
#
# Both nvtrain.py and nvinfer.py support all original command
# line parameters but tensorflow dependency for logging has
# been removed.

import torch

from torchbenchmark.models.attention_is_all_you_need_pytorch.train import train
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import RECOMMENDATION
from typing import Tuple

import gc
from .nvtrain import DeepRecommenderTrainBenchmark
from .nvinfer import DeepRecommenderInferenceBenchmark

class Model(BenchmarkModel):

  task = RECOMMENDATION.RECOMMENDATION
  DEFAULT_TRAIN_BSIZE = 256
  DEFAULT_EVAL_BSIZE = 256

  def __init__(self, test, device, batch_size=None, jit=False, extra_args=[]):
    if device == "cpu" and batch_size is None:
      batch_size = 1
    super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
    self.eval_mode = True if self.test == "eval" else False

    if test == "train":
      self.model = DeepRecommenderTrainBenchmark(device = self.device, jit = jit, batch_size=self.batch_size)
    elif test == "eval":
      self.model = DeepRecommenderInferenceBenchmark(device = self.device, jit = jit, batch_size=self.batch_size)

  def jit_callback(self):
    assert self.jit, "Calling JIT callback without specifying the JIT option."
    self.model.rencoder = torch.jit.trace(self.model.rencoder, (self.model.toyinputs, ))

  def get_module(self):
    if self.eval_mode:
       return self.model.rencoder, (self.model.toyinputs,)
    return self.model.rencoder, (self.model.toyinputs,)

  def set_module(self, new_model):
    self.model.rencoder = new_model

  def set_eval(self):
    self.eval_mode = True

  def set_train(self):
    self.eval_mode = False

  def train(self):
    self.model.train()

  def eval(self) -> Tuple[torch.Tensor]:
    out = self.model.eval()
    return (out, )

  def timed_infer(self):
    self.model.TimedInferenceRun()

  def timed_train(self):
    self.model.TimedTrainingRun()
