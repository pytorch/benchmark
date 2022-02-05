# Benchmark created from NVidia DeepRecommender github project:
#   https://github.com/NVIDIA/DeepRecommender
#   a32a8a5c23092c551616acf6fac5b32e1155d18b
# Test supports eval and train modes for cpu and cuda targets.
#
# Both nvtrain.py and nvinfer.py support all original command
# line parameters but tensorflow dependency for logging has
# been removed.

import torch
import torch.optim as optim
import torchvision.models as models

from torchbenchmark.models.attention_is_all_you_need_pytorch.train import train
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import RECOMMENDATION

import gc
from .nvtrain import DeepRecommenderTrainBenchmark
from .nvinfer import DeepRecommenderInferenceBenchmark

class Model(BenchmarkModel):

  task = RECOMMENDATION.RECOMMENDATION

  def __init__(self, test, device, train_bs=256, eval_bs=256, jit=False, extra_args=[]):
    super().__init__()
    self.device = device
    self.jit = jit
    self.train_bs = train_bs
    self.eval_bs = eval_bs
    self.test = test
    self.extra_args = extra_args
    self.not_implemented_reason = "Implemented"
    self.eval_mode = True #default to inference

    if jit:
      self.not_implemented_reason = "Jit Not Supported"

    elif self.device != "cpu" and self.device != "cuda":
      self.not_implemented_reason = "device type not supported"

    elif self.device == "cuda" and torch.cuda.is_available() == False:
      self.not_implemented_reason = "cuda not available on this device"

    else:
      if test == "train":
        self.train_obj = DeepRecommenderTrainBenchmark(device = self.device, jit = jit, batch_size=train_bs)
      elif test == "eval":
        self.infer_obj = DeepRecommenderInferenceBenchmark(device = self.device, jit = jit, batch_size=eval_bs)

  def get_module(self):
    if self.eval_mode:
       return self.infer_obj.rencoder, (self.infer_obj.toyinputs,)

    return self.train_obj.rencoder, (self.train_obj.toyinputs,)

  def set_eval(self):
    self.eval_mode = True

  def set_train(self):
    self.eval_mode = False

  def train(self, niter=1):
    self.check_implemented()

    for i in range(niter):
      self.train_obj.train(niter)

  def eval(self, niter=1):
    self.check_implemented()

    for i in range(niter):
      self.infer_obj.eval(niter)

  def check_implemented(self):
    if self.not_implemented_reason != "Implemented":
      raise NotImplementedError(self.not_implemented_reason)

  def timed_infer(self):
    self.check_implemented()
    self.infer_obj.TimedInferenceRun()

  def timed_train(self):
    self.check_implemented()
    self.train_obj.TimedTrainingRun()
