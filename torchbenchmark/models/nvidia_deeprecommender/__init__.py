import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import RECOMMENDATION

import gc
from .nvtrain import DeepRecommenderTrainBenchmark
from .nvinfer import DeepRecommenderInferenceBenchmark

class Model(BenchmarkModel):
  
  task = RECOMMENDATION.RECOMMENDATION
  
  def __init__(self, device="cpu", jit=False):
    super().__init__()
    self.devicename = device
    self.notimplementedreason = "Implemented"
    self.train_obj = DeepRecommenderTrainBenchmark(device = device, jit = jit)
    self.infer_obj = DeepRecommenderInferenceBenchmark(device = device, jit = jit)
    if jit:
      self.notimplementedreason = "Jit Not Supported"
    
    elif self.devicename != "cpu" and self.devicename != "cuda":
      self.notimplementedreason = "device type not supported"

    self.evalMode = True

  def get_module(self):
    if self.evalMode:
        return lambda x: self.eval(), [0]

    return lambda x: self.train(), [0]

  def set_eval(self):
    self.evalMode = True
    return
    
  def set_train(self):
    self.evalMode = False
    return

  def train(self, niter=1):
    self.checkimplemented()

    for i in range(niter):
      self.train_obj.train(self.train_obj.args.num_epochs)
  
    self.cleanup()

  def eval(self, niter=1):
    self.checkimplemented()

    for i in range(niter):
      self.infer_obj.eval(niter)
    
    self.cleanup()

  def checkimplemented(self):
    if self.notimplementedreason != "Implemented":
      raise NotImplementedError(self.notimplementedreason)

  def cleanup(self):
    self.train = 0
    self.infer = 0
    gc.collect()
    if self.devicename == "cuda":
      torch.cuda.empty_cache()

  def timedInfer(self):
    self.checkimplemented()
    self.infer_obj.TimedInferenceRun()

  def timedTrain(self):
    self.checkimplemented()
    self.train_obj.TimedTrainingRun()

def main():
  cudaBenchMark = DeepRecommenderBenchmark(device = 'cuda', jit = False)
  cudaBenchMark.timedTrain()
  cudaBenchMark.timedInfer()

  cpuBenchMark = DeepRecommenderBenchmark(device = 'cpu', jit = False)
  cpuBenchMark.timedTrain()
  cpuBenchMark.timedInfer()

if __name__ == '__main__':
  main()
