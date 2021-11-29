
# Generated by gen_torchvision_benchmark.py
import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

#######################################################
#
#       DO NOT MODIFY THESE FILES DIRECTLY!!!
#       USE `gen_torchvision_benchmarks.py`
#
#######################################################
class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION
    optimized_for_inference = True
    # Original train batch size: 512, out of memory on V100 GPU
    # Use hierarchical batching to scale down: 512 = batch_size (32) * epoch_size (16)
    # Source: https://github.com/forresti/SqueezeNet
    def __init__(self, device=None, jit=False, train_bs=32, eval_bs=16):
        super().__init__()
        self.device = device
        self.jit = jit
        self.epoch_size = 16
        self.model = models.squeezenet1_1().to(self.device)
        self.eval_model = models.squeezenet1_1().to(self.device)
        self.example_inputs = (torch.randn((train_bs, 3, 224, 224)).to(self.device),)
        self.infer_example_inputs = (torch.randn((eval_bs, 3, 224, 224)).to(self.device),)

        if self.jit:
            if hasattr(torch.jit, '_script_pdt'):
                self.model = torch.jit._script_pdt(self.model, example_inputs=[self.example_inputs, ])
                self.eval_model = torch.jit._script_pdt(self.eval_model)
            else:
                self.model = torch.jit.script(self.model, example_inputs=[self.example_inputs, ])
                self.eval_model = torch.jit.script(self.eval_model)
            # model needs to in `eval`
            # in order to be optimized for inference
            self.eval_model.eval()
            self.eval_model = torch.jit.optimize_for_inference(self.eval_model)


    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=1):
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            for _ in range(self.epoch_size):
                optimizer.zero_grad()
                pred = self.model(*self.example_inputs)
                y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
                loss(pred, y).backward()
                optimizer.step()

    def eval(self, niter=1):
        model = self.eval_model
        example_inputs = self.infer_example_inputs
        for i in range(niter):
            model(*example_inputs)
