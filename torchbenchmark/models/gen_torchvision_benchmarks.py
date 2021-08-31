import os
from pathlib import Path

class_models = ['alexnet', 'vgg16', 'resnet18', 'resnet50', 'squeezenet1_1', 'densenet121', 'mobilenet_v2', 'mobilenet_v3_large', 'shufflenet_v2_x1_0', 'resnext50_32x4d', 'mnasnet1_0']
for class_model in class_models:
    folder = Path(class_model)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    input_shape = (32, 3, 224, 224)
    example_inputs = f'(torch.randn({input_shape}).to(self.device),)'
    eval_inputs = 'example_inputs[0][0].unsqueeze(0)'
    init_program = f"""
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
    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = models.{class_model}().to(self.device)
        self.eval_model = models.{class_model}().to(self.device)
        self.example_inputs = {example_inputs}

        if self.jit:
            self.model = torch.jit.script(self.model, example_inputs=[self.example_inputs, ])
            self.eval_model = torch.jit.script(self.eval_model)
            # model needs to in `eval`
            # in order to be optimized for inference
            self.eval_model.eval()
            self.eval_model = torch.jit.optimize_for_inference(self.eval_model)


    def get_module(self):
        return self.model, self.example_inputs

    # vision models have another model
    # instance for inference that has
    # already been optimized for inference
    def set_eval(self):
        pass

    def train(self, niter=3):
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def eval(self, niter=1):
        model = self.eval_model
        example_inputs = self.example_inputs
        example_inputs = {eval_inputs}
        for i in range(niter):
            model(example_inputs)


if __name__ == "__main__":
    m = Model(device="cuda", jit=True)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train(niter=1)
    m.eval(niter=1)
"""
    with open(folder / '__init__.py', 'w') as f:
        f.write(init_program)
    with open(folder / 'install.py', 'w') as f:
        pass
