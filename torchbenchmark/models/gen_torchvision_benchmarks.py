import os
from pathlib import Path

class_models = ['alexnet', 'vgg16', 'resnet18', 'resnet50', 'squeezenet1_1', 'densenet121', 'inception_v3', 'mobilenet_v2', 'shufflenet_v2_x1_0', 'resnext50_32x4d', 'mnasnet1_0']
for class_model in class_models:

    folder = Path(class_model)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    model = class_model
    if class_model == 'inception_v3': # Exception among imagenet models
        input_shape = (32, 3, 299, 299)
    input_shape = (32, 3, 224, 224)
    example_inputs = f'(torch.randn({input_shape}),)'
    eval_inputs = 'example_inputs[0][0].unsqueeze(0)'
    init_program = f"""
import torch
import torch.optim as optim
import torchvision.models as models

class Model:
    def __init__(self, device="cpu", jit=False):
        self.device = device
        self.jit = jit
        self.model = models.{model}()
        if self.jit:
            self.model = torch.jit.script(self.model)
        self.example_inputs = {example_inputs}

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=3):
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
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