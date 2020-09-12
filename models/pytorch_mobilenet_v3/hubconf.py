import torch
import torch.optim as optim
from mobilenetv3 import MobileNetV3


class Model:
    def __init__(self, device="cpu", jit=False):
        """ Required """
        self.device = device
        self.jit = jit
        self.model = MobileNetV3().to(device)
        if self.jit:
            self.model = torch.jit.script(self.model)
        input_size = (1, 3, 224, 224)
        self.example_inputs = (torch.randn(input_size, device=device),)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=3):
        optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(
                pred.shape[0], dtype=torch.long, device=self.device
            ).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def eval(self, niter=1):
        model, example_inputs = self.get_module()
        for i in range(niter):
            model(*example_inputs)


if __name__ == "__main__":
    m = Model(device="cuda", jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train(niter=1)
    m.eval(niter=1)
