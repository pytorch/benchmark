from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION
import torch.optim as optim
import torch
import torchvision.models as models

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Original train batch size: 512, out of memory on V100 GPU
    # Use hierarchical batching to scale down: 512 = batch_size (32) * epoch_size (16)
    # Source: https://github.com/forresti/SqueezeNet
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 16

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(model_name="squeezenet1_1", test=test, device=device, jit=jit,
                         batch_size=batch_size, weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1,
                         extra_args=extra_args)
        self.epoch_size = 16

    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        for _ in range(self.epoch_size):
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
            loss(pred, y).backward()
        optimizer.step()
