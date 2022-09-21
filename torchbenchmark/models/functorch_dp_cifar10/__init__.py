import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from functorch import make_functional_with_buffers, vmap, grad
from typing import Tuple

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER


def compute_norms(sample_grads):
    batch_size = sample_grads[0].shape[0]
    norms = [sample_grad.view(batch_size, -1).norm(2, dim=-1) for sample_grad in sample_grads]
    norms = torch.stack(norms, dim=0).norm(2, dim=0)
    return norms, batch_size


def clip_and_accumulate_and_add_noise(model, max_per_sample_grad_norm=1.0, noise_multiplier=1.0):
    sample_grads = tuple(param.grad_sample for param in model.parameters())

    # step 0: compute the norms
    sample_norms, batch_size = compute_norms(sample_grads)

    # step 1: compute clipping factors
    clip_factor = max_per_sample_grad_norm / (sample_norms + 1e-6)
    clip_factor = clip_factor.clamp(max=1.0)

    # step 2: clip
    grads = tuple(torch.einsum('i,i...', clip_factor, sample_grad)
                  for sample_grad in sample_grads)

    # step 3: add gaussian noise
    stddev = max_per_sample_grad_norm * noise_multiplier
    noises = tuple(torch.normal(0, stddev, grad_param.shape, device=grad_param.device)
                   for grad_param in grads)
    grads = tuple(noise + grad_param for noise, grad_param in zip(noises, grads))

    # step 4: assign the new grads, delete the sample grads
    for param, param_grad in zip(model.parameters(), grads):
        param.grad = param_grad / batch_size
        del param.grad_sample


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    DEFAULT_TRAIN_BSIZE = 64
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        # Generate a resnet18, patch the BatchNorm layers to be GroupNorm
        self.model = models.__dict__['resnet18'](
            # min(32, c) is a reasonable default value, see the following:
            # https://github.com/pytorch/opacus/blob/6a3e9bd99dca314596bc0313bb4241eac7c9a5d0/opacus/validators/batch_norm.py#L84-L86
            pretrained=False, norm_layer=(lambda c: nn.GroupNorm(min(c, 32), c))
        )
        self.model = self.model.to(device)

        # Cifar10 images are 32x32 and have 10 classes
        self.example_inputs = (
            torch.randn((self.batch_size, 3, 32, 32), device=self.device),
        )
        self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        model = self.model
        model.train()
        fnet, params, buffers = make_functional_with_buffers(self.model)

        (images, ) = self.example_inputs
        targets = self.example_target

        def compute_loss(params, buffers, image, target):
            image = image.unsqueeze(0)
            target = target.unsqueeze(0)
            pred = fnet(params, buffers, image)
            loss = self.criterion(pred, target)
            return loss

        sample_grads = vmap(grad(compute_loss), (None, None, 0, 0))(params, buffers, images, targets)

        for grad_sample, weight in zip(sample_grads, model.parameters()):
            weight.grad_sample = grad_sample.detach()

        clip_and_accumulate_and_add_noise(model)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval(self) -> Tuple[torch.Tensor]:
        model = self.model
        (images, ) = self.example_inputs
        model.eval()
        targets = self.example_target
        with torch.no_grad():
            out = model(images)
        return (out, )
