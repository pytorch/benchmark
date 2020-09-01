import torch

@torch.jit.interface
class TensorToTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
