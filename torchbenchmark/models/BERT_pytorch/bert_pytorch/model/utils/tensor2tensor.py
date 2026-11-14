import torch

# torch.jit.interface is not available in PyTorch builds without TorchScript.
_jit_interface = getattr(torch.jit, "interface", lambda cls: cls)


@_jit_interface
class TensorToTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
