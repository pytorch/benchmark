import torch
from core.block import GEMMBlock
from core.axis import TorchAxis
from core.utils.logging import get_logger

import torch

logger = get_logger(__name__)


class TorchGEMM(GEMMBlock):
    """
    PyTorch GEMM (General Matrix Multiply) compute block.
    Performs matrix multiplication: C = A @ B
    
    Args:
        axis: Axis object for process group management
        matrix_a: Input matrix A (M x K)
        matrix_b: Input matrix B (K x N)
        matrix_c: Output matrix C (M x N)
        name: Optional name for the block
        transpose_a: Whether to transpose matrix A before multiplication
        transpose_b: Whether to transpose matrix B before multiplication
    """
    
    registry_name = "torch_gemm"
    
    def __init__(self, axis: "TorchAxis", mat_a: torch.Tensor, mat_b: torch.Tensor, name: str = ""):
        
        super().__init__(axis=axis, mat_a=mat_a, mat_b=mat_b, name=name)
        
        # Validate that matrix dimensions are compatible for multiplication
        M, K1 = self.mat_a.shape
        K2, N = self.mat_b.shape

        if K1 != K2:
            raise ValueError(f"Incompatible matrix dimensions: A={self.mat_a.shape}, B={self.mat_b.shape}. "
                           f"Inner dimensions must match (K={K1} vs K={K2})")
        
        # Ensure all tensors are on the same device
        if not (self.mat_a.device == self.mat_b.device):
            raise ValueError(f"All tensors must be on the same device. "
                           f"Got: A={self.mat_a.device}, B={self.mat_b.device}")
        
        # Ensure all tensors are CUDA tensors
        if not (self.mat_a.is_cuda and self.mat_b.is_cuda):
            raise ValueError("All tensors must be CUDA tensors for GPU computation")
    
    def run(self):
        # Use torch.matmul for GEMM
        with self.axis.use_axis():
            torch.matmul(self.mat_a, self.mat_b)
        
