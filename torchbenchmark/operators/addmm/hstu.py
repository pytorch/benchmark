import importlib

from typing import Tuple

import torch
import triton
from torchbenchmark import add_path, SUBMODULE_PATH

with add_path(str(SUBMODULE_PATH)):
    triton_addmm = importlib.import_module(
        "generative-recommenders.ops.triton.triton_addmm"
    )
    _addmm_fwd = triton_addmm._addmm_fwd


class _AddMmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        M, K = x.shape
        KB, N = w.shape
        assert K == KB, f"incompatible dimensions {K}, {KB}"

        is_y_1d = y.dim() == 1
        NY = y.shape[0] if is_y_1d else y.shape[1]
        assert N == NY, f"incompatible dimensions {N}, {NY}"

        # Allocate output
        z = torch.empty((M, N), device=x.device, dtype=x.dtype)
        if M == 0 or N == 0:
            ctx.save_for_backward(x, w)
            ctx.is_y_1d = False
            return z

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

        _addmm_fwd[grid](
            x,
            w,
            y,
            z,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0) if not is_y_1d else 0,
            y.stride(1) if not is_y_1d else y.stride(0),
            z.stride(0),
            z.stride(1),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            BROADCAST_Y=is_y_1d,
        )
        ctx.save_for_backward(x, w)
        ctx.is_y_1d = is_y_1d
        return z

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (x, w) = ctx.saved_tensors
        if ctx.is_y_1d:
            dy = torch.sum(dz, dim=0)
        else:
            dy = dz
        dw = torch.mm(x.t(), dz)
        dx = torch.mm(dz, w.t())

        return dx, dw, dy


@torch.fx.wrap
def triton_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    return _AddMmFunction.apply(mat1, mat2, input)
