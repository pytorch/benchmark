import torch
import triton

import triton.language as tl


@triton.jit
def nop_kernel():
    pass


@triton.jit
def nop_with_args_kernel(
    t1,
    t2,
    t3,
    t4,
    t5,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
):
    pass


@torch.compile
def trivial_add_kernel(*args):
    return sum([torch.tensor(1.0, device="cuda"), *args])
