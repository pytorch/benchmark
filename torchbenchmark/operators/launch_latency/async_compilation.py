from torch._inductor.codecache import AsyncCompile


async_compile = AsyncCompile()

inductor_nop = async_compile.triton(
    "inductor_nop",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_heuristics

@triton_heuristics.pointwise(
    size_hints=[1],
    triton_meta={'signature': {0: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(), equal_to_1=())]},
)
@triton.jit
def inductor_nop(x):
    pass
""",
    device_str="cuda",
)


inductor_nop_args = async_compile.triton(
    "inductor_nop_args",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_heuristics

@triton_heuristics.pointwise(
    size_hints=[1],
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(5, 6, 7, 8, 9, 10, 11, 12, 13))]},
)
@triton.jit
def inductor_nop_args(t1, t2, t3, t4, t5, i1, i2, i3, i4, i5, i6, i7, i8, i9):
    pass
""",
    device_str="cuda",
)
