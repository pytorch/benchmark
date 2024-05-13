"""
Based on https://github.com/pytorch/pytorch/issues/121661
"""

import torch

import triton
import triton.language as tl
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 2048,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 64,
                "RBLOCK": 8,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 64,
                "RBLOCK": 4,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 8,
                "RBLOCK": 512,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 8,
                "RBLOCK": 256,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 64,
                "RBLOCK": 64,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_mv_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = xindex
    # x0 // rnumel should have the same value of either 0 or 1
    tmp0 = tl.load(in_ptr0 + ((x0 // rnumel)), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex # size (1, RBLOCK)
        tmp7 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0 + 8
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0) # size (XBLOCK, 1)
        # in_ptr1 has (B, S, S) shape, tmp3 is the 2nd dimension with stride of S * S.
        tmp4 = tl.load(in_ptr1 + (r1 + (rnumel*(x0 % rnumel)) + (rnumel*rnumel*tmp3)), None, eviction_policy='evict_first')
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 * tmp8 # (XBLOCK, RBLOCK) * (1, RBLOCK)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tmp12
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tmp11.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp13, None)


def triton_gemv_0(arg0_1, arg1_1, arg2_1):
    S, = arg2_1.shape
    assert_size_stride(arg0_1, (8, S, S), (S*S, S, 1))
    assert_size_stride(arg1_1, (2, ), (1, ))
    assert_size_stride(arg2_1, (S, ), (1, ))
    xnumel = 2*S
    rnumel = S
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # size will be double
        buf1 = empty_strided_cuda((2*S, ), (1, ), torch.bfloat16)

        grid = lambda META: (
            triton.cdiv(2*S, META["XBLOCK"]),
        )
        triton_red_fused_mv_0[grid](arg1_1, arg0_1, arg2_1, buf1, xnumel, rnumel)
    return (reinterpret_tensor(buf1, (2, S), (S, 1), 0), )

