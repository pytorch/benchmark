"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch
import numpy as np

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# [64, 128], [3, 4, 7], [4, 8]
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [64, 128]\
    for s in  [3, 4, 7]\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                    K_desc_ptr, V_desc_ptr, Q, qvk_offset, stride_kn, stride_vn, stride_vk,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # Required TMA fences are added in _attn_fwd_tma prior to calling this function.
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl._experimental_descriptor_load(  # load in row major
            K_desc_ptr,
            [start_n.to(tl.int32) + (qvk_offset // stride_kn).to(tl.int32), 0],
            [BLOCK_N, HEAD_DIM],
            Q.dtype.element_ty,
        )
        k = tl.trans(k)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if fp8_v:
            v = tl._experimental_descriptor_load(  # load in row major
                V_desc_ptr,
                [(qvk_offset // stride_vn).to(tl.int32), start_n.to(tl.int32)],
                [HEAD_DIM, BLOCK_N],
                Q.dtype.element_ty,
            )
            v = tl.trans(v)
        else:
            v = tl._experimental_descriptor_load(  # load in row major
                V_desc_ptr,
                [(qvk_offset // stride_vk + start_n).to(tl.int32), 0],
                [BLOCK_N, HEAD_DIM],
                Q.dtype.element_ty,
            )
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


@triton.autotune(list(filter(keep, configs)), key=["N_CTX"])
@triton.jit
def _attn_fwd_tma(#Q, V, desc_k, desc_v, sm_scale, M, Out,  #
              Q, V, Out, desc_q, desc_k, desc_v, sm_scale, M, desc_o,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    # TODO(embg) remove TMA fence after __grid_constant__ lands
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_q], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_k], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_v], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_o], dtype=tl.int32, is_pure=False, pack=1)

    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    #Q_block_ptr = tl.make_block_ptr(
    #    base=Q + qvk_offset,
    #    shape=(N_CTX, HEAD_DIM),
    #    strides=(stride_qm, stride_qk),
    #    offsets=(start_m * BLOCK_M, 0),
    #    block_shape=(BLOCK_M, HEAD_DIM),
    #    order=(1, 0),
    #)
    #O_block_ptr = tl.make_block_ptr(
    #    base=Out + qvk_offset,
    #    shape=(N_CTX, HEAD_DIM),
    #    strides=(stride_om, stride_on),
    #    offsets=(start_m * BLOCK_M, 0),
    #    block_shape=(BLOCK_M, HEAD_DIM),
    #    order=(1, 0),
    #)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    #q = tl.load(Q_block_ptr)
    q = tl._experimental_descriptor_load(  # load in row major
         desc_q,
         [(qvk_offset // stride_qm + start_m * BLOCK_M).to(tl.int32), 0],
         [BLOCK_M, HEAD_DIM],
         Q.dtype.element_ty,
    )
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q, desc_k, desc_v, Q, qvk_offset, stride_kn, stride_vn,
                                        stride_vk,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q, desc_k, desc_v, Q, qvk_offset, stride_kn, stride_vn,
                                        stride_vk,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl._experimental_descriptor_store(desc_o, acc.to(Out.type.element_ty),
        [(qvk_offset // stride_om + start_m * BLOCK_M).to(tl.int32), 0])
    #tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.autotune(list(filter(keep, configs)), key=["N_CTX"])
@triton.jit
def _attn_fwd_persistent(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              NUM_SMS: tl.constexpr,
              STAGE: tl.constexpr  #
              ):
    num_pid_hz = Z * H
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_tiles = num_pid_m * num_pid_hz
    start_pid = tl.program_id(0)
    n_iters = tl.cdiv(N_CTX, BLOCK_N) # go through n_iters, then switch to next tile

    tile_id = start_pid - NUM_SMS # start with start_pid
    ni = -1 # the actual loop from 0 to N_CTX step BLOCK_N

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = 0
    qvk_offset = start_m.to(tl.int64)
    off_hz = 0 # cross iteration
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # should we re-calculate qk_scale? or keep it as cross-loop variable?
    # value is the same across tiles
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # how to initialize q?
    q = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.bfloat16) # type of Q
    offs_hDim = tl.arange(0, HEAD_DIM)
    offs_hDim_qk = stride_qk * offs_hDim[None, :]
    offs_hDim_kk = stride_kk * offs_hDim[:, None]
    offs_hDim_vn = stride_vn * offs_hDim[None, :]
    for _ in range(0, n_iters * tiles_per_SM):
        ni = tl.where(ni == n_iters - 1, 0, ni + 1) # 0, ..., n_iters - 1, 0, ...
        if ni == 0:
            # prologue
            tile_id += NUM_SMS
            off_hz = tile_id // num_pid_m
            start_m = tile_id % num_pid_m
            off_z = off_hz // H
            off_h = off_hz % H
            qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        if ni == 0: # seperate this so it will stay in stage 0
            # load q: it will stay in SRAM throughout
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            #offs_hDim = tl.arange(0, HEAD_DIM)
            Q_ptrs = Q + qvk_offset + stride_qm * offs_m[:, None] + offs_hDim_qk #stride_qk * offs_hDim[None, :]
            q = tl.load(Q_ptrs)

        start_n = tl.multiple_of(ni * BLOCK_N, BLOCK_N)
        offs_n = ni * BLOCK_N + tl.arange(0, BLOCK_N)
        #offs_hDim_2 = tl.arange(0, HEAD_DIM)
        # is it better to not always recompute?
        K_ptrs = K + qvk_offset + offs_hDim_kk + stride_kn * offs_n[None, :] # dim 1
        V_ptrs = V + qvk_offset + offs_hDim_vn + stride_vk * offs_n[:, None] # dim 0
        # -- compute qk ----
        k = tl.load(K_ptrs)
        qk = tl.dot(q, k)
        # non-causal here
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_ptrs)
        #if fp8_v: # only handles bf16 for now
        #    p = p.to(tl.float8e5)
        #else:
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij

        if ni == n_iters - 1:
            # epilogue
            m_i += tl.math.log2(l_i)
            acc = acc / l_i[:, None]
            offs_m_2 = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # keep live range small by duplicating
            m_ptrs = M + off_hz * N_CTX + offs_m_2
            offs_hDim_3 = tl.arange(0, HEAD_DIM)
            O_ptrs = Out + qvk_offset + stride_om * offs_m_2[:, None] + stride_on * offs_hDim_3[None, :]
            tl.store(m_ptrs, m_i)
            tl.store(O_ptrs, acc.to(Out.type.element_ty))
            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)


@triton.autotune(list(filter(keep, configs)), key=["N_CTX"])
@triton.jit
def _attn_fwd_persistent_tma(Q, Out, desc_q, desc_k, desc_v, sm_scale, M, desc_o,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              NUM_SMS: tl.constexpr,
              STAGE: tl.constexpr  #
              ):
    # TODO(embg) remove TMA fence after __grid_constant__ lands
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_q], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_k], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_v], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_o], dtype=tl.int32, is_pure=False, pack=1)

    num_pid_hz = Z * H
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_tiles = num_pid_m * num_pid_hz
    start_pid = tl.program_id(0)
    n_iters = tl.cdiv(N_CTX, BLOCK_N) # go through n_iters, then switch to next tile

    tile_id = start_pid - NUM_SMS # start with start_pid
    ni = -1 # the actual loop from 0 to N_CTX step BLOCK_N

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = 0
    qvk_offset = start_m.to(tl.int64)
    off_hz = 0 # cross iteration
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # should we re-calculate qk_scale? or keep it as cross-loop variable?
    # value is the same across tiles
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # how to initialize q?
    q = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.bfloat16) # type of Q
    for _ in range(0, n_iters * tiles_per_SM):
        ni = tl.where(ni == n_iters - 1, 0, ni + 1) # 0, ..., n_iters - 1, 0, ...
        if ni == 0:
            # prologue
            tile_id += NUM_SMS
            off_hz = tile_id // num_pid_m
            start_m = tile_id % num_pid_m
            off_z = off_hz // H
            off_h = off_hz % H
            qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        if ni == 0: # seperate this so it will stay in stage 0
            # load q: it will stay in SRAM throughout
            q = tl._experimental_descriptor_load(  # load in row major
                desc_q,
                [(qvk_offset // stride_qm + start_m * BLOCK_M).to(tl.int32), 0],
                [BLOCK_M, HEAD_DIM],
                Q.dtype.element_ty,
            )

        start_n = tl.multiple_of(ni * BLOCK_N, BLOCK_N)
        # is it better to not always recompute?
        # -- compute qk ----
        k = tl._experimental_descriptor_load(  # load in row major
            desc_k,
            [start_n.to(tl.int32) + (qvk_offset // stride_kn).to(tl.int32), 0],
            [BLOCK_N, HEAD_DIM],
            Q.dtype.element_ty,
        )
        k = tl.trans(k)
        qk = tl.dot(q, k)
        # non-causal here
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        #if fp8_v: # only handles bf16 for now
        #    p = p.to(tl.float8e5)
        #else:
        v = tl._experimental_descriptor_load(  # load in row major
            desc_v,
            [(qvk_offset // stride_vk + start_n).to(tl.int32), 0],
            [BLOCK_N, HEAD_DIM],
            Q.dtype.element_ty,
        )
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij

        if ni == n_iters - 1:
            # epilogue
            m_i += tl.math.log2(l_i)
            acc = acc / l_i[:, None]
            offs_m_2 = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # keep live range small by duplicating
            m_ptrs = M + off_hz * N_CTX + offs_m_2
            tl.store(m_ptrs, m_i)
            tl._experimental_descriptor_store(desc_o, acc.to(Out.type.element_ty),
                [(qvk_offset // stride_om + start_m * BLOCK_M).to(tl.int32), 0])
            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.bfloat16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.bfloat16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                            Q, k, v, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        return dq, dk, dv, None, None


class _attention_tma(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        TMA_SIZE = 128
        BATCH, H, N_CTX = q.shape[0], q.shape[1], q.shape[2]
        # no autotune with fixed BLOCK_N
        '''
        BLOCK_N = 128
        desc_k = np.empty(TMA_SIZE, dtype=np.int8)
        desc_v = np.empty(TMA_SIZE, dtype=np.int8)
        # order is (0, 1) for fp8 in make_block_ptr, reverse here
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
            k.data_ptr(),
            BATCH * H * N_CTX,
            HEAD_DIM_Q,
            BLOCK_N,
            HEAD_DIM_Q,
            k.element_size(),
            desc_k,
        )
        if v.dtype == torch.float8_e5m2:
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                v.data_ptr(),
                BATCH * H * HEAD_DIM_Q,
                N_CTX,
                HEAD_DIM_Q,
                BLOCK_N,
                v.element_size(),
                desc_v,
            )
        else:
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                v.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                BLOCK_N,
                HEAD_DIM_Q,
                v.element_size(),
                desc_v,
            )
        desc_k = torch.tensor(desc_k, device=v.device)
        desc_v = torch.tensor(desc_v, device=v.device)
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        '''
        desc_k = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
        desc_v = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
        desc_q = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
        desc_o = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
        def grid_tma(META):
            nonlocal desc_k
            nonlocal desc_v
            nonlocal desc_q
            nonlocal desc_o
            q_buf = torch.empty_like(desc_q, device="cpu", pin_memory=True)
            k_buf = torch.empty_like(desc_k, device="cpu", pin_memory=True)
            v_buf = torch.empty_like(desc_v, device="cpu", pin_memory=True)
            o_buf = torch.empty_like(desc_o, device="cpu", pin_memory=True)
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                k.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META['BLOCK_N'],
                HEAD_DIM_Q,
                k.element_size(),
                k_buf.numpy(),
            )
            if v.dtype == torch.float8_e5m2:
                triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                    v.data_ptr(),
                    BATCH * H * HEAD_DIM_Q,
                    N_CTX,
                    HEAD_DIM_Q,
                    META['BLOCK_N'],
                    v.element_size(),
                    v_buf.numpy(),
                )
            else:
                triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                    v.data_ptr(),
                    BATCH * H * N_CTX,
                    HEAD_DIM_Q,
                    META['BLOCK_N'],
                    HEAD_DIM_Q,
                    v.element_size(),
                    v_buf.numpy(),
                )
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                q.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META['BLOCK_M'],
                HEAD_DIM_Q,
                q.element_size(),
                q_buf.numpy(),
            )
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                o.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META['BLOCK_M'],
                HEAD_DIM_Q,
                o.element_size(),
                o_buf.numpy(),
            )
            desc_q.copy_(q_buf, non_blocking=True)
            desc_k.copy_(k_buf, non_blocking=True)
            desc_v.copy_(v_buf, non_blocking=True)
            desc_o.copy_(o_buf, non_blocking=True)
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd_tma[grid_tma](
            q, v, o, desc_q, desc_k, desc_v, sm_scale, M, desc_o,  #
            #q, v, desc_k, desc_v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid_tma
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


class _attention_persistent(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert causal == False
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        #print(NUM_SMS, q.shape[2], q.shape[1], q.shape[0], HEAD_DIM_K)
        grid = lambda args: (min(NUM_SMS, triton.cdiv(q.shape[2], args["BLOCK_M"]) * q.shape[0] * q.shape[1]), 1, 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd_persistent[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            NUM_SMS=NUM_SMS,
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


class _attention_persistent_tma(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert causal == False
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        TMA_SIZE = 128
        BATCH, H, N_CTX = q.shape[0], q.shape[1], q.shape[2]

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        #print(NUM_SMS, q.shape[2], q.shape[1], q.shape[0], HEAD_DIM_K)

        desc_q = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
        desc_k = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
        desc_v = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
        desc_o = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
        def grid_tma(META):
            nonlocal desc_k
            nonlocal desc_v
            q_buf = torch.empty_like(desc_q, device="cpu", pin_memory=True)
            k_buf = torch.empty_like(desc_k, device="cpu", pin_memory=True)
            v_buf = torch.empty_like(desc_v, device="cpu", pin_memory=True)
            o_buf = torch.empty_like(desc_o, device="cpu", pin_memory=True)
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                q.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META['BLOCK_M'],
                HEAD_DIM_Q,
                q.element_size(),
                q_buf.numpy(),
            )
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                k.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META['BLOCK_N'],
                HEAD_DIM_Q,
                k.element_size(),
                k_buf.numpy(),
            )
            if v.dtype == torch.float8_e5m2:
                triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                    v.data_ptr(),
                    BATCH * H * HEAD_DIM_Q,
                    N_CTX,
                    HEAD_DIM_Q,
                    META['BLOCK_N'],
                    v.element_size(),
                    v_buf.numpy(),
                )
            else:
                triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                    v.data_ptr(),
                    BATCH * H * N_CTX,
                    HEAD_DIM_Q,
                    META['BLOCK_N'],
                    HEAD_DIM_Q,
                    v.element_size(),
                    v_buf.numpy(),
                )
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
                o.data_ptr(),
                BATCH * H * N_CTX,
                HEAD_DIM_Q,
                META['BLOCK_M'],
                HEAD_DIM_Q,
                o.element_size(),
                o_buf.numpy(),
            )
            desc_q.copy_(q_buf, non_blocking=True)
            desc_k.copy_(k_buf, non_blocking=True)
            desc_v.copy_(v_buf, non_blocking=True)
            desc_o.copy_(o_buf, non_blocking=True)
            #return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
            return (min(NUM_SMS, triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1]), 1, 1)

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd_persistent_tma[grid_tma](
            q, o, desc_q, desc_k, desc_v, sm_scale, M, desc_o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            NUM_SMS=NUM_SMS,
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid_tma
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


attention = _attention.apply
attention_tma = _attention_tma.apply
attention_persistent = _attention_persistent.apply
attention_persistent_tma = _attention_persistent_tma.apply
