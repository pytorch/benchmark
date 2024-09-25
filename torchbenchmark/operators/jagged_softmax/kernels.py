import itertools

import triton
import triton.language as tl


BLOCK_SIZES_RAGGED = [2**n for n in range(3, 12, 4)]
BLOCK_SIZES_M = [2**n for n in range(3, 7, 3)]
NUM_WARPS = [4, 8]
NUM_STAGES = [2, 4]


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_RAGGED": b_r,
                "BLOCK_SIZE_M": b_m,
            },
            num_warps=w,
            num_stages=s,
        )
        for b_r, b_m, w, s in itertools.product(
            BLOCK_SIZES_RAGGED,  # block sizes on non-reduction dimension
            BLOCK_SIZES_M,  # block sizes on reduction dimension
            NUM_WARPS,  # number of warps
            NUM_STAGES,  # number of stages
        )
    ],
    key=["M"],
)
@triton.jit
def triton_jagged_softmax_kernel_simple_fused_buffer_then_sum(
    input_ptr_values,  # pointer to input values (2D tensor)
    input_ptr_offsets,  # pointer to input offsets (1D tensor)
    output_ptr,  # pointer to output tensor (2D tensor)
    # matrix dimensions (input)
    M,  # number of elements in M-th dimension, with logical dimensions (B, *, M)
    MAX_SEQLEN,  # max length of ragged dimension
    # block sizes (input)
    BLOCK_SIZE_RAGGED: tl.constexpr,  # number of elements in ragged dimension per block, with logical dimensions (B, *, M)
    BLOCK_SIZE_M: tl.constexpr,  # number of elements in M-th dimension per block, with logical dimensions (B, *, M)
):
    pid = tl.program_id(axis=0)  # i-th tensor in nested tensor
    pid_b = pid // tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)

    buffer = tl.zeros(
        (BLOCK_SIZE_RAGGED, BLOCK_SIZE_M), dtype=tl.float32
    )  # create buffer as a row tensor

    # generate offsets and mask for BLOCK_SIZE_M (corresponding to pid_m)
    block_start_m = pid_m * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M

    ragged_start, ragged_end = tl.load(input_ptr_offsets + pid_b), tl.load(
        input_ptr_offsets + (pid_b + 1)
    )  # load start and end offsets for current program, similar to offsets[i] and offsets[i + 1]

    buffer_max_all = tl.full(
        (BLOCK_SIZE_RAGGED, BLOCK_SIZE_M), value=float("-inf"), dtype=tl.float32
    )  # compile buffer max (maximum value of buffer along ragged dimension)

    # calculate maximum value of buffer (along ragged dimension)
    for block_pos in range(
        0, MAX_SEQLEN, BLOCK_SIZE_RAGGED
    ):  # loop over ragged dimension, ranging until maximum seqlen
        block_start_ragged = (
            ragged_start + block_pos
        )  # offset block position by start of current program
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end

        idxs = (offsets_ragged[:, None] * M) + offsets_m
        mask = mask_ragged[:, None] & mask_m

        input = tl.load(input_ptr_values + idxs, mask=mask, other=float("-inf"))
        buffer_max_all = tl.maximum(buffer_max_all, input)

    buffer_max = tl.max(buffer_max_all, axis=0, keep_dims=True)

    # add exponentiated stable input to the buffer
    for block_pos in range(
        0, MAX_SEQLEN, BLOCK_SIZE_RAGGED
    ):  # loop over ragged dimension, ranging until maximum seqlen
        block_start_ragged = (
            ragged_start + block_pos
        )  # offset block position by start of current program
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end

        idxs = (offsets_ragged[:, None] * M) + offsets_m
        mask = mask_ragged[:, None] & mask_m

        input = tl.load(
            input_ptr_values + idxs, mask=mask, other=float("-inf")
        )  # cannot pad with 0, because input values may be 0
        buffer += tl.exp(input - buffer_max)

    # calculate sum of exponents (denominator of softmax function)
    buffer_exp_sum = tl.sum(buffer, axis=0)  # 2D tensor of shape (1, BLOCK_SIZE_M)

    # divide input (numerator of softmax function) by sum of exponents
    for block_pos in range(
        0, MAX_SEQLEN, BLOCK_SIZE_RAGGED
    ):  # loop over ragged dimension, ranging until maximum seqlen
        block_start_ragged = (
            ragged_start + block_pos
        )  # offset block position by start of current program
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end

        idxs = (offsets_ragged[:, None] * M) + offsets_m
        mask = mask_ragged[:, None] & mask_m

        input = tl.load(input_ptr_values + idxs, mask=mask, other=float("-inf"))
        output = tl.fdiv(tl.exp(input - buffer_max), buffer_exp_sum)

        tl.store(output_ptr + idxs, output, mask=mask)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_RAGGED": b_r,
                "BLOCK_SIZE_M": b_m,
            },
            num_warps=w,
            num_stages=s,
        )
        for b_r, b_m, w, s in itertools.product(
            BLOCK_SIZES_RAGGED,  # block sizes on non-reduction dimension
            BLOCK_SIZES_M,  # block sizes on reduction dimension
            NUM_WARPS,  # number of warps
            NUM_STAGES,  # number of stages
        )
    ],
    key=["M"],
)
@triton.jit
def triton_jagged_softmax_kernel_variable_length_loop_buffer_then_sum(
    input_ptr_values,  # pointer to input values (2D tensor)
    input_ptr_offsets,  # pointer to input offsets (1D tensor)
    output_ptr,  # pointer to output tensor (2D tensor)
    # matrix dimensions (input)
    M,  # number of elements in M-th dimension, with logical dimensions (B, *, M)
    # block sizes (input)
    BLOCK_SIZE_RAGGED: tl.constexpr,  # number of elements in ragged dimension per block, with logical dimensions (B, *, M)
    BLOCK_SIZE_M: tl.constexpr,  # number of elements in M-th dimension per block, with logical dimensions (B, *, M)
):
    pid = tl.program_id(axis=0)  # i-th tensor in nested tensor
    pid_b = pid // tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)

    buffer = tl.zeros(
        (BLOCK_SIZE_RAGGED, BLOCK_SIZE_M), dtype=tl.float32
    )  # create buffer as a row tensor

    # generate offsets and mask for BLOCK_SIZE_M (corresponding to pid_m)
    block_start_m = pid_m * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M

    ragged_start, ragged_end = tl.load(input_ptr_offsets + pid_b), tl.load(
        input_ptr_offsets + (pid_b + 1)
    )  # load start and end offsets for current program, similar to offsets[i] and offsets[i + 1]

    buffer_max_all = tl.full(
        (BLOCK_SIZE_RAGGED, BLOCK_SIZE_M), value=float("-inf"), dtype=tl.float32
    )  # compile buffer max (maximum value of buffer along ragged dimension)

    # calculate maximum value of buffer (along ragged dimension)
    for block_start_ragged in range(
        ragged_start, ragged_end, BLOCK_SIZE_RAGGED
    ):  # loop over ragged dimension, ranging until maximum seqlen
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end

        idxs = (offsets_ragged[:, None] * M) + offsets_m
        mask = mask_ragged[:, None] & mask_m

        input = tl.load(input_ptr_values + idxs, mask=mask, other=float("-inf"))
        buffer_max_all = tl.maximum(buffer_max_all, input)

    buffer_max = tl.max(buffer_max_all, axis=0, keep_dims=True)

    # add exponentiated stable input to the buffer
    for block_start_ragged in range(
        ragged_start, ragged_end, BLOCK_SIZE_RAGGED
    ):  # loop over ragged dimension, ranging until maximum seqlen
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end

        idxs = (offsets_ragged[:, None] * M) + offsets_m
        mask = mask_ragged[:, None] & mask_m

        input = tl.load(
            input_ptr_values + idxs, mask=mask, other=float("-inf")
        )  # cannot pad with 0, because input values may be 0
        buffer += tl.exp(input - buffer_max)

    # calculate sum of exponents (denominator of softmax function)
    buffer_exp_sum = tl.sum(buffer, axis=0)  # 2D tensor of shape (1, BLOCK_SIZE_M)

    # divide input (numerator of softmax function) by sum of exponents
    for block_start_ragged in range(
        ragged_start, ragged_end, BLOCK_SIZE_RAGGED
    ):  # loop over ragged dimension, ranging until maximum seqlen
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end

        idxs = (offsets_ragged[:, None] * M) + offsets_m
        mask = mask_ragged[:, None] & mask_m

        input = tl.load(input_ptr_values + idxs, mask=mask, other=float("-inf"))
        output = tl.fdiv(tl.exp(input - buffer_max), buffer_exp_sum)

        tl.store(output_ptr + idxs, output, mask=mask)
