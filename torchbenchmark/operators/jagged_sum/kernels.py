import itertools

import triton
import triton.language as tl


BLOCK_SIZES = [2**n for n in range(3, 7, 3)]
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
            BLOCK_SIZES,  # block sizes on non-reduction dimension
            BLOCK_SIZES,  # block sizes on reduction dimension
            NUM_WARPS,  # number of warps
            NUM_STAGES,  # number of stages
        )
    ],
    key=["M"],
)
@triton.jit
def triton_jagged_sum_kernel_simple_fused_sum_then_buffer(
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
    pid_ragged = pid // tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)

    buffer = tl.zeros(
        (1, BLOCK_SIZE_M), dtype=tl.float32
    )  # create buffer as a row tensor

    block_start_m = pid_m * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M

    ragged_start, ragged_end = tl.load(input_ptr_offsets + pid_ragged), tl.load(
        input_ptr_offsets + (pid_ragged + 1)
    )  # load start and end offsets for current program, similar to offsets[i] and offsets[i + 1]

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

        input = tl.load(input_ptr_values + idxs, mask=mask, other=0)

        buffer += tl.sum(input, axis=0)

    buffer_view = buffer.reshape(
        (BLOCK_SIZE_M,),
    )  # reshape buffer to 1D, as tl.sum may return a 2D tensor

    output_offsets = offsets_m + (
        pid_ragged * M
    )  # output is offset by both ragged dimension and M-th dimension
    output_mask = output_offsets < (M * (pid_ragged + 1))

    tl.store(output_ptr + output_offsets, buffer_view, mask=output_mask)


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
            BLOCK_SIZES,  # block sizes on non-reduction dimension
            BLOCK_SIZES,  # block sizes on reduction dimension
            NUM_WARPS,  # number of warps
            NUM_STAGES,  # number of stages
        )
    ],
    key=["M"],
)
@triton.jit
def triton_jagged_sum_kernel_simple_fused_buffer_then_sum(
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
    pid_ragged = pid // tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)

    buffer = tl.zeros(
        (BLOCK_SIZE_RAGGED, BLOCK_SIZE_M), dtype=tl.float32
    )  # create buffer as a row tensor

    block_start_m = pid_m * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M

    ragged_start, ragged_end = tl.load(input_ptr_offsets + pid_ragged), tl.load(
        input_ptr_offsets + (pid_ragged + 1)
    )  # load start and end offsets for current program, similar to offsets[i] and offsets[i + 1]

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

        buffer += tl.load(input_ptr_values + idxs, mask=mask, other=0)

    buffer_sum = tl.sum(buffer, axis=0)

    buffer_view = buffer_sum.reshape(
        (BLOCK_SIZE_M,),
    )  # reshape buffer to 1D, as tl.sum may return a 2D tensor

    output_offsets = offsets_m + (
        pid_ragged * M
    )  # output is offset by both ragged dimension and M-th dimension
    output_mask = output_offsets < (M * (pid_ragged + 1))

    tl.store(output_ptr + output_offsets, buffer_view, mask=output_mask)


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
            BLOCK_SIZES,  # block sizes on non-reduction dimension
            BLOCK_SIZES,  # block sizes on reduction dimension
            NUM_WARPS,  # number of warps
            NUM_STAGES,  # number of stages
        )
    ],
    key=["M"],
)
@triton.jit
def triton_jagged_sum_kernel_variable_length_loop_sum_then_buffer(
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
        (1, BLOCK_SIZE_M), dtype=tl.float32
    )  # create buffer as a row tensor

    block_start_m = pid_m * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M

    ragged_start, ragged_end = tl.load(input_ptr_offsets + pid_b), tl.load(
        input_ptr_offsets + (pid_b + 1)
    )  # load start and end offsets for current program, similar to offsets[i] and offsets[i + 1]

    for block_start_ragged in range(
        ragged_start, ragged_end, BLOCK_SIZE_RAGGED
    ):  # loop over ragged dimension, ranging from beginning to end of this program in values tensor
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end

        idxs = (offsets_ragged[:, None] * M) + offsets_m
        mask = mask_ragged[:, None] & mask_m

        input = tl.load(input_ptr_values + idxs, mask=mask, other=0)

        buffer += tl.sum(input, axis=0)

    buffer_view = buffer.reshape(
        (BLOCK_SIZE_M,),
    )  # reshape buffer to 1D, as tl.sum may return a 2D tensor

    output_offsets = offsets_m + (
        pid_b * M
    )  # output is offset by both B-th dimension and M-th dimension
    output_mask = output_offsets < (M * (pid_b + 1))

    tl.store(output_ptr + output_offsets, buffer_view, mask=output_mask)


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
            BLOCK_SIZES,  # block sizes on non-reduction dimension
            BLOCK_SIZES,  # block sizes on reduction dimension
            NUM_WARPS,  # number of warps
            NUM_STAGES,  # number of stages
        )
    ],
    key=["M"],
)
@triton.jit
def triton_jagged_sum_kernel_variable_length_loop_buffer_then_sum(
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
    pid_ragged = pid // tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)

    buffer = tl.zeros(
        (BLOCK_SIZE_RAGGED, BLOCK_SIZE_M), dtype=tl.float32
    )  # create buffer as a row tensor

    block_start_m = pid_m * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M

    ragged_start, ragged_end = tl.load(input_ptr_offsets + pid_ragged), tl.load(
        input_ptr_offsets + (pid_ragged + 1)
    )  # load start and end offsets for current program, similar to offsets[i] and offsets[i + 1]

    for block_start_ragged in range(
        ragged_start, ragged_end, BLOCK_SIZE_RAGGED
    ):  # loop over ragged dimension, ranging from beginning to end of this program in values tensor
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end

        idxs = (offsets_ragged[:, None] * M) + offsets_m
        mask = mask_ragged[:, None] & mask_m

        buffer += tl.load(input_ptr_values + idxs, mask=mask, other=0)

    buffer_sum = tl.sum(buffer, axis=0)

    buffer_view = buffer_sum.reshape(
        (BLOCK_SIZE_M,),
    )  # reshape buffer to 1D, as tl.sum may return a 2D tensor

    output_offsets = offsets_m + (
        pid_ragged * M
    )  # output is offset by both ragged dimension and M-th dimension
    output_mask = output_offsets < (M * (pid_ragged + 1))

    tl.store(output_ptr + output_offsets, buffer_view, mask=output_mask)
