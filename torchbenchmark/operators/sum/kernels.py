import itertools

import torch
import triton
import triton.language as tl


@triton.jit
def triton_sum_kernel_scalar_result(
    input_ptr,  # pointer to input matrix
    output_ptr,  # pointer to output matrix
    M,  # number of elements
    BLOCK_SIZE_M: tl.constexpr,  # number of elements per block
):
    pid = tl.program_id(axis=0)  # i-th block of input

    block_start = pid * BLOCK_SIZE_M
    # offsets have shape equal to input shape
    offsets = block_start + tl.arange(
        0, BLOCK_SIZE_M
    )  # create 1D vector (input shape) ranging from beginning to end of this program's block

    # mask has shape equal to input shape
    mask = offsets < M  # mask out offsets that are out of bounds for input

    # loaded pointers have shape equal to input shape
    x = tl.load(
        input_ptr + offsets, mask=mask, other=mask
    )  # load input, where the loaded pointers are in the desired input shape

    output = tl.sum(x)

    # output_offsets have shape equal to output shape
    output_offsets = tl.arange(
        0, 1
    )  # create offsets for scalar output pointer (output shape == (1,))

    # stored pointers have shape equal to output shape
    tl.store(
        output_ptr + output_offsets, output
    )  # store output, where the stored pointers are in the desired output shape


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_NON_REDUCE_DIM": b_nr,
                "BLOCK_SIZE_REDUCE_DIM": b_r,
            },
            num_warps=w,
        )
        for b_nr, b_r, w in itertools.product(
            [2, 4, 8, 16], [2, 4, 8, 16], [2, 4, 8]  # block sizes on non-reduction dimension, block sizes on reduction dimension, number of warps
        )
    ],
    key=["M", "N"],
)
@triton.jit
def triton_sum_kernel_1D_result_sum_then_buffer(
    input_ptr,  # pointer to input matrix
    output_ptr,  # pointer to output matrix
    # matrix dimensions (input)
    M,  # number of rows
    N,  # number of columns
    # block sizes (input)
    BLOCK_SIZE_NON_REDUCE_DIM: tl.constexpr,  # number of elements in non-reduction dimension per block
    BLOCK_SIZE_REDUCE_DIM: tl.constexpr,  # number of elements in reduction dimension per block
    # reduction dimension
    dim: tl.constexpr,  # dimension along which to sum
):
    """
    Sum blocks of input using Triton and store in buffer
    """

    pid = tl.program_id(axis=0)  # i-th block of input

    reduce_dim_len = M if dim == 0 else N
    non_reduce_dim_len = N if dim == 0 else M

    buffer = tl.zeros(
        (1, BLOCK_SIZE_NON_REDUCE_DIM), dtype=tl.float32
    )  # create buffer as a row tensor

    block_start_non_reduce_dim = pid * BLOCK_SIZE_NON_REDUCE_DIM
    offsets_non_reduce_dim = block_start_non_reduce_dim + tl.arange(
        0, BLOCK_SIZE_NON_REDUCE_DIM
    )
    mask_non_reduce_dim = offsets_non_reduce_dim < non_reduce_dim_len

    for block_start_reduce_dim in range(0, reduce_dim_len, BLOCK_SIZE_REDUCE_DIM):
        offsets_reduce_dim = block_start_reduce_dim + tl.arange(
            0, BLOCK_SIZE_REDUCE_DIM
        )
        mask_reduce_dim = offsets_reduce_dim < reduce_dim_len

        idxs, mask = None, None
        if dim == 0:
            idxs = (
                offsets_reduce_dim[:, None] * non_reduce_dim_len
            ) + offsets_non_reduce_dim
            mask = mask_reduce_dim[:, None] & mask_non_reduce_dim
        elif dim == 1:
            idxs = (
                offsets_non_reduce_dim[:, None] * reduce_dim_len
            ) + offsets_reduce_dim
            mask = mask_non_reduce_dim[:, None] & mask_reduce_dim

        input = tl.load(input_ptr + idxs, mask=mask, other=mask)

        buffer += tl.sum(input, axis=dim)

    buffer_view = buffer.reshape(
        (BLOCK_SIZE_NON_REDUCE_DIM,), can_reorder=True
    )  # reshape buffer to 1D, as tl.sum may return a 2D tensor

    tl.store(output_ptr + offsets_non_reduce_dim, buffer_view, mask=mask_non_reduce_dim)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_NON_REDUCE_DIM": b,
                "BLOCK_SIZE_REDUCE_DIM": b,
            },
            num_warps=w,
        )
        for b, w in itertools.product(
            [2, 4, 8, 16], [2, 4, 8]  # block sizes  # number of warps
        )
    ],
    key=["M", "N"],
)
@triton.jit
def triton_sum_kernel_1D_result_buffer_then_sum(
    input_ptr,  # pointer to input matrix
    output_ptr,  # pointer to output matrix
    # matrix dimensions (input)
    M,  # number of rows
    N,  # number of columns
    # block sizes (input)
    BLOCK_SIZE_NON_REDUCE_DIM: tl.constexpr,  # number of elements in non-reduction dimension per block
    BLOCK_SIZE_REDUCE_DIM: tl.constexpr,  # number of elements in reduction dimension per block
    # reduction dimension
    dim: tl.constexpr,  # dimension along which to sum
):
    """
    Add blocks of input to a buffer and sum the buffer using Triton
    """

    pid = tl.program_id(axis=0)  # i-th block of input

    reduce_dim_len = M if dim == 0 else N
    non_reduce_dim_len = N if dim == 0 else M

    buffer = tl.zeros(
        (BLOCK_SIZE_REDUCE_DIM, BLOCK_SIZE_NON_REDUCE_DIM), dtype=tl.float32
    )  # create buffer as a 2D tensor

    block_start_non_reduce_dim = pid * BLOCK_SIZE_NON_REDUCE_DIM
    offsets_non_reduce_dim = block_start_non_reduce_dim + tl.arange(
        0, BLOCK_SIZE_NON_REDUCE_DIM
    )
    mask_non_reduce_dim = offsets_non_reduce_dim < non_reduce_dim_len

    for block_start_reduce_dim in range(0, reduce_dim_len, BLOCK_SIZE_REDUCE_DIM):
        offsets_reduce_dim = block_start_reduce_dim + tl.arange(
            0, BLOCK_SIZE_REDUCE_DIM
        )
        mask_reduce_dim = offsets_reduce_dim < reduce_dim_len

        idxs, mask = None, None
        if dim == 0:
            idxs = (
                offsets_reduce_dim[:, None] * non_reduce_dim_len
            ) + offsets_non_reduce_dim
            mask = mask_reduce_dim[:, None] & mask_non_reduce_dim
        elif dim == 1:
            idxs = (
                offsets_non_reduce_dim[:, None] * reduce_dim_len
            ) + offsets_reduce_dim
            mask = mask_non_reduce_dim[:, None] & mask_reduce_dim

        buffer += tl.load(input_ptr + idxs, mask=mask, other=mask)

    buffer_sum = tl.sum(buffer, axis=dim)

    buffer_view = buffer_sum.reshape(
        (BLOCK_SIZE_NON_REDUCE_DIM,), can_reorder=True
    )  # reshape buffer to 1D, as tl.sum may return a 2D tensor

    tl.store(output_ptr + offsets_non_reduce_dim, buffer_view, mask=mask_non_reduce_dim)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_K": b},
            num_warps=w,
        )
        for b, w in itertools.product(
            [2, 4, 16, 32, 128, 256], [2, 4, 8]  # block sizes, number of warps
        )
    ],
    key=["N"],
)
@triton.jit
def triton_sum_kernel_2D_result_dim_1(
    input_ptr,  # pointer to input matrix
    output_ptr,  # pointer to output matrix
    # matrix dimensions (input)
    M: tl.constexpr,  # number of elements in M-th dimension
    N: tl.constexpr,  # number of elements in N-th dimension
    K: tl.constexpr,  # number of elements in K-th dimension
    # block sizes (input)
    BLOCK_SIZE_N: tl.constexpr,  # number of elements in block on N-th dimension
    BLOCK_SIZE_K: tl.constexpr,  # number of elements in block on K-th dimension
):
    # input block shape: (1, N, BLOCK_SIZE_K)

    pid = tl.program_id(axis=0)  # i-th block of input

    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid % tl.cdiv(K, BLOCK_SIZE_K)

    block_start_n = (
        0  # assuming that the entire reduction dimension fits within one thread block
    )
    block_start_k = pid_k * BLOCK_SIZE_K

    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    mask_n = offsets_n < N
    mask_k = offsets_k < K

    # idxs has shape (N, BLOCK_SIZE_K)
    idxs_base = (offsets_n[:, None] * K) + offsets_k
    idxs = idxs_base + (
        pid_m * N * K
    )  # increment idxs by the number of elements in all previous blocks

    # mask has shape (N, BLOCK_SIZE_K)
    mask = mask_n[:, None] & mask_k

    # loaded pointers have shape (N, K)
    input = tl.load(
        input_ptr + idxs, mask=mask, other=0
    )  # zero out masked values from input

    # output has shape (1, BLOCK_SIZE_K)
    output = tl.sum(input, axis=0)

    output_offsets = (pid_m * K) + offsets_k

    # stored pointers have shape (1, BLOCK_SIZE_K)
    tl.store(
        output_ptr + output_offsets, output, mask=mask_k
    )  # store a 1D vector into a specific row of 2D output
