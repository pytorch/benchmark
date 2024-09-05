// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <mutex>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "h100_fwd.cu"
#include "torch_helpers.cuh"

void tk_attention_forward(
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& o) {
  TORCH_CHECK(q.dim() == 4);
  TORCH_CHECK(k.dim() == 4);
  TORCH_CHECK(v.dim() == 4);

  // Batch sizes
  TORCH_CHECK(q.size(0) == k.size(0));
  TORCH_CHECK(q.size(0) == v.size(0));

  // Sequence length
  TORCH_CHECK(k.size(1) == v.size(1));

  // Num heads
  TORCH_CHECK(q.size(2) == k.size(2));
  TORCH_CHECK(q.size(2) == v.size(2));

  // Embedding per head
  TORCH_CHECK(q.size(3) == k.size(3));

  auto batch = q.size(0);
  auto heads = q.size(1);
  auto N = q.size(2);
  auto D = q.size(3);

  auto threads = NUM_WORKERS * kittens::WARP_THREADS;

  // make sure sequence length is multiple of 128 for now
  TORCH_CHECK(
      N % (NUM_WORKERS * kittens::TILE_DIM) == 0,
      "Please pad sequence length to be multiple of 128");

  // make sure D = 64 or 128
  TORCH_CHECK(
      D == 64 || D == 128, "Currently, only D = 64 or 128 is supported");

  // input must be bf16
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
  TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bf16");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bf16");
  TORCH_CHECK(o.scalar_type() == at::kBFloat16, "o must be bf16");

  const bf16* q_bf = reinterpret_cast<const bf16*>(q.data_ptr());
  const bf16* k_bf = reinterpret_cast<const bf16*>(k.data_ptr());
  const bf16* v_bf = reinterpret_cast<const bf16*>(v.data_ptr());
  bf16* o_bf = reinterpret_cast<bf16*>(o.data_ptr());

  if (D == 64) {
    CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<
        fwd_attend_ker_tile_dims<64>::qo_height,
        fwd_attend_ker_tile_dims<64>::tile_width,
        layout_q>>(
        q_bf,
        (batch * heads * N) / (fwd_attend_ker_tile_dims<64>::qo_height * 16));
    CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<
        fwd_attend_ker_tile_dims<64>::kv_height,
        fwd_attend_ker_tile_dims<64>::tile_width,
        layout_k>>(
        k_bf,
        (batch * heads * N) / (fwd_attend_ker_tile_dims<64>::kv_height * 16));
    CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<
        fwd_attend_ker_tile_dims<64>::kv_height,
        fwd_attend_ker_tile_dims<64>::tile_width,
        layout_v>>(
        v_bf,
        (batch * heads * N) / (fwd_attend_ker_tile_dims<64>::kv_height * 16));
    CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<
        fwd_attend_ker_tile_dims<64>::qo_height,
        fwd_attend_ker_tile_dims<64>::tile_width,
        layout_o>>(
        o_bf,
        (batch * heads * N) / (fwd_attend_ker_tile_dims<64>::qo_height * 16));

    unsigned long mem_size = 112000;
    cudaFuncSetAttribute(
        fwd_attend_ker_dim<64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size);

    dim3 grid(N / (NUM_WORKERS * kittens::TILE_DIM), batch * heads, 1);

    fwd_attend_ker_dim<64>
        <<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
  } else {
    CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<
        fwd_attend_ker_tile_dims<128>::qo_height,
        fwd_attend_ker_tile_dims<128>::tile_width,
        layout_q>>(
        q_bf,
        (batch * heads * N) / (fwd_attend_ker_tile_dims<128>::qo_height * 16));
    CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<
        fwd_attend_ker_tile_dims<128>::kv_height,
        fwd_attend_ker_tile_dims<128>::tile_width,
        layout_k>>(
        k_bf,
        (batch * heads * N) / (fwd_attend_ker_tile_dims<128>::kv_height * 16));
    CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<
        fwd_attend_ker_tile_dims<128>::kv_height,
        fwd_attend_ker_tile_dims<128>::tile_width,
        layout_v>>(
        v_bf,
        (batch * heads * N) / (fwd_attend_ker_tile_dims<128>::kv_height * 16));
    CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<
        fwd_attend_ker_tile_dims<128>::qo_height,
        fwd_attend_ker_tile_dims<128>::tile_width,
        layout_o>>(
        o_bf,
        (batch * heads * N) / (fwd_attend_ker_tile_dims<128>::qo_height * 16));

    unsigned long mem_size = 112000;
    cudaFuncSetAttribute(
        fwd_attend_ker_dim<128>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size);

    dim3 grid(N / (NUM_WORKERS * kittens::TILE_DIM), batch * heads, 1);

    fwd_attend_ker_dim<128>
        <<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
  }

  CHECK_CUDA_ERROR(cudaGetLastError());
}

// Abstract implementation
void tk_attention_forward_meta(
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& o) {
  TORCH_CHECK(q.dim() == 4);
  TORCH_CHECK(k.dim() == 4);
  TORCH_CHECK(v.dim() == 4);

  // Batch sizes
  TORCH_CHECK(q.sym_size(0) == k.sym_size(0));
  TORCH_CHECK(q.sym_size(0) == v.sym_size(0));

  // Sequence length
  TORCH_CHECK(k.sym_size(1) == v.sym_size(1));

  // Num heads
  TORCH_CHECK(q.sym_size(2) == k.sym_size(2));
  TORCH_CHECK(q.sym_size(2) == v.sym_size(2));

  // Embedding per head
  TORCH_CHECK(q.sym_size(3) == k.sym_size(3));

  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
  TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bf16");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16, "v must be bf16");
  TORCH_CHECK(o.scalar_type() == at::kBFloat16, "o must be bf16");
  return;
}

// torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o
TORCH_LIBRARY_FRAGMENT(tk, m) {
  m.def("attention_forward(Tensor q, Tensor k, Tensor v, Tensor o) -> ()");
}

TORCH_LIBRARY_IMPL(tk, CUDA, m) {
  m.impl("attention_forward", tk_attention_forward);
}

TORCH_LIBRARY_IMPL(tk, Meta, m) {
  m.impl("attention_forward", TORCH_FN(tk_attention_forward_meta));
}
