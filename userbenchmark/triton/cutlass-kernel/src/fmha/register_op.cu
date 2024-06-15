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

// #include "autogen/cutlassF.h"
#include "pytorch_utils.h"
#include "fmha_forward.h"

template <typename PrecType, typename OutputType, int HEADDIM>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fmha_forward(
    const int64_t& seq_length,
    const int64_t& key_length,
    const int64_t& batch,
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const float& scale) {
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  //   CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  //   CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  //   CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  at::Tensor S = at::empty(
      {B, M, num_heads, Kv},
      query.options().dtype(CutlassToAtenDtype<PrecType>::atScalarType()));
  at::Tensor ret = at::empty(
      {B, M, num_heads, Kv},
      query.options().dtype(CutlassToAtenDtype<OutputType>::atScalarType()));
  using AccumType = float; // AccumType is always float.

  at::Tensor devMiOut = at::empty(
      {B, M, num_heads},
      query.options().dtype(CutlassToAtenDtype<AccumType>::atScalarType()));
  at::Tensor devSprimeOut = at::empty(
      {B, M, num_heads},
      query.options().dtype(CutlassToAtenDtype<AccumType>::atScalarType()));

  fmhaForwardDevice<PrecType, OutputType, AccumType, HEADDIM>(
      seq_length,
      key_length,
      num_heads,
      B,
      reinterpret_cast<PrecType const*>(query.data_ptr()),
      reinterpret_cast<PrecType const*>(key.data_ptr()),
      reinterpret_cast<OutputType const*>(value.data_ptr()),
      reinterpret_cast<OutputType*>(S.data_ptr()),
      reinterpret_cast<OutputType*>(ret.data_ptr()),
      reinterpret_cast<AccumType*>(devMiOut.data_ptr()),
      reinterpret_cast<AccumType*>(devSprimeOut.data_ptr()),
      1,
      scale,
      stream);

  return std::make_tuple(S, ret, devMiOut, devSprimeOut);
}

template<typename compute_data_type, typename output_data_type>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
launch_forward(
    const int64_t& seq_length,
    const int64_t& key_length,
    const int64_t& batch,
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const double& scale,
    const int64_t& Kdim) {
    if (Kdim == 64) {
      return fmha_forward<compute_data_type, output_data_type, 64>(
          seq_length, key_length, batch, query, key, value, scale);
    } else if (Kdim == 128) {
      return fmha_forward<compute_data_type, output_data_type, 128>(
          seq_length, key_length, batch, query, key, value, scale);
    } else if (Kdim == 256) {
      return fmha_forward<compute_data_type, output_data_type, 256>(
          seq_length, key_length, batch, query, key, value, scale);
    }
    throw std::runtime_error("Kdim wrong");
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fmha_forward_dispatch(
    const int64_t& seq_length,
    const int64_t& key_length,
    const int64_t& batch,
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const double& scale) {
  int64_t Kdim = query.size(-1);

  if (query.scalar_type() == at::kHalf){
    return launch_forward<cutlass::half_t, cutlass::half_t>(seq_length, key_length, batch, query, key, value, scale, Kdim);
  }
  else if (query.scalar_type() == at::kBFloat16){
    return launch_forward<cutlass::bfloat16_t, cutlass::bfloat16_t>(seq_length, key_length, batch, query, key, value, scale, Kdim);
  }
  else if (query.scalar_type() == at::kFloat8_e4m3fn){
    return launch_forward<cutlass::float_e4m3_t, cutlass::bfloat16_t>(seq_length, key_length, batch, query, key, value, scale, Kdim);
  }
  else {
        std::cout << "unsupported data type: " << query.scalar_type() << std::endl;
        throw std::runtime_error("Unsupported data type");
  }

}

// Abstract implementation
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fmha_forward_dispatch_meta(
    const int64_t& seq_length,
    const int64_t& key_length,
    const int64_t& batch,
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const double& scale) {

  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.sym_size(0) == key.sym_size(0));
  TORCH_CHECK(query.sym_size(0) == value.sym_size(0));

  // Sequence length
  TORCH_CHECK(key.sym_size(1) == value.sym_size(1));

  // Num heads
  TORCH_CHECK(query.sym_size(2) == key.sym_size(2));
  TORCH_CHECK(query.sym_size(2) == value.sym_size(2));

  // Embedding per head
  TORCH_CHECK(query.sym_size(3) == key.sym_size(3));

  at::SymInt B = query.sym_size(0);
  at::SymInt M = query.sym_size(1);
  at::SymInt num_heads = query.sym_size(-2);
  at::SymInt Kv = value.sym_size(-1);

  at::Tensor S = at::empty_symint({B, M, num_heads, Kv}, query.options());
  at::Tensor ret = at::empty_symint({B, M, num_heads, Kv}, query.options());
  at::Tensor devMiOut = at::empty_symint({B, M, num_heads}, query.options());
  at::Tensor devSprimeOut = at::empty_symint({B, M, num_heads}, query.options());

  return std::make_tuple(S, ret, devMiOut, devSprimeOut);
}

TORCH_LIBRARY_FRAGMENT(cutlass, m) {
  m.def(
      "fmha_forward(int seq_length, int key_length, int batch, Tensor query, Tensor key, Tensor value, float scale) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(cutlass, CUDA, m) {
  m.impl("fmha_forward", fmha_forward_dispatch);
}

TORCH_LIBRARY_IMPL(cutlass, Meta, m) {
  m.impl(
      "fmha_forward",
      TORCH_FN(fmha_forward_dispatch_meta));
}
