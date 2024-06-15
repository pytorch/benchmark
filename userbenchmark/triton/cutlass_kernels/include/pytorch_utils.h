// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cutlass/numeric_types.h>

template <typename scalar_t>
struct CutlassToAtenDtype;

template <>
struct CutlassToAtenDtype<cutlass::half_t> {
  using scalar_t = cutlass::half_t;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Half;
  }
};

template <>
struct CutlassToAtenDtype<cutlass::bfloat16_t> {
  using scalar_t = cutlass::bfloat16_t;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::BFloat16;
  }
};

template <>
struct CutlassToAtenDtype<float> {
  using scalar_t = float;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Float;
  }
};

template <>
struct CutlassToAtenDtype<cutlass::float_e4m3_t> {
  using scalar_t = float;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Float8_e4m3fn;
  }
};
