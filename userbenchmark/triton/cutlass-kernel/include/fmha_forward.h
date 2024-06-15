// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

template <typename PrecType, typename OutputType, typename AccumType, int HEADDIM>
void fmhaForwardDevice(
    int SEQLEN,
    int KEYLEN,
    int NUMHEADS,
    int BATCH,
    PrecType const* tensorQ,
    PrecType const* tensorK,
    OutputType const* tensorV,
    OutputType* tensorS,
    OutputType* tensorO,
    AccumType* miOut,
    AccumType* sPrimeOut,
    int iterations,
    float scale,
    cudaStream_t stream = 0);
