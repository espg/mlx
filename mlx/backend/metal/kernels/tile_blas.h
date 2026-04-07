// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/defines.h"

// clang-format off

// Instantiation macros for tile BLAS kernels.
// Each macro instantiates a kernel for a given tile size N.

#define instantiate_tile_potrf(N) \
    instantiate_kernel("tile_potrf_float32_" #N, tile_potrf, N)

#define instantiate_tile_trsm_variant(N, SIDE, UPLO, SNAME, UNAME) \
    instantiate_kernel( \
        "tile_trsm_" SNAME "_" UNAME "_float32_" #N, tile_trsm, N, SIDE, UPLO)

#define instantiate_tile_trsm_all(N) \
    instantiate_tile_trsm_variant(N, 0, 0, "left", "lower") \
    instantiate_tile_trsm_variant(N, 1, 0, "right", "lower")

#define instantiate_tile_syrk(N) \
    instantiate_kernel("tile_syrk_float32_" #N, tile_syrk, N)

#define instantiate_tile_syrk_atomic(N) \
    instantiate_kernel("tile_syrk_atomic_float32_" #N, tile_syrk_atomic, N)

#define instantiate_tile_gemm(N) \
    instantiate_kernel("tile_gemm_float32_" #N, tile_gemm, N)

#define instantiate_tile_gemm_atomic(N) \
    instantiate_kernel("tile_gemm_atomic_float32_" #N, tile_gemm_atomic, N)

#define instantiate_tile_geadd(N) \
    instantiate_kernel("tile_geadd_float32_" #N, tile_geadd, N)

#define instantiate_tile_blas_all(N) \
    instantiate_tile_potrf(N) \
    instantiate_tile_trsm_all(N) \
    instantiate_tile_syrk(N) \
    instantiate_tile_syrk_atomic(N) \
    instantiate_tile_gemm(N) \
    instantiate_tile_gemm_atomic(N) \
    instantiate_tile_geadd(N)

// clang-format on
