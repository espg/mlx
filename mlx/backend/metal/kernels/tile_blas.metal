// Copyright © 2025 Apple Inc.

#include <metal_math>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/tile_blas.h"

using namespace metal;

// Padded shared memory stride to avoid bank conflicts.
// stride = N+1 ensures gcd(stride, 32) = 1 for all tile sizes,
// giving conflict-free access when threads index different rows.
template <int N>
constexpr constant int S = N + 1;

// ============================================================
// tile_potrf: In-place lower Cholesky of N×N tiles
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
// Each threadgroup factors one tile. Thread lid handles column lid
// during load/store and distributes sub-diagonal work during factorization.
//
// Buffers:
//   0: tiles     — contiguous tile storage (read/write)
//   1: offsets_a  — per-op element offset to tile A
// ============================================================
template <int N>
[[kernel]] void tile_potrf(
    device float* tiles [[buffer(0)]],
    const device int* offsets_a [[buffer(1)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  device float* A = tiles + offsets_a[op_idx];

  threadgroup float L[N * S<N>];

  // Coalesced load: each thread reads column lid across all rows
  for (int row = 0; row < N; row++) {
    L[row * S<N> + lid] = A[row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Column-by-column Cholesky: L·Lᵀ = A
  for (int j = 0; j < N; j++) {
    // Thread 0 computes diagonal element
    if (lid == 0) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        float v = L[j * S<N> + k];
        sum += v * v;
      }
      float d = L[j * S<N> + j] - sum;
      L[j * S<N> + j] = (d > 0.0f) ? metal::sqrt(d) : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float diag = L[j * S<N> + j];

    // Each thread computes at most one sub-diagonal element
    for (int i = j + 1 + int(lid); i < N; i += N) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        sum += L[i * S<N> + k] * L[j * S<N> + k];
      }
      L[i * S<N> + j] = (L[i * S<N> + j] - sum) / diag;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Coalesced write-back: lower triangle of L, zero upper
  for (int row = 0; row < N; row++) {
    A[row * N + lid] = (row >= int(lid)) ? L[row * S<N> + lid] : 0.0f;
  }
}

// ============================================================
// tile_trsm: Triangular solve on N×N tiles
//
// SIDE=0 (Left):  X = L⁻¹·B  (forward substitution, each column independent)
// SIDE=1 (Right): X = B·L⁻ᵀ  (equivalent to L·Xᵀ = Bᵀ, each row independent)
// UPLO=0: L is lower triangular
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
// L is loaded into shared memory; B is kept in registers.
//
// Buffers:
//   0: tiles     — contiguous tile storage (read/write)
//   1: offsets_l  — per-op element offset to tile L
//   2: offsets_b  — per-op element offset to tile B (overwritten with X)
// ============================================================
template <int N, int SIDE, int UPLO>
[[kernel]] void tile_trsm(
    device float* tiles [[buffer(0)]],
    const device int* offsets_l [[buffer(1)]],
    const device int* offsets_b [[buffer(2)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  const device float* L_dev = tiles + offsets_l[op_idx];
  device float* B_dev = tiles + offsets_b[op_idx];

  threadgroup float sL[N * S<N>];

  // Coalesced load of L into shared memory
  for (int row = 0; row < N; row++) {
    sL[row * S<N> + lid] = L_dev[row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (SIDE == 0 && UPLO == 0) {
    // Left, Lower: solve L·X = B for X = L⁻¹·B
    // Thread lid processes column lid of B (independent columns)
    float b[N];
    for (int i = 0; i < N; i++) {
      b[i] = B_dev[i * N + lid];
    }

    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        sum += sL[j * S<N> + k] * b[k];
      }
      b[j] = (b[j] - sum) / sL[j * S<N> + j];
    }

    for (int i = 0; i < N; i++) {
      B_dev[i * N + lid] = b[i];
    }

  } else if constexpr (SIDE == 1 && UPLO == 0) {
    // Right, Lower: solve X·Lᵀ = B for X = B·L⁻ᵀ
    // Equivalent to L·Xᵀ = Bᵀ (forward sub on transposed system)
    // Thread lid processes row lid of B (independent rows)
    float b[N];
    for (int j = 0; j < N; j++) {
      b[j] = B_dev[lid * N + j];
    }

    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        sum += sL[j * S<N> + k] * b[k];
      }
      b[j] = (b[j] - sum) / sL[j * S<N> + j];
    }

    for (int j = 0; j < N; j++) {
      B_dev[lid * N + j] = b[j];
    }
  }
}

// ============================================================
// tile_syrk: C -= A·Aᵀ (lower triangle of C only)
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
// Thread lid computes row lid of the update (elements j = 0..lid).
//
// Buffers:
//   0: tiles     — contiguous tile storage (read/write)
//   1: offsets_c  — per-op element offset to tile C
//   2: offsets_a  — per-op element offset to tile A
// ============================================================
template <int N>
[[kernel]] void tile_syrk(
    device float* tiles [[buffer(0)]],
    const device int* offsets_c [[buffer(1)]],
    const device int* offsets_a [[buffer(2)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  device float* C_ptr = tiles + offsets_c[op_idx];
  const device float* A_ptr = tiles + offsets_a[op_idx];

  threadgroup float sA[N * S<N>];

  // Coalesced load of A
  for (int row = 0; row < N; row++) {
    sA[row * S<N> + lid] = A_ptr[row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Cache row lid of A in registers for reuse across dot products
  float a_row[N];
  for (int k = 0; k < N; k++) {
    a_row[k] = sA[lid * S<N> + k];
  }

  // C[lid, j] -= dot(A[lid,:], A[j,:]) for j = 0..lid (lower triangle)
  for (int j = 0; j <= int(lid); j++) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += a_row[k] * sA[j * S<N> + k];
    }
    C_ptr[lid * N + j] -= sum;
  }
}

// ============================================================
// tile_syrk_atomic: C -= A·Aᵀ with atomic updates (lower triangle)
//
// Used when multiple threadgroups concurrently update the same C tile.
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
//
// Buffers:
//   0: tiles_out  — output tile storage (atomic float, read/write)
//   1: tiles_in   — input tile storage (float, read-only)
//   2: offsets_c   — per-op element offset to tile C in tiles_out
//   3: offsets_a   — per-op element offset to tile A in tiles_in
// ============================================================
template <int N>
[[kernel]] void tile_syrk_atomic(
    device mlx_atomic<float>* tiles_out [[buffer(0)]],
    const device float* tiles_in [[buffer(1)]],
    const device int* offsets_c [[buffer(2)]],
    const device int* offsets_a [[buffer(3)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  int c_off = offsets_c[op_idx];
  int a_off = offsets_a[op_idx];

  threadgroup float sA[N * S<N>];

  for (int row = 0; row < N; row++) {
    sA[row * S<N> + lid] = tiles_in[a_off + row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float a_row[N];
  for (int k = 0; k < N; k++) {
    a_row[k] = sA[lid * S<N> + k];
  }

  for (int j = 0; j <= int(lid); j++) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += a_row[k] * sA[j * S<N> + k];
    }
    mlx_atomic_fetch_add_explicit(tiles_out, -sum, c_off + lid * N + j);
  }
}

// ============================================================
// tile_gemm: C -= A·B
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
// Thread lid computes row lid of the result.
// Both A and B loaded into shared memory.
//
// Buffers:
//   0: tiles     — contiguous tile storage (read/write)
//   1: offsets_c  — per-op element offset to tile C
//   2: offsets_a  — per-op element offset to tile A
//   3: offsets_b  — per-op element offset to tile B
// ============================================================
template <int N>
[[kernel]] void tile_gemm(
    device float* tiles [[buffer(0)]],
    const device int* offsets_c [[buffer(1)]],
    const device int* offsets_a [[buffer(2)]],
    const device int* offsets_b [[buffer(3)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  device float* C_ptr = tiles + offsets_c[op_idx];
  const device float* A_ptr = tiles + offsets_a[op_idx];
  const device float* B_ptr = tiles + offsets_b[op_idx];

  threadgroup float sA[N * S<N>];
  threadgroup float sB[N * S<N>];

  // Coalesced load of A and B
  for (int row = 0; row < N; row++) {
    sA[row * S<N> + lid] = A_ptr[row * N + lid];
    sB[row * S<N> + lid] = B_ptr[row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate C[lid,:] -= A[lid,:] · B
  // Restructured as outer product accumulation for better data reuse
  float c[N];
  for (int j = 0; j < N; j++) {
    c[j] = C_ptr[lid * N + j];
  }

  for (int k = 0; k < N; k++) {
    float a = sA[lid * S<N> + k];
    for (int j = 0; j < N; j++) {
      c[j] -= a * sB[k * S<N> + j];
    }
  }

  for (int j = 0; j < N; j++) {
    C_ptr[lid * N + j] = c[j];
  }
}

// ============================================================
// tile_gemm_atomic: C -= A·B with atomic updates
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
//
// Buffers:
//   0: tiles_out  — output tile storage (atomic float)
//   1: tiles_in   — input tile storage (float, read-only)
//   2: offsets_c   — per-op element offset to C in tiles_out
//   3: offsets_a   — per-op element offset to A in tiles_in
//   4: offsets_b   — per-op element offset to B in tiles_in
// ============================================================
template <int N>
[[kernel]] void tile_gemm_atomic(
    device mlx_atomic<float>* tiles_out [[buffer(0)]],
    const device float* tiles_in [[buffer(1)]],
    const device int* offsets_c [[buffer(2)]],
    const device int* offsets_a [[buffer(3)]],
    const device int* offsets_b [[buffer(4)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  int c_off = offsets_c[op_idx];
  int a_off = offsets_a[op_idx];
  int b_off = offsets_b[op_idx];

  threadgroup float sA[N * S<N>];
  threadgroup float sB[N * S<N>];

  for (int row = 0; row < N; row++) {
    sA[row * S<N> + lid] = tiles_in[a_off + row * N + lid];
    sB[row * S<N> + lid] = tiles_in[b_off + row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0; k < N; k++) {
    float a = sA[lid * S<N> + k];
    for (int j = 0; j < N; j++) {
      float val = a * sB[k * S<N> + j];
      mlx_atomic_fetch_add_explicit(tiles_out, -val, c_off + lid * N + j);
    }
  }
}

// ============================================================
// tile_geadd: C += A (element-wise addition)
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
// Thread lid handles row lid. No shared memory needed.
//
// Buffers:
//   0: tiles     — contiguous tile storage (read/write)
//   1: offsets_c  — per-op element offset to tile C
//   2: offsets_a  — per-op element offset to tile A
// ============================================================
template <int N>
[[kernel]] void tile_geadd(
    device float* tiles [[buffer(0)]],
    const device int* offsets_c [[buffer(1)]],
    const device int* offsets_a [[buffer(2)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  device float* C_ptr = tiles + offsets_c[op_idx];
  const device float* A_ptr = tiles + offsets_a[op_idx];

  for (int col = 0; col < N; col++) {
    C_ptr[lid * N + col] += A_ptr[lid * N + col];
  }
}

// ============================================================
// Kernel instantiation for N=16 and N=32
// ============================================================

// clang-format off
instantiate_tile_blas_all(16)
instantiate_tile_blas_all(32)
// clang-format on
