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

// Stride for kernels that load two tiles into shared memory.
// At N=64, 2×N×(N+1)×4 = 33,280 > 32,768, so we drop padding.
template <int N>
constexpr constant int S2 = (2 * N * (N + 1) * 4 <= 32768) ? (N + 1) : N;

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
// tile_gemm_bt: C -= A·Bᵀ
//
// Variant of gemm where B is transposed: C[i,j] -= sum_k A[i,k] * B[j,k]
// Used for nested dissection fill-in where both operands come from right trsm.
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
// ============================================================
template <int N>
[[kernel]] void tile_gemm_bt(
    device float* tiles [[buffer(0)]],
    const device int* offsets_c [[buffer(1)]],
    const device int* offsets_a [[buffer(2)]],
    const device int* offsets_b [[buffer(3)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  device float* C_ptr = tiles + offsets_c[op_idx];
  const device float* A_ptr = tiles + offsets_a[op_idx];
  const device float* B_ptr = tiles + offsets_b[op_idx];

  threadgroup float sA[N * S2<N>];
  threadgroup float sB[N * S2<N>];

  for (int row = 0; row < N; row++) {
    sA[row * S2<N> + lid] = A_ptr[row * N + lid];
    sB[row * S2<N> + lid] = B_ptr[row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // C[lid, j] -= sum_k A[lid, k] * B[j, k]  (B transposed)
  float c[N];
  for (int j = 0; j < N; j++) {
    c[j] = C_ptr[lid * N + j];
  }

  for (int j = 0; j < N; j++) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += sA[lid * S2<N> + k] * sB[j * S2<N> + k];
    }
    c[j] -= sum;
  }

  for (int j = 0; j < N; j++) {
    C_ptr[lid * N + j] = c[j];
  }
}

// ============================================================
// tile_syrk_t_atomic: C -= Aᵀ·A with atomic updates (lower triangle)
//
// Transpose variant of syrk_atomic. Used for left-neighbor diagonal
// updates in nested dissection where the left trsm gives Q = L⁻¹·E
// and we need C -= Qᵀ·Q = Eᵀ·D⁻¹·E.
//
// C[i,j] -= sum_k A[k,i] * A[k,j]  for i >= j
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
// ============================================================
template <int N>
[[kernel]] void tile_syrk_t_atomic(
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

  // C[lid, j] -= sum_k A[k, lid] * A[k, j] for j = 0..lid
  // Column lid of A = sA[k * S + lid] for k = 0..N-1
  for (int j = 0; j <= int(lid); j++) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += sA[k * S<N> + lid] * sA[k * S<N> + j];
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

  threadgroup float sA[N * S2<N>];
  threadgroup float sB[N * S2<N>];

  // Coalesced load of A and B
  for (int row = 0; row < N; row++) {
    sA[row * S2<N> + lid] = A_ptr[row * N + lid];
    sB[row * S2<N> + lid] = B_ptr[row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate C[lid,:] -= A[lid,:] · B
  // Restructured as outer product accumulation for better data reuse
  float c[N];
  for (int j = 0; j < N; j++) {
    c[j] = C_ptr[lid * N + j];
  }

  for (int k = 0; k < N; k++) {
    float a = sA[lid * S2<N> + k];
    for (int j = 0; j < N; j++) {
      c[j] -= a * sB[k * S2<N> + j];
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

  threadgroup float sA[N * S2<N>];
  threadgroup float sB[N * S2<N>];

  for (int row = 0; row < N; row++) {
    sA[row * S2<N> + lid] = tiles_in[a_off + row * N + lid];
    sB[row * S2<N> + lid] = tiles_in[b_off + row * N + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0; k < N; k++) {
    float a = sA[lid * S2<N> + k];
    for (int j = 0; j < N; j++) {
      float val = a * sB[k * S2<N> + j];
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
// block_tridiag_step: Fused syrk + potrf + trsm for one block
// column of a block tridiagonal Cholesky factorization.
//
// Reduces dispatch count from 3N to N by keeping tiles in shared
// memory across the three operations.
//
// Grid: (num_ops, 1, 1), Threadgroup: (N, 1, 1)
// Each threadgroup processes one (batch, block) pair.
//
// Buffers:
//   0: tiles      — combined work buffer (D then E tiles)
//   1: d_offsets   — per-op element offset to D[i] tile
//   2: eprev_offsets — per-op element offset to E[i-1] tile (-1 = skip syrk)
//   3: ecurr_offsets — per-op element offset to E[i] tile (-1 = skip trsm)
// ============================================================
template <int N>
[[kernel]] void block_tridiag_step(
    device float* tiles [[buffer(0)]],
    const device int* d_offsets [[buffer(1)]],
    const device int* eprev_offsets [[buffer(2)]],
    const device int* ecurr_offsets [[buffer(3)]],
    uint op_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  constexpr int ST = S2<N>;  // stride that fits 2 tiles in 32 KB

  int d_off = d_offsets[op_idx];
  int ep_off = eprev_offsets[op_idx];
  int ec_off = ecurr_offsets[op_idx];

  device float* D = tiles + d_off;

  threadgroup float sD[N * ST];  // D[i] tile (in-place syrk then potrf)
  threadgroup float sE[N * ST];  // reused for E[i-1] then E[i]

  // Load D[i]
  for (int row = 0; row < N; row++) {
    sD[row * ST + lid] = D[row * N + lid];
  }

  // ---- Step 1: syrk  D[i] -= E[i-1] · E[i-1]^T (if i > 0) ----
  if (ep_off >= 0) {
    const device float* Ep = tiles + ep_off;
    for (int row = 0; row < N; row++) {
      sE[row * ST + lid] = Ep[row * N + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float a_row[N];
    for (int k = 0; k < N; k++) {
      a_row[k] = sE[lid * ST + k];
    }
    for (int j = 0; j <= int(lid); j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) {
        sum += a_row[k] * sE[j * ST + k];
      }
      sD[lid * ST + j] -= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- Step 2: potrf  L_diag[i] = chol(D[i]) ----
  for (int j = 0; j < N; j++) {
    if (lid == 0) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        float v = sD[j * ST + k];
        sum += v * v;
      }
      float d = sD[j * ST + j] - sum;
      sD[j * ST + j] = (d > 0.0f) ? metal::sqrt(d) : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float diag = sD[j * ST + j];
    for (int i = j + 1 + int(lid); i < N; i += N) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        sum += sD[i * ST + k] * sD[j * ST + k];
      }
      sD[i * ST + j] = (sD[i * ST + j] - sum) / diag;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write factored D[i] back (lower triangle, zero upper)
  for (int row = 0; row < N; row++) {
    D[row * N + lid] = (row >= int(lid)) ? sD[row * ST + lid] : 0.0f;
  }

  // ---- Step 3: trsm  E[i] = E[i] · L_diag[i]^{-T} (if i < N-1) ----
  if (ec_off >= 0) {
    device float* Ec = tiles + ec_off;
    // Load E[i] row by row into registers (each thread handles one row)
    float b[N];
    for (int j = 0; j < N; j++) {
      b[j] = Ec[lid * N + j];
    }

    // Solve X · L^T = B  (L is in sD, lower triangular)
    // Same as: L · X^T = B^T → forward substitution on rows
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        sum += sD[j * ST + k] * b[k];
      }
      b[j] = (b[j] - sum) / sD[j * ST + j];
    }

    // Write back
    for (int j = 0; j < N; j++) {
      Ec[lid * N + j] = b[j];
    }
  }
}

// ============================================================
// nd_level_step: Fused nested dissection level kernel.
//
// One threadgroup per (block, batch) pair. Each threadgroup does:
//   1. potrf on D[block_i]
//   2. right trsm on E_right (if exists)
//   3. left trsm on E_left (if exists)
//   4. syrk on D[right_neighbor] via device memory (non-atomic, own row)
//   5. syrk_t on D[left_neighbor] via device memory
//   6. gemm_bt fill-in E_next (if both neighbors exist)
//
// No offset arrays needed — offsets computed from constants.
//
// Grid: (num_blocks_at_level * batch, 1, 1)
// Threadgroup: (N, 1, 1)
//
// Buffers:
//   0: d_tiles   — D tile storage (batch * N_blocks * N * N)
//   1: e_tiles   — E_all tile storage (batch * n_E_total * N * N)
//
// Constants (set_bytes):
//   2: step        — current level step (1, 2, 4, ...)
//   3: N_blocks    — total number of diagonal blocks
//   4: n_E_total   — total E_all tiles per batch
//   5: num_at_level— number of blocks processed at this level
//   6: level_e_off — E_all tile index offset for this level
//   7: next_e_off  — E_all tile index offset for next level (fill-in dest)
// ============================================================
template <int N>
[[kernel]] void nd_level_step(
    device float* d_tiles [[buffer(0)]],
    device float* e_tiles [[buffer(1)]],
    constant int& step [[buffer(2)]],
    constant int& N_blocks [[buffer(3)]],
    constant int& n_E_total [[buffer(4)]],
    constant int& num_at_level [[buffer(5)]],
    constant int& level_e_off [[buffer(6)]],
    constant int& next_e_off [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {

  constexpr int nn = N * N;
  constexpr int ST = S2<N>;

  // Decode (block_index_within_level, batch_index) from tgid
  int block_local = tgid / 1;  // will be tgid % num_at_level after batch split
  int batch_idx = tgid / num_at_level;
  int local_idx = tgid % num_at_level;

  // The i-th block at this level is at position: step + local_idx * 2 * step
  int block_i = step + local_idx * 2 * step;
  if (block_i >= N_blocks) return;

  int pos = block_i / step; // position in remaining set
  bool has_left = (block_i - step >= 0);
  bool has_right = (block_i + step < N_blocks);

  // Tile pointers
  device float* Di = d_tiles + (batch_idx * N_blocks + block_i) * nn;

  int el_tile = has_left ? (level_e_off + pos - 1) : -1;
  int er_tile = has_right ? (level_e_off + pos) : -1;

  device float* El = has_left ?
      (e_tiles + (batch_idx * n_E_total + el_tile) * nn) : nullptr;
  device float* Er = has_right ?
      (e_tiles + (batch_idx * n_E_total + er_tile) * nn) : nullptr;

  threadgroup float sD[N * ST];
  threadgroup float sE[N * ST]; // reused for left then right

  // ---- Load D[i] ----
  for (int row = 0; row < N; row++)
    sD[row * ST + lid] = Di[row * N + lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- potrf ----
  for (int j = 0; j < N; j++) {
    if (lid == 0) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        float v = sD[j * ST + k]; sum += v * v;
      }
      float d = sD[j * ST + j] - sum;
      sD[j * ST + j] = (d > 0.0f) ? metal::sqrt(d) : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float diag = sD[j * ST + j];
    for (int i2 = j + 1 + int(lid); i2 < N; i2 += N) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++)
        sum += sD[i2 * ST + k] * sD[j * ST + k];
      sD[i2 * ST + j] = (sD[i2 * ST + j] - sum) / diag;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write D[i] back (lower triangle)
  for (int row = 0; row < N; row++)
    Di[row * N + lid] = (row >= int(lid)) ? sD[row * ST + lid] : 0.0f;
  threadgroup_barrier(mem_flags::mem_device);

  // ---- Right trsm: Er = Er · L^{-T} ----
  // Thread lid handles row lid of Er
  if (has_right) {
    float b[N];
    for (int j = 0; j < N; j++) b[j] = Er[lid * N + j];
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) sum += sD[j * ST + k] * b[k];
      b[j] = (b[j] - sum) / sD[j * ST + j];
    }
    for (int j = 0; j < N; j++) Er[lid * N + j] = b[j];
  }

  // ---- Left trsm: El(stored as E^T) = El · L^{-T} ----
  if (has_left) {
    float b[N];
    for (int j = 0; j < N; j++) b[j] = El[lid * N + j];
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) sum += sD[j * ST + k] * b[k];
      b[j] = (b[j] - sum) / sD[j * ST + j];
    }
    for (int j = 0; j < N; j++) El[lid * N + j] = b[j];
  }
  threadgroup_barrier(mem_flags::mem_device);

  // ---- syrk on right neighbor: D[i+step] -= Er · Er^T ----
  if (has_right) {
    device float* Dn = d_tiles + (batch_idx * N_blocks + block_i + step) * nn;
    // Load Er into shared memory for reuse
    for (int row = 0; row < N; row++)
      sE[row * ST + lid] = Er[row * N + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float a_row[N];
    for (int k = 0; k < N; k++) a_row[k] = sE[lid * ST + k];
    for (int j = 0; j <= int(lid); j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) sum += a_row[k] * sE[j * ST + k];
      // Use atomic to handle multiple blocks updating same neighbor
      device atomic<float>* dst = reinterpret_cast<device atomic<float>*>(Dn);
      atomic_fetch_add_explicit(&dst[lid * N + j], -sum, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
  }

  // ---- syrk_t on left neighbor: D[i-step] -= El^T · El ----
  if (has_left) {
    device float* Dn = d_tiles + (batch_idx * N_blocks + block_i - step) * nn;
    for (int row = 0; row < N; row++)
      sE[row * ST + lid] = El[row * N + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // C[lid, j] -= sum_k sE[k, lid] * sE[k, j]  (column-dot)
    for (int j = 0; j <= int(lid); j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++)
        sum += sE[k * ST + lid] * sE[k * ST + j];
      device atomic<float>* dst = reinterpret_cast<device atomic<float>*>(Dn);
      atomic_fetch_add_explicit(&dst[lid * N + j], -sum, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
  }

  // ---- gemm_bt fill-in: E_next -= Er · El^T ----
  if (has_left && has_right) {
    int next_j = (block_i - step) / (2 * step);
    int en_tile = next_e_off + next_j;
    device float* En = e_tiles + (batch_idx * n_E_total + en_tile) * nn;

    // Er is already in Er pointer, El in El pointer.
    // Load Er into sE for row access by thread lid.
    for (int row = 0; row < N; row++)
      sE[row * ST + lid] = Er[row * N + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // C[lid, j] -= sum_k Er[lid, k] * El[j, k]
    // El is in device memory, read row j.
    float c[N];
    for (int j = 0; j < N; j++) c[j] = En[lid * N + j];

    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++)
        sum += sE[lid * ST + k] * El[j * N + k];
      c[j] -= sum;
    }
    for (int j = 0; j < N; j++) En[lid * N + j] = c[j];
    threadgroup_barrier(mem_flags::mem_device);

    // Transpose E_next if it will be a left connection at next level
    if (next_j % 2 == 0) {
      // In-place transpose using shared memory
      for (int row = 0; row < N; row++)
        sE[row * ST + lid] = En[row * N + lid];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int row = 0; row < N; row++)
        En[row * N + lid] = sE[lid * ST + row];
    }
  }
}

// ============================================================
// nd_final_potrf: Factor the root block (block 0) after all levels.
//
// Grid: (batch, 1, 1), Threadgroup: (N, 1, 1)
// ============================================================
template <int N>
[[kernel]] void nd_final_potrf(
    device float* d_tiles [[buffer(0)]],
    constant int& N_blocks [[buffer(1)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  constexpr int nn = N * N;
  constexpr int ST = S<N>;
  device float* D0 = d_tiles + batch_idx * N_blocks * nn;

  threadgroup float sD[N * ST];
  for (int row = 0; row < N; row++)
    sD[row * ST + lid] = D0[row * N + lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j = 0; j < N; j++) {
    if (lid == 0) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        float v = sD[j * ST + k]; sum += v * v;
      }
      float d = sD[j * ST + j] - sum;
      sD[j * ST + j] = (d > 0.0f) ? metal::sqrt(d) : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float diag = sD[j * ST + j];
    for (int i2 = j + 1 + int(lid); i2 < N; i2 += N) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++)
        sum += sD[i2 * ST + k] * sD[j * ST + k];
      sD[i2 * ST + j] = (sD[i2 * ST + j] - sum) / diag;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int row = 0; row < N; row++)
    D0[row * N + lid] = (row >= int(lid)) ? sD[row * ST + lid] : 0.0f;
}

// ============================================================
// Kernel instantiation for N=16, 32, 64
// ============================================================

// clang-format off
instantiate_tile_blas_all(16)
instantiate_tile_blas_all(32)
instantiate_tile_blas_all(64)
instantiate_kernel("tile_gemm_bt_float32_16", tile_gemm_bt, 16)
instantiate_kernel("tile_gemm_bt_float32_32", tile_gemm_bt, 32)
instantiate_kernel("tile_gemm_bt_float32_64", tile_gemm_bt, 64)
instantiate_kernel("block_tridiag_step_float32_16", block_tridiag_step, 16)
instantiate_kernel("block_tridiag_step_float32_32", block_tridiag_step, 32)
instantiate_kernel("block_tridiag_step_float32_64", block_tridiag_step, 64)
instantiate_kernel("nd_level_step_float32_16", nd_level_step, 16)
instantiate_kernel("nd_level_step_float32_32", nd_level_step, 32)
instantiate_kernel("nd_level_step_float32_64", nd_level_step, 64)
instantiate_kernel("nd_final_potrf_float32_16", nd_final_potrf, 16)
instantiate_kernel("nd_final_potrf_float32_32", nd_final_potrf, 32)
instantiate_kernel("nd_final_potrf_float32_64", nd_final_potrf, 64)
// clang-format on
