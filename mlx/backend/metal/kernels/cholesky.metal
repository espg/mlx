// Copyright © 2025 Apple Inc.

#include <metal_math>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/defines.h"

using namespace metal;

// ============================================================
// Kernel 1: Small matrix Cholesky (N <= BLOCK_SIZE)
// One threadgroup per batch matrix, entire matrix in shared memory.
// ============================================================
template <int BLOCK_SIZE>
[[kernel]] void cholesky_small(
    device float* out [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant bool& upper [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  device float* mat = out + uint64_t(batch_idx) * N * N;

  threadgroup float L[BLOCK_SIZE * BLOCK_SIZE];

  // Load matrix into shared memory (row-major, BLOCK_SIZE stride)
  for (uint idx = lid; idx < uint(N * N); idx += tg_size) {
    uint row = idx / N;
    uint col = idx % N;
    L[row * BLOCK_SIZE + col] = mat[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Column-by-column Cholesky: L * L^T = A
  for (int j = 0; j < N; j++) {
    // Thread 0 computes diagonal element
    if (lid == 0) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        float v = L[j * BLOCK_SIZE + k];
        sum += v * v;
      }
      float d = L[j * BLOCK_SIZE + j] - sum;
      L[j * BLOCK_SIZE + j] = (d > 0.0f) ? metal::sqrt(d) : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float diag_val = L[j * BLOCK_SIZE + j];

    // All threads compute off-diagonal elements in column j
    for (int i = j + 1 + int(lid); i < N; i += int(tg_size)) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) {
        sum += L[i * BLOCK_SIZE + k] * L[j * BLOCK_SIZE + k];
      }
      L[i * BLOCK_SIZE + j] = (L[i * BLOCK_SIZE + j] - sum) / diag_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write back: output L (lower) or U = L^T (upper), zeroing other triangle
  for (uint idx = lid; idx < uint(N * N); idx += tg_size) {
    uint row = idx / N;
    uint col = idx % N;
    if (upper) {
      mat[idx] = (col >= row) ? L[col * BLOCK_SIZE + row] : 0.0f;
    } else {
      mat[idx] = (row >= col) ? L[row * BLOCK_SIZE + col] : 0.0f;
    }
  }
}

// ============================================================
// Kernel 2: Factor diagonal block + solve panel
// One threadgroup per batch matrix. Used in the blocked algorithm.
// Supports block sizes up to 64 (64*64*4 = 16KB shared memory).
// ============================================================
template <int MAX_B>
[[kernel]] void cholesky_diag_panel(
    device float* out [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant int& k [[buffer(2)]],
    constant int& B [[buffer(3)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  device float* mat = out + uint64_t(batch_idx) * N * N;

  int block_start = k * B;
  int block_end = min(block_start + B, N);
  int bs = block_end - block_start;

  threadgroup float diag[MAX_B * MAX_B];

  // Load diagonal block into shared memory
  for (int idx = int(lid); idx < bs * bs; idx += int(tg_size)) {
    int r = idx / bs;
    int c = idx % bs;
    diag[r * MAX_B + c] = mat[(block_start + r) * N + (block_start + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Factor diagonal block (column-by-column Cholesky)
  for (int j = 0; j < bs; j++) {
    if (lid == 0) {
      float sum = 0.0f;
      for (int p = 0; p < j; p++) {
        float v = diag[j * MAX_B + p];
        sum += v * v;
      }
      float d = diag[j * MAX_B + j] - sum;
      diag[j * MAX_B + j] = (d > 0.0f) ? metal::sqrt(d) : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float dval = diag[j * MAX_B + j];
    for (int i = j + 1 + int(lid); i < bs; i += int(tg_size)) {
      float sum = 0.0f;
      for (int p = 0; p < j; p++) {
        sum += diag[i * MAX_B + p] * diag[j * MAX_B + p];
      }
      diag[i * MAX_B + j] = (diag[i * MAX_B + j] - sum) / dval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write factored diagonal block back (lower triangle only)
  for (int idx = int(lid); idx < bs * bs; idx += int(tg_size)) {
    int r = idx / bs;
    int c = idx % bs;
    float val = (r >= c) ? diag[r * MAX_B + c] : 0.0f;
    mat[(block_start + r) * N + (block_start + c)] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Solve panel below diagonal block using forward substitution
  int panel_rows = N - block_end;
  if (panel_rows <= 0) {
    return;
  }

  for (int j = 0; j < bs; j++) {
    float L_jj = diag[j * MAX_B + j];

    for (int ri = int(lid); ri < panel_rows; ri += int(tg_size)) {
      int gi = block_end + ri;
      int gj = block_start + j;

      float sum = 0.0f;
      for (int p = 0; p < j; p++) {
        sum += mat[gi * N + (block_start + p)] * diag[j * MAX_B + p];
      }
      mat[gi * N + gj] = (mat[gi * N + gj] - sum) / L_jj;
    }
    // Device barrier: panel column j must be visible before column j+1
    threadgroup_barrier(mem_flags::mem_device);
  }
}

// ============================================================
// Kernel 3: Trailing SYRK-like update
// One threadgroup per (bi, bj) tile pair per batch matrix.
// Loads one panel tile into shared memory, streams the other
// from device memory (to stay within 32KB threadgroup limit).
// ============================================================
template <int MAX_B>
[[kernel]] void cholesky_update(
    device float* out [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant int& k [[buffer(2)]],
    constant int& B [[buffer(3)]],
    constant int& num_remaining [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 tg_size3 [[threads_per_threadgroup]]) {
  uint tile_idx = tgid.x;
  uint batch_idx = tgid.y;
  uint lid = lid3.x;
  uint tg_size = tg_size3.x;
  (void)num_remaining;
  device float* mat = out + uint64_t(batch_idx) * N * N;

  // Map linear tile index to (bi_local, bj_local) in the lower triangle
  int bi_local =
      int(floor((metal::sqrt(8.0f * float(tile_idx) + 1.0f) - 1.0f) / 2.0f));
  int bj_local = int(tile_idx) - bi_local * (bi_local + 1) / 2;
  if (bj_local > bi_local) {
    bi_local++;
    bj_local = int(tile_idx) - bi_local * (bi_local + 1) / 2;
  }

  int bi = bi_local + k + 1;
  int bj = bj_local + k + 1;

  int ri_start = bi * B;
  int rj_start = bj * B;
  int ck_start = k * B;
  int bs_k = min(B, N - ck_start);
  int bs_i = min(B, N - ri_start);
  int bs_j = min(B, N - rj_start);

  // Load tile_j into shared memory (reused across all output rows)
  threadgroup float tile_j[MAX_B * MAX_B];

  for (int idx = int(lid); idx < bs_j * bs_k; idx += int(tg_size)) {
    int r = idx / bs_k;
    int c = idx % bs_k;
    tile_j[r * MAX_B + c] = mat[(rj_start + r) * N + (ck_start + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Compute: A[ri+r, rj+c] -= L[ri+r, ck:ck+bs_k] . L[rj+c, ck:ck+bs_k]
  // tile_j is in shared memory; tile_i rows are read from device memory
  for (int idx = int(lid); idx < bs_i * bs_j; idx += int(tg_size)) {
    int r = idx / bs_j;
    int c = idx % bs_j;

    // For diagonal tile, only update lower triangle
    if (bi == bj && r < c) {
      continue;
    }

    float sum = 0.0f;
    for (int p = 0; p < bs_k; p++) {
      // tile_i[r,p] read from device memory
      sum += mat[(ri_start + r) * N + (ck_start + p)] * tile_j[c * MAX_B + p];
    }
    mat[(ri_start + r) * N + (rj_start + c)] -= sum;
  }
}

// ============================================================
// Kernel 4: Finalize -- zero upper triangle (lower output)
// or transpose lower to upper (upper output).
// One threadgroup per batch matrix.
// ============================================================
[[kernel]] void cholesky_finalize_float32(
    device float* out [[buffer(0)]],
    constant int& N [[buffer(1)]],
    constant bool& upper [[buffer(2)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  device float* mat = out + uint64_t(batch_idx) * N * N;

  if (upper) {
    // Copy lower triangle to upper (transposed) and zero lower
    for (uint idx = lid; idx < uint(N * N); idx += tg_size) {
      uint row = idx / N;
      uint col = idx % N;
      if (row > col) {
        mat[col * N + row] = mat[idx];
        mat[idx] = 0.0f;
      }
    }
  } else {
    // Zero the upper triangle
    for (uint idx = lid; idx < uint(N * N); idx += tg_size) {
      uint row = idx / N;
      uint col = idx % N;
      if (col > row) {
        mat[idx] = 0.0f;
      }
    }
  }
}

// clang-format off
instantiate_kernel("cholesky_small_float32_32", cholesky_small, 32)
instantiate_kernel("cholesky_small_float32_64", cholesky_small, 64)
instantiate_kernel("cholesky_diag_panel_float32_32", cholesky_diag_panel, 32)
instantiate_kernel("cholesky_diag_panel_float32_64", cholesky_diag_panel, 64)
instantiate_kernel("cholesky_update_float32_32", cholesky_update, 32)
instantiate_kernel("cholesky_update_float32_64", cholesky_update, 64)
    // clang-format on
