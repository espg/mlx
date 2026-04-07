// Copyright © 2025 Apple Inc.
//
// Block tridiagonal Cholesky via nested dissection.
//
// Elimination tree processes blocks in O(log N) levels.
// At each level, all blocks are independent → batched parallel dispatch.
// Total dispatches: ~6·log₂(N) vs N for sequential.
//
// Key insight for the trsm convention:
//   Right connection E_right stored as E_orig (sub-diagonal).
//   Left connection E_left stored as E_orig^T (transposed).
//   Both use RIGHT trsm: R = E_stored · L^{-T}.
//   Then syrk(R) = R·R^T gives the correct Schur complement for BOTH neighbors.
//   Fill-in uses gemm_bt: C -= R_right · R_left^T.

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

#include <cmath>
#include <cstring>

namespace mlx::core {

static int compute_n_e_total(int N) {
  int total = 0;
  for (int s = 1; s < N; s *= 2) {
    total += (N + s - 1) / s - 1;
  }
  return total;
}

void BlockTridiagCholesky::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const auto& D_in = inputs[0];
  const auto& E_in = inputs[1];

  if (D_in.dtype() != float32)
    throw std::runtime_error(
        "[BlockTridiagCholesky::eval_gpu] Only float32 supported.");

  int n = D_in.shape(-1);
  int N = D_in.shape(-3);
  int nn = n * n;

  if (n != 16 && n != 32 && n != 64)
    throw std::runtime_error(
        "[BlockTridiagCholesky::eval_gpu] Tile size n must be 16, 32, or 64.");

  size_t batch = D_in.size() / (N * nn);
  int n_E_total = compute_n_e_total(N);

  auto& L_diag = outputs[0];
  auto& L_offdiag = outputs[1];
  L_diag.set_data(allocator::malloc(L_diag.nbytes()));
  L_offdiag.set_data(allocator::malloc(L_offdiag.nbytes()));

  auto& enc = metal::get_command_encoder(s);

  // Combined work buffer: D tiles then E_all tiles
  size_t d_elems = batch * N * nn;
  size_t e_elems = batch * n_E_total * nn;
  size_t total_elems = d_elems + e_elems;

  array work({static_cast<int>(total_elems)}, float32, nullptr, {});
  work.set_data(allocator::malloc(total_elems * sizeof(float)));
  enc.add_temporary(work);

  // Zero the whole buffer (E_all fill-in slots must start at 0)
  {
    array zero_val = array(0.0f);
    fill_gpu(zero_val, work, s);
  }

  // Copy D_in → work[0..d_elems)
  copy_gpu_inplace(D_in, work, {static_cast<int>(d_elems)},
                   D_in.strides(), {1}, 0, 0, CopyType::General, s);

  // Copy E_In into E_all section, TRANSPOSING left-connection tiles.
  // E_all level 0 stores N-1 off-diags connecting consecutive blocks.
  // Off-diag j connects block j to block j+1.
  // At step=1, block (2k+1) uses off-diag 2k as LEFT (needs transpose)
  // and off-diag 2k+1 as RIGHT (no transpose).
  // For generality: at level 0, even-indexed off-diags are LEFT connections
  // and odd-indexed are RIGHT connections.
  // Actually, which tiles are left vs right depends on which block uses them.
  // Off-diag j at level 0: used as RIGHT by block j (connects j→j+1) and
  // as LEFT by block j+1 (connects j+1←j). Since blocks processed at step=1
  // are odd blocks (1,3,5,...), off-diag j is LEFT for block j+1 when j+1 is
  // odd, i.e., j is even. So even j: LEFT (transpose), odd j: RIGHT (no trans).

  // Copy with selective transpose via CPU (shared memory, fast)
  {
    auto E_cont = contiguous_copy_gpu(E_in, s);
    enc.add_temporary(E_cont);
    // We'll do this tile-by-tile. First copy all E's without transpose...
    Strides ss = {(int64_t)((N - 1) * nn), (int64_t)nn, (int64_t)n, 1};
    Strides ds = {(int64_t)(n_E_total * nn), (int64_t)nn, (int64_t)n, 1};
    Shape shape = {static_cast<int>(batch), N - 1, n, n};
    copy_gpu_inplace(E_cont, work, shape, ss, ds, 0, d_elems,
                     CopyType::GeneralGeneral, s);

    // ...then transpose the even-indexed tiles in-place.
    // For even j (left connections), transpose tile j: swap [row,col] → [col,row]
    // We do this by copying with transposed strides.
    for (int j = 0; j < N - 1; j += 2) {
      // Transpose tile j in each batch: read with (n, 1) strides, write with (1, n)
      for (size_t b = 0; b < batch; b++) {
        int64_t tile_offset = d_elems + b * n_E_total * nn + j * nn;
        // Read as (n, n) with strides (n, 1), write as (n, n) with strides (1, n)
        copy_gpu_inplace(work, work, {n, n}, {(int64_t)n, 1}, {1, (int64_t)n},
                         tile_offset, tile_offset, CopyType::GeneralGeneral, s);
      }
    }
  }

  // Level offsets in E_all
  std::vector<int> level_e_offset;
  {
    int offset = 0;
    for (int step = 1; step < N; step *= 2) {
      level_e_offset.push_back(offset);
      offset += (N + step - 1) / step - 1;
    }
    level_e_offset.push_back(offset);
  }

  auto e_tile_idx = [&](int level, int j) -> int {
    return level_e_offset[level] + j;
  };
  auto dw_off = [&](size_t b, int i) -> int32_t {
    return static_cast<int32_t>(b * N * nn + i * nn);
  };
  auto ew_off = [&](size_t b, int t) -> int32_t {
    return static_cast<int32_t>(d_elems + b * n_E_total * nn + t * nn);
  };

  // Collect per-level operations
  struct LevelOps {
    std::vector<int32_t> potrf_d;
    std::vector<int32_t> trsm_l, trsm_b;       // right trsm for BOTH connections
    std::vector<int32_t> syrk_c, syrk_a;        // regular syrk_atomic for BOTH
    std::vector<int32_t> gemm_c, gemm_a, gemm_b; // gemm_bt fill-in
  };

  std::vector<LevelOps> all_levels;
  {
    int level = 0;
    for (int step = 1; step < N; step *= 2, level++) {
      LevelOps ops;
      for (int i = step; i < N; i += 2 * step) {
        int p = i / step;
        bool has_left = (i - step >= 0);
        bool has_right = (i + step < N);
        int e_left = has_left ? e_tile_idx(level, p - 1) : -1;
        int e_right = has_right ? e_tile_idx(level, p) : -1;

        for (size_t b = 0; b < batch; b++) {
          ops.potrf_d.push_back(dw_off(b, i));

          // Right trsm on right connection (E_right stored as-is)
          if (has_right) {
            ops.trsm_l.push_back(dw_off(b, i));
            ops.trsm_b.push_back(ew_off(b, e_right));
          }
          // Right trsm on left connection (E_left stored as E^T)
          if (has_left) {
            ops.trsm_l.push_back(dw_off(b, i));
            ops.trsm_b.push_back(ew_off(b, e_left));
          }

          // syrk_atomic on right neighbor: D[i+step] -= R_right · R_right^T
          if (has_right) {
            ops.syrk_c.push_back(dw_off(b, i + step));
            ops.syrk_a.push_back(ew_off(b, e_right));
          }
          // syrk_atomic on left neighbor: D[i-step] -= R_left · R_left^T
          if (has_left) {
            ops.syrk_c.push_back(dw_off(b, i - step));
            ops.syrk_a.push_back(ew_off(b, e_left));
          }

          // gemm_bt fill-in: E_next -= R_right · R_left^T
          if (has_left && has_right) {
            int next_j = (i - step) / (2 * step);
            int e_next = e_tile_idx(level + 1, next_j);
            ops.gemm_c.push_back(ew_off(b, e_next));
            ops.gemm_a.push_back(ew_off(b, e_right)); // A = R_right
            ops.gemm_b.push_back(ew_off(b, e_left));  // B = R_left (B^T in kernel)
          }
        }
      }
      all_levels.push_back(std::move(ops));
    }
  }
  // Final: block 0
  {
    LevelOps ops;
    for (size_t b = 0; b < batch; b++)
      ops.potrf_d.push_back(dw_off(b, 0));
    all_levels.push_back(std::move(ops));
  }

  auto make_arr = [&](const std::vector<int32_t>& v) -> array {
    if (v.empty()) return array({0}, int32, nullptr, {});
    array a({static_cast<int>(v.size())}, int32, nullptr, {});
    a.set_data(allocator::malloc(v.size() * sizeof(int32_t)));
    std::memcpy(a.data<int32_t>(), v.data(), v.size() * sizeof(int32_t));
    enc.add_temporary(a);
    return a;
  };

  std::string ns = std::to_string(n);
  auto k_potrf = d.get_kernel("tile_potrf_float32_" + ns);
  auto k_trsm = d.get_kernel("tile_trsm_right_lower_float32_" + ns);
  auto k_syrk = d.get_kernel("tile_syrk_atomic_float32_" + ns);
  auto k_gemm_bt = d.get_kernel("tile_gemm_bt_float32_" + ns);

  MTL::Size group(n, 1, 1);

  for (auto& ops : all_levels) {
    if (!ops.potrf_d.empty()) {
      auto a = make_arr(ops.potrf_d);
      enc.set_compute_pipeline_state(k_potrf);
      enc.set_output_array(work, 0);
      enc.set_input_array(a, 1);
      enc.dispatch_threadgroups(MTL::Size(ops.potrf_d.size(), 1, 1), group);
    }
    if (!ops.trsm_l.empty()) {
      auto al = make_arr(ops.trsm_l);
      auto ab = make_arr(ops.trsm_b);
      enc.set_compute_pipeline_state(k_trsm);
      enc.set_output_array(work, 0);
      enc.set_input_array(al, 1);
      enc.set_input_array(ab, 2);
      enc.dispatch_threadgroups(MTL::Size(ops.trsm_l.size(), 1, 1), group);
    }
    if (!ops.syrk_c.empty()) {
      auto ac = make_arr(ops.syrk_c);
      auto aa = make_arr(ops.syrk_a);
      enc.set_compute_pipeline_state(k_syrk);
      enc.set_output_array(work, 0);
      enc.set_input_array(work, 1);
      enc.set_input_array(ac, 2);
      enc.set_input_array(aa, 3);
      enc.dispatch_threadgroups(MTL::Size(ops.syrk_c.size(), 1, 1), group);
    }
    if (!ops.gemm_c.empty()) {
      auto ac = make_arr(ops.gemm_c);
      auto aa = make_arr(ops.gemm_a);
      auto ab = make_arr(ops.gemm_b);
      enc.set_compute_pipeline_state(k_gemm_bt);
      enc.set_output_array(work, 0);
      enc.set_input_array(ac, 1);
      enc.set_input_array(aa, 2);
      enc.set_input_array(ab, 3);
      enc.dispatch_threadgroups(MTL::Size(ops.gemm_c.size(), 1, 1), group);
    }
  }

  // Copy results back
  copy_gpu_inplace(work, L_diag, {static_cast<int>(d_elems)},
                   {1}, {1}, 0, 0, CopyType::Vector, s);
  copy_gpu_inplace(work, L_offdiag, {static_cast<int>(e_elems)},
                   {1}, {1}, static_cast<int>(d_elems), 0, CopyType::Vector, s);
}

} // namespace mlx::core
