// Copyright © 2025 Apple Inc.
//
// Block tridiagonal Cholesky via nested dissection.
// Uses fused nd_level_step kernel — ONE dispatch per level.
// Total dispatches: log₂(N) + 1.

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

#include <cmath>

namespace mlx::core {

static int compute_n_e_total(int N) {
  int total = 0;
  for (int s = 1; s < N; s *= 2)
    total += (N + s - 1) / s - 1;
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

  // D_work: copy of D, modified in-place
  copy_gpu(D_in, L_diag, CopyType::General, s);

  // E_work: zero-initialized, first N-1 tiles per batch filled from E_in
  // with even-indexed tiles transposed (left-connection convention).
  {
    array zero_val = array(0.0f);
    fill_gpu(zero_val, L_offdiag, s);
  }
  {
    auto E_cont = contiguous_copy_gpu(E_in, s);
    enc.add_temporary(E_cont);

    // Alloc temp tile for safe transpose
    array tmp({nn}, float32, nullptr, {});
    tmp.set_data(allocator::malloc(nn * sizeof(float)));
    enc.add_temporary(tmp);

    for (size_t b = 0; b < batch; b++) {
      for (int j = 0; j < N - 1; j++) {
        int64_t src_off = b * (N - 1) * nn + j * nn;
        int64_t dst_off = b * n_E_total * nn + j * nn;
        if (j % 2 == 0) {
          // Left connection: transpose via tmp
          copy_gpu_inplace(E_cont, tmp, {n, n},
                           {(int64_t)n, 1}, {1, (int64_t)n},
                           src_off, 0, CopyType::GeneralGeneral, s);
          copy_gpu_inplace(tmp, L_offdiag, {nn}, {1}, {1},
                           0, dst_off, CopyType::Vector, s);
        } else {
          // Right connection: straight copy
          copy_gpu_inplace(E_cont, L_offdiag, {nn}, {1}, {1},
                           src_off, dst_off, CopyType::Vector, s);
        }
      }
    }
  }

  // Compute level E offsets
  std::vector<int> level_e_off;
  {
    int off = 0;
    for (int step = 1; step < N; step *= 2) {
      level_e_off.push_back(off);
      off += (N + step - 1) / step - 1;
    }
    level_e_off.push_back(off); // sentinel for next_e_off
  }

  // Get fused kernels
  std::string ns = std::to_string(n);
  auto k_nd = d.get_kernel("nd_level_step_float32_" + ns);
  auto k_final = d.get_kernel("nd_final_potrf_float32_" + ns);

  MTL::Size group(n, 1, 1);

  // --- Dispatch one kernel per level ---
  int level = 0;
  for (int step = 1; step < N; step *= 2, level++) {
    // Number of blocks at this level
    int num_at_level = 0;
    for (int i = step; i < N; i += 2 * step) num_at_level++;

    int next_off = (level + 1 < (int)level_e_off.size() - 1)
                       ? level_e_off[level + 1]
                       : level_e_off.back();

    enc.set_compute_pipeline_state(k_nd);
    enc.set_output_array(L_diag, 0);   // d_tiles
    enc.set_output_array(L_offdiag, 1); // e_tiles
    enc.set_bytes(step, 2);
    enc.set_bytes(N, 3);
    enc.set_bytes(n_E_total, 4);
    enc.set_bytes(num_at_level, 5);
    enc.set_bytes(level_e_off[level], 6);
    enc.set_bytes(next_off, 7);

    MTL::Size grid(num_at_level * static_cast<int>(batch), 1, 1);
    enc.dispatch_threadgroups(grid, group);
  }

  // Final: factor block 0
  enc.set_compute_pipeline_state(k_final);
  enc.set_output_array(L_diag, 0);
  enc.set_bytes(N, 1);
  enc.dispatch_threadgroups(MTL::Size(batch, 1, 1), group);
}

} // namespace mlx::core
