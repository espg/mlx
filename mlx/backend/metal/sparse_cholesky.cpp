// Copyright © 2025 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

// GPU numerical factorization for tile-based sparse Cholesky.
// Dispatches tile_blas kernels column-by-column using pre-computed offsets.
//
// Inputs (12 arrays):
//   0: tile_data    – (total_elems,) float32, all tiles packed flat
//   1: potrf_off    – (Nt,) int32, element offset per diagonal tile
//   2: trsm_l_off   – (total_trsm,) int32
//   3: trsm_b_off   – (total_trsm,) int32
//   4: trsm_col_ptr – (Nt+1,) int32
//   5: syrk_c_off   – (total_syrk,) int32
//   6: syrk_a_off   – (total_syrk,) int32
//   7: syrk_col_ptr – (Nt+1,) int32
//   8: gemm_c_off   – (total_gemm,) int32
//   9: gemm_a_off   – (total_gemm,) int32
//  10: gemm_b_off   – (total_gemm,) int32
//  11: gemm_col_ptr – (Nt+1,) int32
//
// Output: factored tile_data (same shape as input 0)

void SparseCholeskyFactor::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  if (inputs[0].dtype() != float32) {
    throw std::runtime_error(
        "[SparseCholeskyFactor::eval_gpu] Only float32 supported.");
  }

  int n = tile_size_;
  if (n != 16 && n != 32 && n != 64) {
    throw std::runtime_error(
        "[SparseCholeskyFactor::eval_gpu] tile_size must be 16, 32, or 64.");
  }

  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& enc = metal::get_command_encoder(s);

  // Copy tile_data into output (we operate in-place)
  copy_gpu(inputs[0], out, CopyType::General, s);

  // Read task schedule arrays (they are on GPU after eval)
  const auto& potrf_off = inputs[1];
  const auto& trsm_l_off = inputs[2];
  const auto& trsm_b_off = inputs[3];
  const auto& trsm_cptr = inputs[4];
  const auto& syrk_c_off = inputs[5];
  const auto& syrk_a_off = inputs[6];
  const auto& syrk_cptr = inputs[7];
  const auto& gemm_c_off = inputs[8];
  const auto& gemm_a_off = inputs[9];
  const auto& gemm_b_off = inputs[10];
  const auto& gemm_cptr = inputs[11];

  int Nt = potrf_off.shape(0); // number of tile columns

  // Read col_ptr arrays from CPU (they're small, already evaluated)
  // Since these are on GPU with shared storage, we can read directly
  const int32_t* trsm_cp = trsm_cptr.data<int32_t>();
  const int32_t* syrk_cp = syrk_cptr.data<int32_t>();
  const int32_t* gemm_cp = gemm_cptr.data<int32_t>();

  // Get kernels
  std::string ns = std::to_string(n);
  auto k_potrf = d.get_kernel("tile_potrf_float32_" + ns);
  auto k_syrk = d.get_kernel("tile_syrk_float32_" + ns);
  auto k_trsm_r = d.get_kernel("tile_trsm_right_lower_float32_" + ns);
  auto k_gemm = d.get_kernel("tile_gemm_float32_" + ns);

  MTL::Size group(n, 1, 1);

  for (int j = 0; j < Nt; j++) {
    // --- potrf: factor diagonal tile ---
    enc.set_compute_pipeline_state(k_potrf);
    enc.set_output_array(out, 0);
    enc.set_input_array(potrf_off, 1, j * sizeof(int32_t));
    enc.dispatch_threadgroups(MTL::Size(1, 1, 1), group);

    // --- trsm: solve off-diagonal tiles in column j ---
    int trsm_start = trsm_cp[j];
    int trsm_end = trsm_cp[j + 1];
    int trsm_count = trsm_end - trsm_start;
    if (trsm_count > 0) {
      enc.set_compute_pipeline_state(k_trsm_r);
      enc.set_output_array(out, 0);
      enc.set_input_array(trsm_l_off, 1, trsm_start * sizeof(int32_t));
      enc.set_input_array(trsm_b_off, 2, trsm_start * sizeof(int32_t));
      enc.dispatch_threadgroups(MTL::Size(trsm_count, 1, 1), group);
    }

    // --- syrk: diagonal updates ---
    int syrk_start = syrk_cp[j];
    int syrk_end = syrk_cp[j + 1];
    int syrk_count = syrk_end - syrk_start;
    if (syrk_count > 0) {
      enc.set_compute_pipeline_state(k_syrk);
      enc.set_output_array(out, 0);
      enc.set_input_array(syrk_c_off, 1, syrk_start * sizeof(int32_t));
      enc.set_input_array(syrk_a_off, 2, syrk_start * sizeof(int32_t));
      enc.dispatch_threadgroups(MTL::Size(syrk_count, 1, 1), group);
    }

    // --- gemm: off-diagonal updates ---
    int gemm_start = gemm_cp[j];
    int gemm_end = gemm_cp[j + 1];
    int gemm_count = gemm_end - gemm_start;
    if (gemm_count > 0) {
      enc.set_compute_pipeline_state(k_gemm);
      enc.set_output_array(out, 0);
      enc.set_input_array(gemm_c_off, 1, gemm_start * sizeof(int32_t));
      enc.set_input_array(gemm_a_off, 2, gemm_start * sizeof(int32_t));
      enc.set_input_array(gemm_b_off, 3, gemm_start * sizeof(int32_t));
      enc.dispatch_threadgroups(MTL::Size(gemm_count, 1, 1), group);
    }
  }
}

} // namespace mlx::core
