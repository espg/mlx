// Copyright © 2025 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void BlockTridiagCholesky::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const auto& D_in = inputs[0]; // (..., N, n, n)
  const auto& E_in = inputs[1]; // (..., N-1, n, n)

  if (D_in.dtype() != float32) {
    throw std::runtime_error(
        "[BlockTridiagCholesky::eval_gpu] Only float32 is supported on GPU.");
  }

  int n = D_in.shape(-1);
  int N = D_in.shape(-3);
  int nn = n * n;
  int total_tiles = 2 * N - 1;

  if (n != 16 && n != 32) {
    throw std::runtime_error(
        "[BlockTridiagCholesky::eval_gpu] Tile size n must be 16 or 32.");
  }

  size_t batch = D_in.size() / (N * nn);

  // Allocate outputs
  auto& L_diag = outputs[0];
  auto& L_offdiag = outputs[1];
  L_diag.set_data(allocator::malloc(L_diag.nbytes()));
  L_offdiag.set_data(allocator::malloc(L_offdiag.nbytes()));

  auto& enc = metal::get_command_encoder(s);

  // Allocate combined work buffer: [D tiles | E tiles] per batch
  size_t work_elems = batch * total_tiles * nn;
  array work({static_cast<int>(work_elems)}, float32, nullptr, {});
  work.set_data(allocator::malloc(work_elems * sizeof(float)));
  enc.add_temporary(work);

  // Copy D_in → work[0..N*nn-1] per batch
  {
    auto D_cont = contiguous_copy_gpu(D_in, s);
    enc.add_temporary(D_cont);
    Shape shape = {static_cast<int>(batch), N, n, n};
    Strides ss = {(int64_t)(N * nn), (int64_t)nn, (int64_t)n, 1};
    Strides ds = {(int64_t)(total_tiles * nn), (int64_t)nn, (int64_t)n, 1};
    copy_gpu_inplace(D_cont, work, shape, ss, ds, 0, 0,
                     CopyType::GeneralGeneral, s);
  }
  // Copy E_in → work[N*nn..] per batch
  {
    auto E_cont = contiguous_copy_gpu(E_in, s);
    enc.add_temporary(E_cont);
    Shape shape = {static_cast<int>(batch), N - 1, n, n};
    Strides ss = {(int64_t)((N - 1) * nn), (int64_t)nn, (int64_t)n, 1};
    Strides ds = {(int64_t)(total_tiles * nn), (int64_t)nn, (int64_t)n, 1};
    copy_gpu_inplace(E_cont, work, shape, ss, ds, 0, N * nn,
                     CopyType::GeneralGeneral, s);
  }

  // Pre-compute all offset arrays (shared GPU memory, writable from CPU)
  size_t potrf_count = N * batch;
  size_t syrk_count = (N - 1) * batch;
  size_t trsm_count = (N - 1) * batch;
  size_t total_offsets = potrf_count + 2 * syrk_count + 2 * trsm_count;

  array all_offsets({static_cast<int>(total_offsets)}, int32, nullptr, {});
  all_offsets.set_data(allocator::malloc(total_offsets * sizeof(int32_t)));
  enc.add_temporary(all_offsets);

  int32_t* op = all_offsets.data<int32_t>();
  int32_t* potrf_off = op;
  int32_t* syrk_c_off = potrf_off + potrf_count;
  int32_t* syrk_a_off = syrk_c_off + syrk_count;
  int32_t* trsm_l_off = syrk_a_off + syrk_count;
  int32_t* trsm_b_off = trsm_l_off + trsm_count;

  for (size_t b = 0; b < batch; b++) {
    int base = static_cast<int>(b) * total_tiles * nn;
    for (int i = 0; i < N; i++) {
      potrf_off[i * batch + b] = base + i * nn;
    }
    for (int i = 0; i < N - 1; i++) {
      syrk_c_off[i * batch + b] = base + (i + 1) * nn;
      syrk_a_off[i * batch + b] = base + (N + i) * nn;
    }
    for (int i = 0; i < N - 1; i++) {
      trsm_l_off[i * batch + b] = base + i * nn;
      trsm_b_off[i * batch + b] = base + (N + i) * nn;
    }
  }

  // Get kernels
  std::string ns = std::to_string(n);
  auto k_potrf = d.get_kernel("tile_potrf_float32_" + ns);
  auto k_syrk = d.get_kernel("tile_syrk_float32_" + ns);
  auto k_trsm_r = d.get_kernel("tile_trsm_right_lower_float32_" + ns);

  // Byte offsets into all_offsets for each dispatch section
  int64_t potrf_byte = 0;
  int64_t syrk_c_byte = potrf_count * sizeof(int32_t);
  int64_t syrk_a_byte = (potrf_count + syrk_count) * sizeof(int32_t);
  int64_t trsm_l_byte = (potrf_count + 2 * syrk_count) * sizeof(int32_t);
  int64_t trsm_b_byte =
      (potrf_count + 2 * syrk_count + trsm_count) * sizeof(int32_t);

  int64_t batch_bytes = batch * sizeof(int32_t);

  MTL::Size group(n, 1, 1);
  MTL::Size grid(batch, 1, 1);

  // Sequential block Cholesky: 3 dispatches per block
  for (int i = 0; i < N; i++) {
    // syrk: D[i] -= E[i-1] · E[i-1]^T
    if (i > 0) {
      enc.set_compute_pipeline_state(k_syrk);
      enc.set_output_array(work, 0);
      enc.set_input_array(all_offsets, 1, syrk_c_byte + (i - 1) * batch_bytes);
      enc.set_input_array(all_offsets, 2, syrk_a_byte + (i - 1) * batch_bytes);
      enc.dispatch_threadgroups(grid, group);
    }

    // potrf: L_diag[i] = chol(D[i])
    enc.set_compute_pipeline_state(k_potrf);
    enc.set_output_array(work, 0);
    enc.set_input_array(all_offsets, 1, potrf_byte + i * batch_bytes);
    enc.dispatch_threadgroups(grid, group);

    // trsm: L_offdiag[i] = E[i] · L_diag[i]^{-T}
    if (i < N - 1) {
      enc.set_compute_pipeline_state(k_trsm_r);
      enc.set_output_array(work, 0);
      enc.set_input_array(all_offsets, 1, trsm_l_byte + i * batch_bytes);
      enc.set_input_array(all_offsets, 2, trsm_b_byte + i * batch_bytes);
      enc.dispatch_threadgroups(grid, group);
    }
  }

  // Copy results: work D section → L_diag, work E section → L_offdiag
  {
    Shape shape = {static_cast<int>(batch), N, n, n};
    Strides ss = {(int64_t)(total_tiles * nn), (int64_t)nn, (int64_t)n, 1};
    Strides ds = {(int64_t)(N * nn), (int64_t)nn, (int64_t)n, 1};
    copy_gpu_inplace(work, L_diag, shape, ss, ds, 0, 0,
                     CopyType::GeneralGeneral, s);
  }
  {
    Shape shape = {static_cast<int>(batch), N - 1, n, n};
    Strides ss = {(int64_t)(total_tiles * nn), (int64_t)nn, (int64_t)n, 1};
    Strides ds = {(int64_t)((N - 1) * nn), (int64_t)nn, (int64_t)n, 1};
    copy_gpu_inplace(work, L_offdiag, shape, ss, ds, N * nn, 0,
                     CopyType::GeneralGeneral, s);
  }
}

} // namespace mlx::core
