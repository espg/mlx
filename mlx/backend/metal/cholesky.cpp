// Copyright © 2025 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const auto& a = inputs[0];

  if (a.dtype() != float32) {
    throw std::runtime_error(
        "[Cholesky::eval_gpu] Only float32 is supported on GPU.");
  }

  int N = a.shape(-1);
  size_t num_matrices = a.size() / (N * N);

  // Copy input to output for in-place computation
  copy_gpu(a, out, CopyType::General, s);

  if (N <= 1) {
    if (N == 0) {
      return;
    }
    // N == 1: sqrt is handled by the small kernel below
  }

  auto& compute_encoder = metal::get_command_encoder(s);

  if (N <= 32) {
    auto kernel = d.get_kernel("cholesky_small_float32_32");
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_bytes(N, 1);
    compute_encoder.set_bytes(upper_, 2);

    int tg_size = 32;
    MTL::Size grid_dims(tg_size * num_matrices, 1, 1);
    MTL::Size group_dims(tg_size, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  } else if (N <= 64) {
    auto kernel = d.get_kernel("cholesky_small_float32_64");
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_bytes(N, 1);
    compute_encoder.set_bytes(upper_, 2);

    int tg_size = 64;
    MTL::Size grid_dims(tg_size * num_matrices, 1, 1);
    MTL::Size group_dims(tg_size, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  } else {
    // Blocked algorithm with B=64 (halves dispatch count vs B=32)
    const int BLOCK_SIZE = 64;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto kernel_diag_panel = d.get_kernel("cholesky_diag_panel_float32_64");
    auto kernel_update = d.get_kernel("cholesky_update_float32_64");
    auto kernel_finalize = d.get_kernel("cholesky_finalize_float32");

    int panel_tg_size = 256;
    int update_tg_size = 256;

    for (int k = 0; k < num_blocks; k++) {
      // Step 1: Factor diagonal block + solve panel
      compute_encoder.set_compute_pipeline_state(kernel_diag_panel);
      compute_encoder.set_output_array(out, 0);
      compute_encoder.set_bytes(N, 1);
      compute_encoder.set_bytes(k, 2);
      compute_encoder.set_bytes(BLOCK_SIZE, 3);

      MTL::Size dp_grid(panel_tg_size * num_matrices, 1, 1);
      MTL::Size dp_group(panel_tg_size, 1, 1);
      compute_encoder.dispatch_threads(dp_grid, dp_group);

      // Step 2: Update trailing submatrix
      int remaining = num_blocks - k - 1;
      if (remaining > 0) {
        int num_tiles = remaining * (remaining + 1) / 2;

        compute_encoder.set_compute_pipeline_state(kernel_update);
        compute_encoder.set_output_array(out, 0);
        compute_encoder.set_bytes(N, 1);
        compute_encoder.set_bytes(k, 2);
        compute_encoder.set_bytes(BLOCK_SIZE, 3);
        compute_encoder.set_bytes(remaining, 4);

        MTL::Size up_grid(
            num_tiles * update_tg_size,
            static_cast<NS::UInteger>(num_matrices),
            1);
        MTL::Size up_group(update_tg_size, 1, 1);
        compute_encoder.dispatch_threads(up_grid, up_group);
      }
    }

    // Finalize: zero upper triangle or transpose for upper output
    compute_encoder.set_compute_pipeline_state(kernel_finalize);
    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_bytes(N, 1);
    compute_encoder.set_bytes(upper_, 2);

    int fin_tg_size = 256;
    MTL::Size fin_grid(fin_tg_size * num_matrices, 1, 1);
    MTL::Size fin_group(fin_tg_size, 1, 1);
    compute_encoder.dispatch_threads(fin_grid, fin_group);
  }
}

} // namespace mlx::core
