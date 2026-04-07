// Copyright © 2025 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

void SparseCholeskyFactor::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& out = outputs[0];

  // Copy tile_data to output (in-place factorization)
  copy_cpu(
      inputs[0],
      out,
      inputs[0].flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream());

  int n = tile_size_;
  int nn = n * n;

  const int32_t* potrf_off = inputs[1].data<int32_t>();
  const int32_t* trsm_l_off = inputs[2].data<int32_t>();
  const int32_t* trsm_b_off = inputs[3].data<int32_t>();
  const int32_t* trsm_cp = inputs[4].data<int32_t>();
  const int32_t* syrk_c_off = inputs[5].data<int32_t>();
  const int32_t* syrk_a_off = inputs[6].data<int32_t>();
  const int32_t* syrk_cp = inputs[7].data<int32_t>();
  const int32_t* gemm_c_off = inputs[8].data<int32_t>();
  const int32_t* gemm_a_off = inputs[9].data<int32_t>();
  const int32_t* gemm_b_off = inputs[10].data<int32_t>();
  const int32_t* gemm_cp = inputs[11].data<int32_t>();

  int Nt = inputs[1].shape(0);

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_output_array(out);

  encoder.dispatch([tiles = out.data<float>(),
                    potrf_off,
                    trsm_l_off,
                    trsm_b_off,
                    trsm_cp,
                    syrk_c_off,
                    syrk_a_off,
                    syrk_cp,
                    gemm_c_off,
                    gemm_a_off,
                    gemm_b_off,
                    gemm_cp,
                    n,
                    nn,
                    Nt]() {
    float alpha = 1.0f;
    float neg_one = -1.0f;
    float one = 1.0f;

    for (int j = 0; j < Nt; j++) {
      // --- potrf ---
      float* D = tiles + potrf_off[j];
      int info;
      potrf<float>("U", &n, D, &n, &info);
      // Zero upper triangle
      for (int r = 0; r < n; r++)
        for (int c = r + 1; c < n; c++)
          D[r * n + c] = 0.0f;

      // --- trsm: B = B · L^{-T} ---
      for (int t = trsm_cp[j]; t < trsm_cp[j + 1]; t++) {
        float* L = tiles + trsm_l_off[t];
        float* B = tiles + trsm_b_off[t];
        cblas_strsm(
            CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
            n, n, 1.0f, L, n, B, n);
      }

      // --- syrk: C -= A · A^T (lower triangle) ---
      for (int t = syrk_cp[j]; t < syrk_cp[j + 1]; t++) {
        float* C = tiles + syrk_c_off[t];
        float* A = tiles + syrk_a_off[t];
        cblas_ssyrk(
            CblasRowMajor, CblasLower, CblasNoTrans,
            n, n, -1.0f, A, n, 1.0f, C, n);
        // Symmetrise for later potrf
        for (int r = 0; r < n; r++)
          for (int c = r + 1; c < n; c++)
            C[r * n + c] = C[c * n + r];
      }

      // --- gemm: C -= A · B^T ---
      for (int t = gemm_cp[j]; t < gemm_cp[j + 1]; t++) {
        float* C = tiles + gemm_c_off[t];
        float* A = tiles + gemm_a_off[t];
        float* B = tiles + gemm_b_off[t];
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            n, n, n, -1.0f, A, n, B, n, 1.0f, C, n);
      }
    }
  });
}

} // namespace mlx::core
