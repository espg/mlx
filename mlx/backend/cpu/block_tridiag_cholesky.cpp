// Copyright © 2025 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

void BlockTridiagCholesky::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& D_in = inputs[0];
  const auto& E_in = inputs[1];

  auto& L_diag = outputs[0];
  auto& L_offdiag = outputs[1];

  copy_cpu(
      D_in,
      L_diag,
      D_in.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream());
  copy_cpu(
      E_in,
      L_offdiag,
      E_in.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream());

  int n = D_in.shape(-1);
  int N = D_in.shape(-3);
  int nn = n * n;
  size_t batch = D_in.size() / (N * nn);

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_output_array(L_diag);
  encoder.set_output_array(L_offdiag);

  encoder.dispatch([d_ptr = L_diag.data<float>(),
                    e_ptr = L_offdiag.data<float>(),
                    n,
                    N,
                    nn,
                    batch]() mutable {
    for (size_t b = 0; b < batch; b++) {
      float* D = d_ptr + b * N * nn;
      float* E = e_ptr + b * (N - 1) * nn;

      for (int i = 0; i < N; i++) {
        float* Di = D + i * nn;

        // Step 1: D[i] -= L_offdiag[i-1] · L_offdiag[i-1]^T (syrk)
        if (i > 0) {
          float* Eprev = E + (i - 1) * nn;
          // C = Di (n×n), A = Eprev (n×n)
          // D[i] -= Eprev · Eprev^T  (lower triangle)
          cblas_ssyrk(
              CblasRowMajor,
              CblasLower,
              CblasNoTrans,
              n,  // N
              n,  // K
              -1.0f,
              Eprev,
              n,
              1.0f,
              Di,
              n);
          // Symmetrize: copy lower to upper for potrf
          for (int r = 0; r < n; r++) {
            for (int c = r + 1; c < n; c++) {
              Di[r * n + c] = Di[c * n + r];
            }
          }
        }

        // Step 2: L_diag[i] = chol(D[i])
        // LAPACK potrf: row-major lower ↔ col-major upper
        int info;
        potrf<float>(
            /* uplo */ "U",
            /* n */ &n,
            /* a */ Di,
            /* lda */ &n,
            /* info */ &info);

        // Zero upper triangle
        for (int r = 0; r < n; r++) {
          for (int c = r + 1; c < n; c++) {
            Di[r * n + c] = 0.0f;
          }
        }

        // Step 3: L_offdiag[i] = E[i] · L_diag[i]^{-T}
        // Solve X · L^T = E, overwriting E with X
        // In row-major: cblas_strsm(Right, Lower, Trans, NonUnit, ...)
        if (i < N - 1) {
          float* Ei = E + i * nn;
          cblas_strsm(
              CblasRowMajor,
              CblasRight,
              CblasLower,
              CblasTrans,
              CblasNonUnit,
              n,     // M (rows of B)
              n,     // N (cols of B)
              1.0f,  // alpha
              Di,    // A (triangular, n×n)
              n,     // lda
              Ei,    // B (n×n), overwritten with solution
              n);    // ldb
        }
      }
    }
  });
}

} // namespace mlx::core
