// Copyright © 2025 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace mlx::core {

static int compute_n_e_total(int N) {
  int total = 0;
  for (int s = 1; s < N; s *= 2)
    total += (N + s - 1) / s - 1;
  return total;
}

// In-place transpose of an n×n tile stored row-major.
static void transpose_tile(float* tile, int n) {
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      std::swap(tile[i * n + j], tile[j * n + i]);
}

void BlockTridiagCholesky::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& D_in = inputs[0];
  const auto& E_in = inputs[1];

  int n = D_in.shape(-1);
  int N = D_in.shape(-3);
  int nn = n * n;
  size_t batch = D_in.size() / (N * nn);
  int n_E_total = compute_n_e_total(N);

  auto& L_diag = outputs[0];
  auto& L_offdiag = outputs[1];
  L_diag.set_data(allocator::malloc(L_diag.nbytes()));
  L_offdiag.set_data(allocator::malloc(L_offdiag.nbytes()));

  // Copy D → L_diag
  copy_cpu(D_in, L_diag,
           D_in.flags().row_contiguous ? CopyType::Vector : CopyType::General,
           stream());

  // Zero E_all, copy E_In into level 0 slots, transpose left-connection tiles
  std::memset(L_offdiag.data<float>(), 0, L_offdiag.nbytes());
  {
    const float* e_src = E_in.data<float>();
    float* e_dst = L_offdiag.data<float>();
    for (size_t b = 0; b < batch; b++) {
      std::memcpy(e_dst + b * n_E_total * nn, e_src + b * (N - 1) * nn,
                  (N - 1) * nn * sizeof(float));
      // Transpose even-indexed level-0 tiles (left connections)
      for (int j = 0; j < N - 1; j += 2) {
        transpose_tile(e_dst + b * n_E_total * nn + j * nn, n);
      }
    }
  }

  // Level offsets
  std::vector<int> level_e_offset;
  {
    int off = 0;
    for (int step = 1; step < N; step *= 2) {
      level_e_offset.push_back(off);
      off += (N + step - 1) / step - 1;
    }
    level_e_offset.push_back(off);
  }

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_output_array(L_diag);
  encoder.set_output_array(L_offdiag);

  encoder.dispatch([d_ptr = L_diag.data<float>(),
                    e_ptr = L_offdiag.data<float>(),
                    n, N, nn, batch, n_E_total,
                    lev_off = std::move(level_e_offset)]() {
    for (size_t b = 0; b < batch; b++) {
      float* D = d_ptr + b * N * nn;
      float* E_all = e_ptr + b * n_E_total * nn;

      int level = 0;
      for (int step = 1; step < N; step *= 2, level++) {
        for (int i = step; i < N; i += 2 * step) {
          int p = i / step;
          bool has_left = (i - step >= 0);
          bool has_right = (i + step < N);
          int el_idx = has_left ? (lev_off[level] + p - 1) : -1;
          int er_idx = has_right ? (lev_off[level] + p) : -1;

          float* Di = D + i * nn;

          // potrf
          int info;
          potrf<float>("U", &n, Di, &n, &info);
          for (int r = 0; r < n; r++)
            for (int c = r + 1; c < n; c++)
              Di[r * n + c] = 0.0f;

          // Right trsm on RIGHT connection: R_right = E_right · L^{-T}
          if (has_right) {
            float* Er = E_all + er_idx * nn;
            cblas_strsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans,
                        CblasNonUnit, n, n, 1.0f, Di, n, Er, n);
          }

          // Right trsm on LEFT connection (stored as E^T): R_left = E^T · L^{-T}
          if (has_left) {
            float* El = E_all + el_idx * nn;
            cblas_strsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans,
                        CblasNonUnit, n, n, 1.0f, Di, n, El, n);
          }

          // syrk on right neighbor: D[i+step] -= R_right · R_right^T
          if (has_right) {
            float* Er = E_all + er_idx * nn;
            float* Dn = D + (i + step) * nn;
            cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                        n, n, -1.0f, Er, n, 1.0f, Dn, n);
            for (int r = 0; r < n; r++)
              for (int c = r + 1; c < n; c++)
                Dn[r * n + c] = Dn[c * n + r];
          }

          // syrk on left neighbor: D[i-step] -= R_left · R_left^T
          if (has_left) {
            float* El = E_all + el_idx * nn;
            float* Dn = D + (i - step) * nn;
            cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                        n, n, -1.0f, El, n, 1.0f, Dn, n);
            for (int r = 0; r < n; r++)
              for (int c = r + 1; c < n; c++)
                Dn[r * n + c] = Dn[c * n + r];
          }

          // gemm_bt fill-in: E_next -= R_right · R_left^T
          if (has_left && has_right) {
            int next_j = (i - step) / (2 * step);
            int en_idx = lev_off[level + 1] + next_j;
            float* En = E_all + en_idx * nn;
            float* Er = E_all + er_idx * nn;
            float* El = E_all + el_idx * nn;
            // C -= A · B^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n, n, n, -1.0f, Er, n, El, n, 1.0f, En, n);
            // If next_j is even, this tile is a LEFT connection at next level
            // → transpose to match convention
            if (next_j % 2 == 0) {
              transpose_tile(En, n);
            }
          }
        }
      }

      // Final: factor block 0
      int info;
      potrf<float>("U", &n, D, &n, &info);
      for (int r = 0; r < n; r++)
        for (int c = r + 1; c < n; c++)
          D[r * n + c] = 0.0f;
    }
  });
}

} // namespace mlx::core
