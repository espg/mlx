# Copyright © 2025 Apple Inc.

"""Tests for tile-based sparse Cholesky factorization."""

import unittest

import mlx.core as mx
import numpy as np

# Import the Python symbolic analysis module
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx"))
from mlx._sparse_cholesky import (
    tile_structure_from_csc,
    symbolic_factorization,
    build_ctsf,
    build_task_schedule,
)


def make_arrowhead_spd(N, arrow_width, tile_size, seed=42):
    """Create an arrowhead SPD matrix (banded + dense border).

    Structure: tridiagonal core + dense last `arrow_width` rows/cols.
    Guaranteed SPD via diagonal dominance.
    """
    rng = np.random.RandomState(seed)
    # Start with tridiagonal
    diag = rng.rand(N).astype(np.float32) + 2.0 * N
    off = rng.randn(N - 1).astype(np.float32) * 0.3
    A = np.diag(diag) + np.diag(off, -1) + np.diag(off, 1)
    # Add dense arrowhead border
    border = rng.randn(arrow_width, N).astype(np.float32) * 0.2
    A[-arrow_width:, :] += border
    A[:, -arrow_width:] += border.T
    # Ensure SPD
    A = A @ A.T + N * np.eye(N, dtype=np.float32)
    return A


def make_banded_spd(N, bandwidth, seed=42):
    """Create a banded SPD matrix."""
    rng = np.random.RandomState(seed)
    A = np.zeros((N, N), dtype=np.float32)
    for d in range(-bandwidth, bandwidth + 1):
        vals = rng.randn(N - abs(d)).astype(np.float32) * 0.3
        A += np.diag(vals, d)
    A = A @ A.T + N * np.eye(N, dtype=np.float32)
    return A


def dense_to_lower_csc(A):
    """Convert dense symmetric matrix to lower-triangular CSC (no scipy)."""
    N = A.shape[0]
    indptr = [0]
    indices = []
    data = []
    for col in range(N):
        for row in range(col, N):  # lower triangle: row >= col
            val = A[row, col]
            if val != 0.0 or row == col:  # always include diagonal
                indices.append(row)
                data.append(val)
        indptr.append(len(indices))
    return (
        np.array(indptr, dtype=np.int32),
        np.array(indices, dtype=np.int32),
        np.array(data, dtype=np.float32),
    )


def run_sparse_cholesky(A, tile_size, device):
    """Run the full sparse Cholesky pipeline and return dense L."""
    N = A.shape[0]
    indptr, indices, data = dense_to_lower_csc(A)

    # 1. Tile structure from CSC
    tile_col_ptr, tile_row_idx = tile_structure_from_csc(
        indptr, indices, N, tile_size
    )

    # 2. Symbolic factorization (fill-in)
    L_col_ptr, L_row_idx = symbolic_factorization(tile_col_ptr, tile_row_idx)

    # 3. Build CTSF tiles
    tiles_np = build_ctsf(indptr, indices, data, N, tile_size, L_col_ptr, L_row_idx)

    # 4. Build task schedule
    schedule = build_task_schedule(L_col_ptr, L_row_idx, tile_size)

    # 5. Numerical factorization via C++ primitive
    tile_data_mx = mx.array(tiles_np.reshape(-1))
    result = mx.linalg.sparse_cholesky_factor(
        tile_data_mx,
        mx.array(schedule["potrf_offsets"]),
        mx.array(schedule["trsm_l_offsets"]),
        mx.array(schedule["trsm_b_offsets"]),
        mx.array(schedule["trsm_col_ptr"]),
        mx.array(schedule["syrk_c_offsets"]),
        mx.array(schedule["syrk_a_offsets"]),
        mx.array(schedule["syrk_col_ptr"]),
        mx.array(schedule["gemm_c_offsets"]),
        mx.array(schedule["gemm_a_offsets"]),
        mx.array(schedule["gemm_b_offsets"]),
        mx.array(schedule["gemm_col_ptr"]),
        tile_size,
        stream=device,
    )
    mx.eval(result)

    # 6. Unpack tiles into dense lower triangular L
    Nt = len(L_col_ptr) - 1
    n = tile_size
    factored = np.array(result).reshape(-1, n, n)
    M = Nt * n
    L_dense = np.zeros((M, M), dtype=np.float32)
    idx = 0
    for j in range(Nt):
        for k in range(L_col_ptr[j], L_col_ptr[j + 1]):
            row = int(L_row_idx[k])
            L_dense[row * n : (row + 1) * n, j * n : (j + 1) * n] = factored[k]
            idx += 1

    # Trim to original size
    L_dense = L_dense[:N, :N]
    return L_dense


class TestSparseCholesky(unittest.TestCase):

    def _check(self, A, tile_size, device, atol=1e-2):
        L = run_sparse_cholesky(A, tile_size, device)
        reconstructed = L @ L.T
        N = A.shape[0]
        np.testing.assert_allclose(
            reconstructed[:N, :N], A, atol=atol, rtol=atol,
            err_msg=f"L@L^T != A on {device}, tile_size={tile_size}"
        )

    # --- CPU tests ---

    def test_cpu_banded_64_ts16(self):
        A = make_banded_spd(64, bandwidth=5)
        self._check(A, 16, mx.cpu)

    def test_cpu_banded_128_ts32(self):
        A = make_banded_spd(128, bandwidth=10, seed=99)
        self._check(A, 32, mx.cpu)

    def test_cpu_arrowhead_64_ts16(self):
        A = make_arrowhead_spd(64, arrow_width=4, tile_size=16)
        self._check(A, 16, mx.cpu)

    def test_cpu_matches_numpy(self):
        """Element-wise comparison with numpy dense Cholesky."""
        A = make_banded_spd(32, bandwidth=3)
        L_sparse = run_sparse_cholesky(A, 16, mx.cpu)
        L_ref = np.linalg.cholesky(A)
        np.testing.assert_allclose(L_sparse, L_ref, atol=1e-4, rtol=1e-4)

    # --- GPU tests ---

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_banded_64_ts16(self):
        A = make_banded_spd(64, bandwidth=5)
        self._check(A, 16, mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_banded_128_ts32(self):
        A = make_banded_spd(128, bandwidth=10, seed=99)
        self._check(A, 32, mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_arrowhead_64_ts16(self):
        A = make_arrowhead_spd(64, arrow_width=4, tile_size=16)
        # Arrowhead has large diag values (~16K) so absolute tol must be wider
        self._check(A, 16, mx.gpu, atol=1.0)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_matches_numpy(self):
        A = make_banded_spd(32, bandwidth=3)
        L_sparse = run_sparse_cholesky(A, 16, mx.gpu)
        L_ref = np.linalg.cholesky(A)
        np.testing.assert_allclose(L_sparse, L_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
