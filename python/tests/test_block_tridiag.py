# Copyright © 2025 Apple Inc.

"""Tests for block tridiagonal Cholesky factorization."""

import unittest

import mlx.core as mx
import numpy as np


def build_block_tridiag(D, E):
    """Build a dense matrix from block tridiagonal components.

    Args:
        D: (N, n, n) diagonal blocks
        E: (N-1, n, n) off-diagonal blocks (sub-diagonal)

    Returns:
        (N*n, N*n) dense symmetric matrix
    """
    N, n, _ = D.shape
    M = N * n
    A = np.zeros((M, M), dtype=D.dtype)
    for i in range(N):
        r = i * n
        A[r : r + n, r : r + n] = D[i]
    for i in range(N - 1):
        r = (i + 1) * n
        c = i * n
        A[r : r + n, c : c + n] = E[i]
        A[c : c + n, r : r + n] = E[i].T
    return A


def build_block_bidiag(L_diag, L_offdiag):
    """Build dense lower triangular from block bidiagonal factor."""
    N, n, _ = L_diag.shape
    M = N * n
    L = np.zeros((M, M), dtype=L_diag.dtype)
    for i in range(N):
        r = i * n
        L[r : r + n, r : r + n] = L_diag[i]
    for i in range(N - 1):
        r = (i + 1) * n
        c = i * n
        L[r : r + n, c : c + n] = L_offdiag[i]
    return L


def make_spd_block_tridiag(N, n, seed=42):
    """Create a random block tridiagonal SPD matrix.

    Generates D and E such that the assembled matrix is SPD.
    Strategy: build A = X^T X + diag_dominance * I for a banded X.
    """
    rng = np.random.RandomState(seed)
    # Make diagonally dominant block tridiagonal
    D = np.zeros((N, n, n), dtype=np.float32)
    E = rng.randn(N - 1, n, n).astype(np.float32) * 0.3

    for i in range(N):
        X = rng.randn(n, n).astype(np.float32)
        D[i] = X.T @ X + (2.0 * n) * np.eye(n, dtype=np.float32)

    return D, E


class TestBlockTridiagCholesky(unittest.TestCase):

    def _check_factorization(self, D, E, device, atol=1e-3):
        """Factor and verify L @ L^T == A."""
        D_mx = mx.array(D)
        E_mx = mx.array(E)

        L_diag, L_offdiag = mx.linalg.block_tridiag_cholesky(
            D_mx, E_mx, stream=device
        )
        mx.eval(L_diag, L_offdiag)

        L_diag_np = np.array(L_diag)
        L_offdiag_np = np.array(L_offdiag)

        # Build dense matrices
        A_dense = build_block_tridiag(D, E)
        L_dense = build_block_bidiag(L_diag_np, L_offdiag_np)

        # Check L @ L^T ≈ A
        reconstructed = L_dense @ L_dense.T
        np.testing.assert_allclose(
            reconstructed, A_dense, atol=atol, rtol=atol,
            err_msg=f"L @ L^T != A on {device}"
        )

    def test_cpu_N4_n16(self):
        D, E = make_spd_block_tridiag(4, 16)
        self._check_factorization(D, E, mx.cpu)

    def test_cpu_N8_n16(self):
        D, E = make_spd_block_tridiag(8, 16)
        self._check_factorization(D, E, mx.cpu)

    def test_cpu_N16_n16(self):
        D, E = make_spd_block_tridiag(16, 16)
        self._check_factorization(D, E, mx.cpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N4_n16(self):
        D, E = make_spd_block_tridiag(4, 16)
        self._check_factorization(D, E, mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N8_n16(self):
        D, E = make_spd_block_tridiag(8, 16)
        self._check_factorization(D, E, mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N4_n32(self):
        D, E = make_spd_block_tridiag(4, 32, seed=123)
        self._check_factorization(D, E, mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N16_n16(self):
        D, E = make_spd_block_tridiag(16, 16, seed=99)
        self._check_factorization(D, E, mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N8_n32(self):
        D, E = make_spd_block_tridiag(8, 32, seed=77)
        self._check_factorization(D, E, mx.gpu, atol=1e-2)

    def test_matches_dense_cholesky(self):
        """Compare block tridiag factor with numpy dense Cholesky."""
        N, n = 4, 16
        D, E = make_spd_block_tridiag(N, n)
        A_dense = build_block_tridiag(D, E)
        L_ref = np.linalg.cholesky(A_dense)

        D_mx = mx.array(D)
        E_mx = mx.array(E)
        L_diag, L_offdiag = mx.linalg.block_tridiag_cholesky(
            D_mx, E_mx, stream=mx.cpu
        )
        mx.eval(L_diag, L_offdiag)

        L_diag_np = np.array(L_diag)
        L_offdiag_np = np.array(L_offdiag)

        # The block diagonal/off-diagonal of L should match numpy's cholesky
        for i in range(N):
            r = i * n
            np.testing.assert_allclose(
                L_diag_np[i], L_ref[r : r + n, r : r + n],
                atol=1e-4, rtol=1e-4,
                err_msg=f"L_diag[{i}] mismatch"
            )
        for i in range(N - 1):
            r = (i + 1) * n
            c = i * n
            np.testing.assert_allclose(
                L_offdiag_np[i], L_ref[r : r + n, c : c + n],
                atol=1e-4, rtol=1e-4,
                err_msg=f"L_offdiag[{i}] mismatch"
            )


if __name__ == "__main__":
    unittest.main()
