# Copyright © 2025 Apple Inc.

"""Tests for block tridiagonal Cholesky via nested dissection."""

import unittest

import mlx.core as mx
import numpy as np


def build_block_tridiag(D, E):
    """Build dense matrix from block tridiagonal (D, E)."""
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


def compute_level_offsets(N):
    """Compute E_all level structure for nested dissection."""
    offsets = []
    step = 1
    while step < N:
        n_remain = (N + step - 1) // step
        offsets.append(n_remain - 1)
        step *= 2
    return offsets


def compute_elimination_order(N):
    """Return list of (block_index, level, step) in elimination order."""
    order = []
    step = 1
    level = 0
    while step < N:
        for i in range(step, N, 2 * step):
            order.append((i, level, step))
        step *= 2
        level += 1
    order.append((0, level, step))  # final block
    return order


def build_nd_factor(L_diag, E_all, N, n):
    """Build dense lower triangular L_nd from nested dissection output.

    Returns (L_dense, perm) where L_dense is the factor in the permuted
    ordering and perm maps original block index to permuted position.
    """
    # Build elimination order → permutation
    elim_order = compute_elimination_order(N)
    perm = [0] * N
    for pos, (block_idx, _, _) in enumerate(elim_order):
        perm[block_idx] = pos

    # Build permutation matrix
    M = N * n
    P = np.zeros((M, M), dtype=np.float32)
    for orig_block in range(N):
        perm_pos = perm[orig_block]
        for k in range(n):
            P[perm_pos * n + k, orig_block * n + k] = 1.0

    # Build L_nd (dense lower triangular in permuted ordering)
    L = np.zeros((M, M), dtype=np.float32)

    # Level offsets in E_all
    level_e_offset = []
    offset = 0
    step = 1
    while step < N:
        level_e_offset.append(offset)
        n_remain = (N + step - 1) // step
        offset += n_remain - 1
        step *= 2
    level_e_offset.append(offset)

    # Fill diagonal blocks
    for block_idx in range(N):
        pp = perm[block_idx]
        r = pp * n
        L[r : r + n, r : r + n] = L_diag[block_idx]

    # Fill off-diagonal blocks from trsm results
    step = 1
    level = 0
    while step < N:
        for i in range(step, N, 2 * step):
            p_i = perm[i]
            pos_in_remaining = i // step

            # Left connection: i → i-step
            if i - step >= 0:
                e_idx = level_e_offset[level] + pos_in_remaining - 1
                p_left = perm[i - step]
                # L_nd[p_left, p_i] = E_all[e_idx]  (left trsm result)
                r_row = p_left * n
                r_col = p_i * n
                if r_row > r_col:  # lower triangle
                    L[r_row : r_row + n, r_col : r_col + n] = E_all[e_idx]

            # Right connection: i → i+step
            if i + step < N:
                e_idx = level_e_offset[level] + pos_in_remaining
                p_right = perm[i + step]
                r_row = p_right * n
                r_col = p_i * n
                if r_row > r_col:
                    L[r_row : r_row + n, r_col : r_col + n] = E_all[e_idx]

        step *= 2
        level += 1

    return L, P


def make_spd_block_tridiag(N, n, seed=42):
    """Create random block tridiagonal SPD matrix."""
    rng = np.random.RandomState(seed)
    D = np.zeros((N, n, n), dtype=np.float32)
    E = rng.randn(N - 1, n, n).astype(np.float32) * 0.3
    for i in range(N):
        X = rng.randn(n, n).astype(np.float32)
        D[i] = X.T @ X + (2.0 * n) * np.eye(n, dtype=np.float32)
    return D, E


class TestBlockTridiagCholesky(unittest.TestCase):

    def _check(self, D, E, device, atol=1e-2):
        """Factor and verify L_nd @ L_nd^T == P @ A @ P^T."""
        N, n, _ = D.shape
        D_mx = mx.array(D)
        E_mx = mx.array(E)

        L_diag_mx, E_all_mx = mx.linalg.block_tridiag_cholesky(
            D_mx, E_mx, stream=device
        )
        mx.eval(L_diag_mx, E_all_mx)

        L_diag_np = np.array(L_diag_mx)
        E_all_np = np.array(E_all_mx).reshape(-1, n, n)

        # Build dense A
        A = build_block_tridiag(D, E)
        # Build factor and permutation
        L_nd, P = build_nd_factor(L_diag_np, E_all_np, N, n)
        # Verify L_nd @ L_nd^T == P @ A @ P^T
        PAP = P @ A @ P.T
        LLT = L_nd @ L_nd.T

        np.testing.assert_allclose(
            LLT, PAP, atol=atol, rtol=atol,
            err_msg=f"L_nd @ L_nd^T != P@A@P^T on {device}"
        )

    # CPU tests
    def test_cpu_N4_n16(self):
        self._check(*make_spd_block_tridiag(4, 16), mx.cpu)

    def test_cpu_N8_n16(self):
        self._check(*make_spd_block_tridiag(8, 16), mx.cpu)

    def test_cpu_N16_n16(self):
        self._check(*make_spd_block_tridiag(16, 16), mx.cpu)

    # GPU tests
    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N4_n16(self):
        self._check(*make_spd_block_tridiag(4, 16), mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N8_n16(self):
        self._check(*make_spd_block_tridiag(8, 16), mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N4_n32(self):
        self._check(*make_spd_block_tridiag(4, 32, seed=123), mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N16_n16(self):
        self._check(*make_spd_block_tridiag(16, 16, seed=99), mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N8_n32(self):
        self._check(*make_spd_block_tridiag(8, 32, seed=77), mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N64_n32(self):
        self._check(*make_spd_block_tridiag(64, 32, seed=55), mx.gpu, atol=0.1)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N128_n32(self):
        self._check(*make_spd_block_tridiag(128, 32, seed=44), mx.gpu, atol=0.5)


if __name__ == "__main__":
    unittest.main()
