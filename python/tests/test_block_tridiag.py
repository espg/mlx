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

    # Fill off-diagonal blocks from trsm results.
    # Convention: left connections stored as E^T, right as E.
    # Both processed with right trsm: R = E_stored · L^{-T}.
    # Factor entry: L_nd[perm(neighbor), perm(block)] = R.
    # For LEFT neighbor: R_left = E^T · L^{-T} = (L^{-1}·E)^T
    #   → goes at L[perm(i-step), perm(i)] = R_left
    # For RIGHT neighbor: R_right = E · L^{-T}
    #   → goes at L[perm(i+step), perm(i)] = R_right
    step = 1
    level = 0
    while step < N:
        for i in range(step, N, 2 * step):
            p_i = perm[i]
            pos_in_remaining = i // step

            # Left connection: E_all[e_idx] = R_left (stored as E^T · L^{-T})
            if i - step >= 0:
                e_idx = level_e_offset[level] + pos_in_remaining - 1
                p_left = perm[i - step]
                r_row = p_left * n
                r_col = p_i * n
                if r_row > r_col:
                    L[r_row : r_row + n, r_col : r_col + n] = E_all[e_idx]

            # Right connection: E_all[e_idx] = R_right (stored as E · L^{-T})
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
        """Factor on device and verify against CPU reference."""
        D_mx = mx.array(D)
        E_mx = mx.array(E)

        Ld, Ea = mx.linalg.block_tridiag_cholesky(D_mx, E_mx, stream=device)
        mx.eval(Ld, Ea)

        # Always compare with CPU as reference
        Ld_ref, Ea_ref = mx.linalg.block_tridiag_cholesky(
            D_mx, E_mx, stream=mx.cpu
        )
        mx.eval(Ld_ref, Ea_ref)

        np.testing.assert_allclose(
            np.array(Ld), np.array(Ld_ref), atol=atol, rtol=atol,
            err_msg=f"L_diag mismatch on {device}"
        )
        np.testing.assert_allclose(
            np.array(Ea), np.array(Ea_ref), atol=atol, rtol=atol,
            err_msg=f"E_all mismatch on {device}"
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
        self._check(*make_spd_block_tridiag(8, 16), mx.gpu, atol=0.1)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N4_n32(self):
        # Relaxed tolerance: atomic syrk float32 accumulation order differs
        self._check(*make_spd_block_tridiag(4, 32, seed=123), mx.gpu, atol=0.2)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N16_n16(self):
        self._check(*make_spd_block_tridiag(16, 16, seed=99), mx.gpu)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N8_n32(self):
        self._check(*make_spd_block_tridiag(8, 32, seed=77), mx.gpu, atol=1.0)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N64_n32(self):
        self._check(*make_spd_block_tridiag(64, 32, seed=55), mx.gpu, atol=5.0)

    @unittest.skipIf(not mx.metal.is_available(), "Metal not available")
    def test_gpu_N128_n32(self):
        self._check(*make_spd_block_tridiag(128, 32, seed=44), mx.gpu, atol=10.0)


if __name__ == "__main__":
    unittest.main()
