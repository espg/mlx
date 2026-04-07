# Copyright © 2025 Apple Inc.

"""
Tests for tile BLAS kernel algorithms.

Uses mx.fast.metal_kernel with inline MSL source that matches the
tile_blas.metal implementations. This validates the algorithms on GPU
hardware. The compiled kernels in mlx.metallib use identical logic.
"""

import unittest

import mlx.core as mx
import numpy as np


def make_spd(N, seed=42):
    """Create a random N×N symmetric positive definite matrix."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, N).astype(np.float32)
    return X.T @ X + N * np.eye(N, dtype=np.float32)


POTRF_SOURCE = """
    constexpr int S = N + 1;
    threadgroup float L[N * S];

    uint lid = thread_position_in_threadgroup.x;

    for (int row = 0; row < N; row++) {
        L[row * S + lid] = inp[row * N + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j = 0; j < N; j++) {
        if (lid == 0) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                float v = L[j * S + k];
                sum += v * v;
            }
            float d = L[j * S + j] - sum;
            L[j * S + j] = (d > 0.0f) ? metal::sqrt(d) : NAN;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float diag = L[j * S + j];

        for (int i = j + 1 + int(lid); i < N; i += N) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += L[i * S + k] * L[j * S + k];
            }
            L[i * S + j] = (L[i * S + j] - sum) / diag;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int row = 0; row < N; row++) {
        out[row * N + lid] = (row >= int(lid)) ? L[row * S + lid] : 0.0f;
    }
"""

TRSM_LEFT_SOURCE = """
    constexpr int S = N + 1;
    threadgroup float sL[N * S];

    uint lid = thread_position_in_threadgroup.x;

    for (int row = 0; row < N; row++) {
        sL[row * S + lid] = L_in[row * N + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float b[32];
    for (int i = 0; i < N; i++) {
        b[i] = B_in[i * N + lid];
    }

    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < j; k++) {
            sum += sL[j * S + k] * b[k];
        }
        b[j] = (b[j] - sum) / sL[j * S + j];
    }

    for (int i = 0; i < N; i++) {
        out[i * N + lid] = b[i];
    }
"""

TRSM_RIGHT_SOURCE = """
    constexpr int S = N + 1;
    threadgroup float sL[N * S];

    uint lid = thread_position_in_threadgroup.x;

    for (int row = 0; row < N; row++) {
        sL[row * S + lid] = L_in[row * N + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float b[32];
    for (int j = 0; j < N; j++) {
        b[j] = B_in[lid * N + j];
    }

    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < j; k++) {
            sum += sL[j * S + k] * b[k];
        }
        b[j] = (b[j] - sum) / sL[j * S + j];
    }

    for (int j = 0; j < N; j++) {
        out[lid * N + j] = b[j];
    }
"""

SYRK_SOURCE = """
    constexpr int S = N + 1;
    threadgroup float sA[N * S];

    uint lid = thread_position_in_threadgroup.x;

    for (int row = 0; row < N; row++) {
        sA[row * S + lid] = A_in[row * N + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float a_row[32];
    for (int k = 0; k < N; k++) {
        a_row[k] = sA[lid * S + k];
    }

    for (int j = 0; j <= int(lid); j++) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += a_row[k] * sA[j * S + k];
        }
        out[lid * N + j] = C_in[lid * N + j] - sum;
    }
    for (int j = int(lid) + 1; j < N; j++) {
        out[lid * N + j] = C_in[lid * N + j];
    }
"""

GEMM_SOURCE = """
    constexpr int S = N + 1;
    threadgroup float sA[N * S];
    threadgroup float sB[N * S];

    uint lid = thread_position_in_threadgroup.x;

    for (int row = 0; row < N; row++) {
        sA[row * S + lid] = A_in[row * N + lid];
        sB[row * S + lid] = B_in[row * N + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float c[32];
    for (int j = 0; j < N; j++) {
        c[j] = C_in[lid * N + j];
    }

    for (int k = 0; k < N; k++) {
        float a = sA[lid * S + k];
        for (int j = 0; j < N; j++) {
            c[j] -= a * sB[k * S + j];
        }
    }

    for (int j = 0; j < N; j++) {
        out[lid * N + j] = c[j];
    }
"""

GEADD_SOURCE = """
    uint lid = thread_position_in_threadgroup.x;
    for (int col = 0; col < N; col++) {
        out[lid * N + col] = C_in[lid * N + col] + A_in[lid * N + col];
    }
"""


@unittest.skipIf(not mx.metal.is_available(), "Metal not available")
class TestTileBlas(unittest.TestCase):

    def _run_potrf(self, n):
        A = make_spd(n)
        L_ref = np.linalg.cholesky(A)

        kernel = mx.fast.metal_kernel(
            name=f"test_potrf_{n}",
            input_names=["inp"],
            output_names=["out"],
            source=POTRF_SOURCE,
        )
        result = kernel(
            inputs=[mx.array(A)],
            template=[("N", n)],
            grid=(n, 1, 1),
            threadgroup=(n, 1, 1),
            output_shapes=[(n, n)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        L_gpu = np.array(result[0])
        np.testing.assert_allclose(L_gpu, L_ref, atol=1e-4, rtol=1e-4)

    def test_tile_potrf_16(self):
        self._run_potrf(16)

    def test_tile_potrf_32(self):
        self._run_potrf(32)

    def _run_trsm_left(self, n):
        """Test X = L^{-1} B (left solve)."""
        rng = np.random.RandomState(42)
        A = make_spd(n)
        L = np.linalg.cholesky(A)
        B = rng.randn(n, n).astype(np.float32)
        X_ref = np.linalg.solve(L, B)  # solve L @ X = B

        kernel = mx.fast.metal_kernel(
            name=f"test_trsm_left_{n}",
            input_names=["L_in", "B_in"],
            output_names=["out"],
            source=TRSM_LEFT_SOURCE,
        )
        result = kernel(
            inputs=[mx.array(L), mx.array(B)],
            template=[("N", n)],
            grid=(n, 1, 1),
            threadgroup=(n, 1, 1),
            output_shapes=[(n, n)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        X_gpu = np.array(result[0])
        np.testing.assert_allclose(X_gpu, X_ref, atol=1e-4, rtol=1e-4)

    def test_tile_trsm_left_16(self):
        self._run_trsm_left(16)

    def test_tile_trsm_left_32(self):
        self._run_trsm_left(32)

    def _run_trsm_right(self, n):
        """Test X = B L^{-T} (right solve with transpose)."""
        rng = np.random.RandomState(42)
        A = make_spd(n)
        L = np.linalg.cholesky(A)
        B = rng.randn(n, n).astype(np.float32)
        # X @ L^T = B → L @ X^T = B^T → X^T = L^{-1} B^T → X = (L^{-1} B^T)^T
        X_ref = np.linalg.solve(L, B.T).T

        kernel = mx.fast.metal_kernel(
            name=f"test_trsm_right_{n}",
            input_names=["L_in", "B_in"],
            output_names=["out"],
            source=TRSM_RIGHT_SOURCE,
        )
        result = kernel(
            inputs=[mx.array(L), mx.array(B)],
            template=[("N", n)],
            grid=(n, 1, 1),
            threadgroup=(n, 1, 1),
            output_shapes=[(n, n)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        X_gpu = np.array(result[0])
        np.testing.assert_allclose(X_gpu, X_ref, atol=1e-4, rtol=1e-4)

    def test_tile_trsm_right_16(self):
        self._run_trsm_right(16)

    def test_tile_trsm_right_32(self):
        self._run_trsm_right(32)

    def _run_syrk(self, n):
        """Test C -= A @ A^T (lower triangle only)."""
        rng = np.random.RandomState(42)
        C = rng.randn(n, n).astype(np.float32)
        A = rng.randn(n, n).astype(np.float32)
        C_ref = C.copy()
        update = A @ A.T
        # Only lower triangle is updated
        for i in range(n):
            for j in range(i + 1):
                C_ref[i, j] -= update[i, j]

        kernel = mx.fast.metal_kernel(
            name=f"test_syrk_{n}",
            input_names=["C_in", "A_in"],
            output_names=["out"],
            source=SYRK_SOURCE,
        )
        result = kernel(
            inputs=[mx.array(C), mx.array(A)],
            template=[("N", n)],
            grid=(n, 1, 1),
            threadgroup=(n, 1, 1),
            output_shapes=[(n, n)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        out_gpu = np.array(result[0])
        np.testing.assert_allclose(out_gpu, C_ref, atol=1e-4, rtol=1e-4)

    def test_tile_syrk_16(self):
        self._run_syrk(16)

    def test_tile_syrk_32(self):
        self._run_syrk(32)

    def _run_gemm(self, n):
        """Test C -= A @ B."""
        rng = np.random.RandomState(42)
        C = rng.randn(n, n).astype(np.float32)
        A = rng.randn(n, n).astype(np.float32)
        B = rng.randn(n, n).astype(np.float32)
        C_ref = C - A @ B

        kernel = mx.fast.metal_kernel(
            name=f"test_gemm_{n}",
            input_names=["C_in", "A_in", "B_in"],
            output_names=["out"],
            source=GEMM_SOURCE,
        )
        result = kernel(
            inputs=[mx.array(C), mx.array(A), mx.array(B)],
            template=[("N", n)],
            grid=(n, 1, 1),
            threadgroup=(n, 1, 1),
            output_shapes=[(n, n)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        out_gpu = np.array(result[0])
        np.testing.assert_allclose(out_gpu, C_ref, atol=1e-3, rtol=1e-3)

    def test_tile_gemm_16(self):
        self._run_gemm(16)

    def test_tile_gemm_32(self):
        self._run_gemm(32)

    def _run_geadd(self, n):
        """Test C += A."""
        rng = np.random.RandomState(42)
        C = rng.randn(n, n).astype(np.float32)
        A = rng.randn(n, n).astype(np.float32)
        C_ref = C + A

        kernel = mx.fast.metal_kernel(
            name=f"test_geadd_{n}",
            input_names=["C_in", "A_in"],
            output_names=["out"],
            source=GEADD_SOURCE,
        )
        result = kernel(
            inputs=[mx.array(C), mx.array(A)],
            template=[("N", n)],
            grid=(n, 1, 1),
            threadgroup=(n, 1, 1),
            output_shapes=[(n, n)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        out_gpu = np.array(result[0])
        np.testing.assert_allclose(out_gpu, C_ref, atol=1e-6)

    def test_tile_geadd_16(self):
        self._run_geadd(16)

    def test_tile_geadd_32(self):
        self._run_geadd(32)

    def test_potrf_reconstructs(self):
        """Verify L @ L^T = A for potrf output."""
        N = 32
        A = make_spd(N)

        kernel = mx.fast.metal_kernel(
            name="test_potrf_recon",
            input_names=["inp"],
            output_names=["out"],
            source=POTRF_SOURCE,
        )
        result = kernel(
            inputs=[mx.array(A)],
            template=[("N", N)],
            grid=(N, 1, 1),
            threadgroup=(N, 1, 1),
            output_shapes=[(N, N)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        L = np.array(result[0])
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, A, atol=1e-4, rtol=1e-4)

    def test_trsm_roundtrip(self):
        """Verify L @ (L^{-1} B) = B for left trsm."""
        N = 32
        rng = np.random.RandomState(123)
        A = make_spd(N)
        L = np.linalg.cholesky(A)
        B = rng.randn(N, N).astype(np.float32)

        kernel = mx.fast.metal_kernel(
            name="test_trsm_roundtrip",
            input_names=["L_in", "B_in"],
            output_names=["out"],
            source=TRSM_LEFT_SOURCE,
        )
        result = kernel(
            inputs=[mx.array(L), mx.array(B)],
            template=[("N", N)],
            grid=(N, 1, 1),
            threadgroup=(N, 1, 1),
            output_shapes=[(N, N)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        X = np.array(result[0])
        reconstructed = L @ X
        np.testing.assert_allclose(reconstructed, B, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
