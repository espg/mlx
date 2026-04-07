# Copyright © 2025 Apple Inc.

"""Benchmark block tridiagonal Cholesky factorization (GPU vs CPU vs NumPy)."""

import argparse
import time

import mlx.core as mx
import numpy as np


def make_spd_block_tridiag(N, n, batch=1, seed=42):
    """Create random block tridiagonal SPD matrix components."""
    rng = np.random.RandomState(seed)
    shape_D = (batch, N, n, n) if batch > 1 else (N, n, n)
    shape_E = (batch, N - 1, n, n) if batch > 1 else (N - 1, n, n)

    D_np = np.zeros(shape_D, dtype=np.float32)
    E_np = rng.randn(*shape_E).astype(np.float32) * 0.3

    if batch > 1:
        for b in range(batch):
            for i in range(N):
                X = rng.randn(n, n).astype(np.float32)
                D_np[b, i] = X.T @ X + (2.0 * n) * np.eye(n, dtype=np.float32)
    else:
        for i in range(N):
            X = rng.randn(n, n).astype(np.float32)
            D_np[i] = X.T @ X + (2.0 * n) * np.eye(n, dtype=np.float32)

    return D_np, E_np


def build_dense(D, E):
    """Build dense matrix from block tridiagonal components (no batch)."""
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


def bench(N, n, batch=1, warmup=5, iters=30):
    D_np, E_np = make_spd_block_tridiag(N, n, batch)
    D_mx = mx.array(D_np)
    E_mx = mx.array(E_np)
    mx.eval(D_mx, E_mx)

    # --- GPU benchmark ---
    for _ in range(warmup):
        Ld, Lo = mx.linalg.block_tridiag_cholesky(D_mx, E_mx, stream=mx.gpu)
        mx.eval(Ld, Lo)

    gpu_start = time.perf_counter()
    for _ in range(iters):
        Ld, Lo = mx.linalg.block_tridiag_cholesky(D_mx, E_mx, stream=mx.gpu)
        mx.eval(Ld, Lo)
    gpu_time = (time.perf_counter() - gpu_start) / iters

    # --- CPU benchmark (MLX LAPACK) ---
    for _ in range(warmup):
        Ld, Lo = mx.linalg.block_tridiag_cholesky(D_mx, E_mx, stream=mx.cpu)
        mx.eval(Ld, Lo)

    cpu_start = time.perf_counter()
    for _ in range(iters):
        Ld, Lo = mx.linalg.block_tridiag_cholesky(D_mx, E_mx, stream=mx.cpu)
        mx.eval(Ld, Lo)
    cpu_time = (time.perf_counter() - cpu_start) / iters

    # --- NumPy dense Cholesky baseline ---
    if batch == 1:
        A_dense = build_dense(D_np, E_np)
        for _ in range(warmup):
            np.linalg.cholesky(A_dense)
        np_start = time.perf_counter()
        for _ in range(iters):
            np.linalg.cholesky(A_dense)
        np_time = (time.perf_counter() - np_start) / iters
    else:
        np_time = float("nan")

    return gpu_time, cpu_time, np_time


def main():
    parser = argparse.ArgumentParser(
        description="Block tridiagonal Cholesky benchmark"
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64, 128],
        help="Number of blocks N",
    )
    parser.add_argument(
        "--tile-sizes",
        nargs="+",
        type=int,
        default=[16, 32],
        help="Tile sizes n",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    print(
        f"{'N':>6} {'n':>4} {'batch':>6} {'dense':>7} "
        f"{'GPU (ms)':>10} {'CPU (ms)':>10} {'NP dense':>10} "
        f"{'GPU/CPU':>8} {'GPU/NP':>8}"
    )
    print("-" * 82)

    for n in args.tile_sizes:
        for N in args.blocks:
            M = N * n
            gpu_t, cpu_t, np_t = bench(
                N, n, args.batch, args.warmup, args.iters
            )
            r_cpu = gpu_t / cpu_t if cpu_t > 0 else float("nan")
            r_np = gpu_t / np_t if np_t > 0 else float("nan")
            print(
                f"{N:>6} {n:>4} {args.batch:>6} {M:>7} "
                f"{gpu_t*1000:>10.3f} {cpu_t*1000:>10.3f} {np_t*1000:>10.3f} "
                f"{r_cpu:>7.2f}x {r_np:>7.2f}x"
            )


if __name__ == "__main__":
    main()
