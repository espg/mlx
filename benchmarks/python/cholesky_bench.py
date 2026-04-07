# Copyright © 2025 Apple Inc.

import argparse
import time

import mlx.core as mx
import numpy as np


def generate_spd(N, batch_size=1, dtype=mx.float32):
    """Generate a batch of symmetric positive definite matrices."""
    shape = (batch_size, N, N) if batch_size > 1 else (N, N)
    A = mx.random.normal(shape)
    # A @ A^T + N*I guarantees positive definiteness
    if batch_size > 1:
        At = mx.transpose(A, axes=(0, 2, 1))
    else:
        At = A.T
    result = A @ At + N * mx.eye(N)
    return result.astype(dtype)


def bench_cholesky(N, batch_size=1, warmup=10, iters=100, upper=False):
    """Benchmark Cholesky decomposition on GPU vs CPU."""
    A = generate_spd(N, batch_size)
    mx.eval(A)

    # GPU benchmark
    for _ in range(warmup):
        L = mx.linalg.cholesky(A, upper=upper, stream=mx.gpu)
        mx.eval(L)

    gpu_start = time.perf_counter()
    for _ in range(iters):
        L = mx.linalg.cholesky(A, upper=upper, stream=mx.gpu)
        mx.eval(L)
    gpu_time = (time.perf_counter() - gpu_start) / iters

    # CPU benchmark
    for _ in range(warmup):
        L = mx.linalg.cholesky(A, upper=upper, stream=mx.cpu)
        mx.eval(L)

    cpu_start = time.perf_counter()
    for _ in range(iters):
        L = mx.linalg.cholesky(A, upper=upper, stream=mx.cpu)
        mx.eval(L)
    cpu_time = (time.perf_counter() - cpu_start) / iters

    # NumPy benchmark
    A_np = np.array(A)
    for _ in range(warmup):
        np.linalg.cholesky(A_np)

    np_start = time.perf_counter()
    for _ in range(iters):
        np.linalg.cholesky(A_np)
    np_time = (time.perf_counter() - np_start) / iters

    return gpu_time, cpu_time, np_time


def main():
    parser = argparse.ArgumentParser(description="Cholesky decomposition benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512],
        help="Matrix sizes to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="Batch sizes to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    print(
        f"{'N':>6} {'Batch':>6} {'GPU (ms)':>10} {'CPU (ms)':>10} "
        f"{'NumPy (ms)':>11} {'GPU/CPU':>8} {'GPU/NumPy':>10}"
    )
    print("-" * 72)

    for batch_size in args.batch_sizes:
        for N in args.sizes:
            gpu_t, cpu_t, np_t = bench_cholesky(N, batch_size, args.warmup, args.iters)
            ratio_cpu = gpu_t / cpu_t
            ratio_np = gpu_t / np_t
            print(
                f"{N:>6} {batch_size:>6} {gpu_t*1000:>10.3f} {cpu_t*1000:>10.3f} "
                f"{np_t*1000:>11.3f} {ratio_cpu:>8.2f}x {ratio_np:>9.2f}x"
            )


if __name__ == "__main__":
    main()
