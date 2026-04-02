#!/usr/bin/env python3
"""Benchmarks for CakeLamp tensor operations.

Measures wall-clock time for key tensor ops at various sizes.
Run with: python benchmarks/bench_ops.py

Timing methodology:
  - Each op is run N times and we report the median time per call.
  - First run is a warm-up and is discarded.
"""

from __future__ import annotations

import time
import statistics
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from cakelamp import Tensor


def _bench(fn, warmup=2, repeats=20):
    """Run fn() `warmup + repeats` times, return median time in ms."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    return statistics.median(times)


def bench_matmul():
    """Benchmark matrix multiplication at various sizes."""
    print("\n=== Matrix Multiplication ===")
    print(f"{'Size':>12} {'Time (ms)':>12} {'GFLOP/s':>12}")
    print("-" * 40)

    for n in [32, 64, 128, 256]:
        a = Tensor.rand([n, n])
        b = Tensor.rand([n, n])
        ms = _bench(lambda: a.matmul(b))
        # 2*n^3 FLOPs for matmul
        gflops = 2.0 * n**3 / (ms / 1000.0) / 1e9
        print(f"{n:>4}x{n:<4}    {ms:>10.3f}   {gflops:>10.3f}")


def bench_element_wise():
    """Benchmark element-wise operations."""
    print("\n=== Element-wise Ops ===")
    print(f"{'Op':>12} {'Size':>10} {'Time (ms)':>12} {'GB/s':>10}")
    print("-" * 48)

    for n in [10_000, 100_000]:
        a = Tensor.rand([n])
        b = Tensor.rand([n])

        for name, fn in [
            ("add", lambda: a + b),
            ("mul", lambda: a * b),
            ("exp", lambda: a.exp()),
            ("relu", lambda: a.relu()),
            ("sigmoid", lambda: a.sigmoid()),
        ]:
            ms = _bench(fn)
            # Bandwidth: read 2 tensors, write 1 (binary) or read 1, write 1 (unary)
            bytes_rw = n * 4 * (3 if name in ("add", "mul") else 2)
            gb_s = bytes_rw / (ms / 1000.0) / 1e9
            print(f"{name:>12} {n:>10,} {ms:>10.3f}   {gb_s:>8.3f}")


def bench_reductions():
    """Benchmark reduction operations."""
    print("\n=== Reductions ===")
    print(f"{'Op':>12} {'Size':>10} {'Time (ms)':>12}")
    print("-" * 38)

    n = 100_000
    a = Tensor.rand([n])
    for name, fn in [
        ("sum", lambda: a.sum()),
        ("mean", lambda: a.mean()),
        ("max", lambda: a.max()),
    ]:
        ms = _bench(fn)
        print(f"{name:>12} {n:>10,} {ms:>10.3f}")


def bench_softmax():
    """Benchmark softmax and log_softmax."""
    print("\n=== Softmax / Log-Softmax ===")
    print(f"{'Op':>16} {'Shape':>12} {'Time (ms)':>12}")
    print("-" * 44)

    for rows, cols in [(64, 10), (256, 10), (64, 128)]:
        a = Tensor.rand([rows, cols])
        for name, fn in [
            ("softmax", lambda: a.softmax(1)),
            ("log_softmax", lambda: a.log_softmax(1)),
        ]:
            ms = _bench(fn)
            print(f"{name:>16} {rows}x{cols:>4}     {ms:>10.3f}")


def bench_autograd():
    """Benchmark autograd forward+backward pass."""
    from cakelamp.autograd.tensor import AutogradTensor

    print("\n=== Autograd Forward+Backward ===")
    print(f"{'Op':>20} {'Time (ms)':>12}")
    print("-" * 36)

    def mlp_fwd_bwd():
        x = AutogradTensor._make(
            [0.1] * (64 * 784), [64, 784], requires_grad=False
        )
        W1 = AutogradTensor._make(
            [0.01] * (784 * 128), [784, 128], requires_grad=True
        )
        W2 = AutogradTensor._make(
            [0.01] * (128 * 10), [128, 10], requires_grad=True
        )
        h = x.mm(W1).relu()
        out = h.mm(W2)
        loss = out.sum()
        loss.backward()

    ms = _bench(mlp_fwd_bwd, warmup=1, repeats=5)
    print(f"{'MLP 784->128->10':>20} {ms:>10.1f}")

    def simple_chain():
        x = AutogradTensor._make([0.5] * 1000, [1000], requires_grad=True)
        y = (x * x).exp().sum()
        y.backward()

    ms = _bench(simple_chain, warmup=2, repeats=10)
    print(f"{'exp(x*x).sum()':>20} {ms:>10.3f}")


def main():
    print("CakeLamp Benchmark Suite")
    print("=" * 50)

    bench_matmul()
    bench_element_wise()
    bench_reductions()
    bench_softmax()
    bench_autograd()

    print("\n" + "=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
