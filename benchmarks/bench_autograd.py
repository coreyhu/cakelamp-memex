#!/usr/bin/env python3
"""Autograd-level benchmarks for CakeLamp.

Times forward and backward passes through the autograd layer,
including Linear layers, MLP architectures, and loss functions.
Run with: python benchmarks/bench_autograd.py

All benchmarks use CakeLamp only -- no external ML library comparisons.
"""

import time
import statistics
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.autograd.engine import no_grad


def bench(name, fn, warmup=3, iterations=20):
    """Run a benchmark and report timing statistics."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    median = statistics.median(times)
    best = min(times)
    print(f"  {name:50s}  mean={mean:8.3f}ms  median={median:8.3f}ms  best={best:8.3f}ms")


def main():
    print("=" * 95)
    print("CakeLamp Autograd Benchmarks")
    print("=" * 95)

    # --- Forward pass ---
    print("\n--- Forward Pass ---\n")

    x32 = AutogradTensor.randn([32, 784])
    w1 = AutogradTensor.randn([784, 128], requires_grad=True)
    b1 = AutogradTensor.randn([1, 128])
    w2 = AutogradTensor.randn([128, 10], requires_grad=True)
    b2 = AutogradTensor.randn([1, 10])

    def linear_fwd():
        return x32.mm(w1) + b1

    bench("Linear forward 32x784 -> 128", linear_fwd)

    def mlp_fwd():
        h = (x32.mm(w1) + b1).relu()
        return h.mm(w2) + b2

    bench("MLP forward 784->128->10 batch=32", mlp_fwd)

    # --- Forward + Backward ---
    print("\n--- Forward + Backward ---\n")

    def linear_fwd_bwd():
        out = x32.mm(w1) + b1
        loss = out.sum()
        loss.backward()
        w1.grad = None

    bench("Linear fwd+bwd 32x784 -> 128", linear_fwd_bwd)

    def mlp_fwd_bwd():
        h = (x32.mm(w1) + b1).relu()
        out = h.mm(w2) + b2
        loss = out.sum()
        loss.backward()
        w1.grad = None
        w2.grad = None

    bench("MLP fwd+bwd 784->128->10 batch=32", mlp_fwd_bwd, warmup=2, iterations=10)

    # --- Cross-entropy loss ---
    print("\n--- Loss Computation ---\n")

    logits = AutogradTensor.randn([32, 10], requires_grad=True)

    def softmax_fwd_bwd():
        ls = logits.log_softmax(dim=1)
        loss = (-ls).sum()
        loss.backward()
        logits.grad = None

    bench("log_softmax fwd+bwd 32x10", softmax_fwd_bwd)

    # --- Reduction operations with autograd ---
    print("\n--- Reduction Ops (with grad tracking) ---\n")

    big = AutogradTensor.randn([64, 784], requires_grad=True)

    def sum_fwd_bwd():
        loss = big.sum()
        loss.backward()
        big.grad = None

    bench("sum fwd+bwd 64x784", sum_fwd_bwd)

    def mean_fwd_bwd():
        loss = big.mean()
        loss.backward()
        big.grad = None

    bench("mean fwd+bwd 64x784", mean_fwd_bwd)

    # --- no_grad overhead ---
    print("\n--- no_grad Context ---\n")

    a = AutogradTensor.randn([1000], requires_grad=True)
    b = AutogradTensor.randn([1000], requires_grad=True)

    def with_grad():
        return (a + b).sum()

    def without_grad():
        with no_grad():
            return (a + b).sum()

    bench("add+sum 1000 (with grad)", with_grad)
    bench("add+sum 1000 (no_grad)", without_grad)

    print("\n" + "=" * 95)
    print("Done.")


if __name__ == "__main__":
    main()
