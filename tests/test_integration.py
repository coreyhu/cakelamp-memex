"""Comprehensive integration tests for CakeLamp.

Tests the autograd engine with real Rust backend tensors (no mocks).
Covers gaps in existing test suite:
- Division backward
- Gradient accumulation (a*a, multi-path)
- Softmax / log_softmax backward
- Complex chain rule expressions
- Manual MLP training loop (matmul + relu + MSE)
- Manual gradient descent convergence

Note: Tests that require nn.Linear + AutogradTensor integration
(e.g., full MLP with nn modules) are marked with xfail pending
PR #23 which integrates nn modules with autograd.
"""

from __future__ import annotations

import math
import random

import pytest

from cakelamp.autograd.tensor import AutogradTensor


# =====================================================================
# Helper
# =====================================================================

def approx_list(actual, expected, tol=1e-4):
    """Assert two flat lists are element-wise close."""
    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) < tol, f"[{i}]: {a} != {e} (diff={abs(a - e):.6f})"


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


# =====================================================================
# Autograd: division backward
# =====================================================================

class TestDivBackward:
    def test_div_basic(self):
        """d(a/b)/da = 1/b, d(a/b)/db = -a/b^2"""
        a = AutogradTensor.from_data([6.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([3.0], [1], requires_grad=True)
        c = a / b
        c.backward()
        assert approx(a.grad.tolist()[0], 1.0 / 3.0)
        assert approx(b.grad.tolist()[0], -6.0 / 9.0)

    def test_div_multi_element(self):
        """Element-wise division backward with multiple elements."""
        a = AutogradTensor.from_data([8.0, 12.0], [2], requires_grad=True)
        b = AutogradTensor.from_data([2.0, 4.0], [2], requires_grad=True)
        c = (a / b).sum()
        c.backward()
        approx_list(a.grad.tolist(), [0.5, 0.25])
        approx_list(b.grad.tolist(), [-2.0, -0.75])

    def test_div_chain(self):
        """d(a/(a+b))/da at a=2,b=2 = b/(a+b)^2 = 0.125"""
        a = AutogradTensor.from_data([2.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([2.0], [1], requires_grad=True)
        result = a / (a + b)
        result.backward()
        assert approx(a.grad.tolist()[0], 0.125)


# =====================================================================
# Autograd: gradient accumulation
# =====================================================================

class TestGradAccumulation:
    def test_variable_used_twice(self):
        """d(a*a)/da = 2*a (same variable used twice)."""
        a = AutogradTensor.from_data([3.0], [1], requires_grad=True)
        c = a * a
        c.backward()
        assert approx(a.grad.tolist()[0], 6.0)

    def test_variable_used_three_times(self):
        """d(a*a*a)/da = 3*a^2 via chain rule."""
        a = AutogradTensor.from_data([2.0], [1], requires_grad=True)
        c = a * a * a
        c.backward()
        assert approx(a.grad.tolist()[0], 12.0)

    def test_accumulation_from_two_paths(self):
        """d(a*b + a*c)/da = b + c."""
        a = AutogradTensor.from_data([1.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([3.0], [1])
        c = AutogradTensor.from_data([5.0], [1])
        result = a * b + a * c
        result.backward()
        assert approx(a.grad.tolist()[0], 8.0)  # 3 + 5

    def test_multi_element_accumulation(self):
        """Multi-element gradient accumulation: d(a*a)/da = 2*a element-wise."""
        a = AutogradTensor.from_data([1.0, 2.0, 3.0], [3], requires_grad=True)
        c = (a * a).sum()
        c.backward()
        approx_list(a.grad.tolist(), [2.0, 4.0, 6.0])


# =====================================================================
# Autograd: softmax / log_softmax backward
# =====================================================================

class TestSoftmaxBackward:
    def test_softmax_backward(self):
        """Softmax backward: d(sum(softmax))/dx should be ~0."""
        x = AutogradTensor.from_data(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True
        )
        s = x.softmax(dim=1)
        loss = s.sum()
        loss.backward()
        grad = x.grad.tolist()
        assert all(math.isfinite(g) for g in grad)
        # sum of softmax = 1 per row; total = 2; gradients should be ~0
        for g in grad:
            assert abs(g) < 1e-4, f"Expected ~0 gradient, got {g}"

    def test_log_softmax_backward(self):
        """Log softmax backward should produce finite gradients."""
        x = AutogradTensor.from_data(
            [2.0, 1.0, 0.5, 0.5, 1.0, 2.0], [2, 3], requires_grad=True
        )
        ls = x.log_softmax(dim=1)
        loss = ls.sum()
        loss.backward()
        grad = x.grad.tolist()
        assert all(math.isfinite(g) for g in grad)

    @pytest.mark.xfail(reason="nll_gather not yet on tensor.py AutogradTensor; pending PR #21")
    def test_log_softmax_with_nll_gather(self):
        """log_softmax + nll_gather = cross-entropy, should backpropagate."""
        logits = AutogradTensor.from_data(
            [2.0, 1.0, 0.1, 0.1, 1.0, 2.0], [2, 3], requires_grad=True
        )
        targets = AutogradTensor.from_data([0.0, 2.0], [2])
        log_probs = logits.log_softmax(dim=1)
        nll = log_probs.nll_gather(targets)
        loss = nll.mean()
        loss.backward()
        assert logits.grad is not None


# =====================================================================
# Autograd: complex chain rule
# =====================================================================

class TestComplexChainRule:
    def test_four_variable_chain(self):
        """d((a*b + c)*d)/da = b*d, d/db = a*d, d/dc = d, d/dd = a*b+c."""
        a = AutogradTensor.from_data([2.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([3.0], [1], requires_grad=True)
        c = AutogradTensor.from_data([1.0], [1], requires_grad=True)
        d = AutogradTensor.from_data([4.0], [1], requires_grad=True)
        result = (a * b + c) * d
        result.backward()
        assert approx(a.grad.tolist()[0], 12.0)  # b*d = 3*4
        assert approx(b.grad.tolist()[0], 8.0)   # a*d = 2*4
        assert approx(c.grad.tolist()[0], 4.0)   # d = 4
        assert approx(d.grad.tolist()[0], 7.0)   # a*b+c = 7

    def test_mixed_ops_chain(self):
        """d((a+b)*(a-b))/da = 2a, d/db = -2b."""
        a = AutogradTensor.from_data([3.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([2.0], [1], requires_grad=True)
        result = (a + b) * (a - b)  # = a^2 - b^2
        result.backward()
        assert approx(a.grad.tolist()[0], 6.0)   # 2*a
        assert approx(b.grad.tolist()[0], -4.0)  # -2*b

    def test_nested_division(self):
        """Nested division: d(1/(a*b))/da = -1/(a^2 * b)."""
        a = AutogradTensor.from_data([2.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([3.0], [1], requires_grad=True)
        one = AutogradTensor.from_data([1.0], [1])
        result = one / (a * b)
        result.backward()
        # d(1/(ab))/da = -1/(a^2 * b) = -1/12
        assert approx(a.grad.tolist()[0], -1.0 / 12.0)


# =====================================================================
# Manual MLP training (using raw AutogradTensors, no nn.Linear)
# =====================================================================

class TestManualMLPTraining:
    def test_regression_loss_decreases(self):
        """Manual matmul-based MLP: loss should decrease over training."""
        random.seed(42)

        # W1: [2, 4], b1: [1, 4], W2: [4, 1], b2: [1, 1]
        W1 = AutogradTensor.from_data(
            [random.gauss(0, 0.5) for _ in range(8)], [2, 4], requires_grad=True
        )
        b1 = AutogradTensor.from_data([0.0] * 4, [1, 4], requires_grad=True)
        W2 = AutogradTensor.from_data(
            [random.gauss(0, 0.5) for _ in range(4)], [4, 1], requires_grad=True
        )
        b2 = AutogradTensor.from_data([0.0], [1, 1], requires_grad=True)

        # Synthetic data: y = x1 + x2
        n = 8
        x_data = [random.uniform(-1, 1) for _ in range(n * 2)]
        y_data = [x_data[i * 2] + x_data[i * 2 + 1] for i in range(n)]
        x = AutogradTensor.from_data(x_data, [n, 2])
        y_target = AutogradTensor.from_data(y_data, [n, 1])

        lr = 0.01
        losses = []

        for step in range(50):
            # Forward: x @ W1 + b1 -> relu -> @ W2 + b2
            h = x.matmul(W1) + b1
            h = h.relu()
            pred = h.matmul(W2) + b2
            diff = pred - y_target
            loss = (diff * diff).mean()
            losses.append(loss.item())

            loss.backward()

            # Manual SGD
            for p in [W1, b1, W2, b2]:
                if p.grad is not None:
                    p.data.sub_alpha_(p.grad.data, lr)
                    p.zero_grad()

        assert losses[-1] < losses[0], (
            f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_manual_matmul_regression_converges(self):
        """Single-layer linear regression converges to correct weights."""
        # y = 2*x, fit w from 0
        w = AutogradTensor.from_data([0.0], [1, 1], requires_grad=True)
        x = AutogradTensor.from_data([1.0, 2.0, 3.0], [3, 1])
        target = AutogradTensor.from_data([2.0, 4.0, 6.0], [3, 1])

        lr = 0.01
        for _ in range(100):
            pred = x.matmul(w)
            diff = pred - target
            loss = (diff * diff).mean()
            loss.backward()

            if w.grad is not None:
                w.data.sub_alpha_(w.grad.data, lr)
                w.zero_grad()

        # w should be close to 2.0
        assert approx(w.tolist()[0], 2.0, tol=0.1)

    def test_classification_loss_decreases(self):
        """Manual classification: logits -> softmax -> manual CE loss decreases."""
        random.seed(99)

        n = 12
        in_dim = 4
        n_classes = 3

        W = AutogradTensor.from_data(
            [random.gauss(0, 0.3) for _ in range(in_dim * n_classes)],
            [in_dim, n_classes], requires_grad=True
        )
        b = AutogradTensor.from_data([0.0] * n_classes, [1, n_classes], requires_grad=True)

        # Separable data with one-hot targets
        flat_data = []
        onehot_data = []
        for i in range(n):
            cls = i % n_classes
            oh = [0.0] * n_classes
            oh[cls] = 1.0
            onehot_data.extend(oh)
            feat = [0.0] * in_dim
            feat[cls] = 1.0 + random.uniform(0, 0.2)
            for j in range(in_dim):
                feat[j] += random.uniform(-0.05, 0.05)
            flat_data.extend(feat)

        x = AutogradTensor.from_data(flat_data, [n, in_dim])
        targets_oh = AutogradTensor.from_data(onehot_data, [n, n_classes])

        lr = 0.1
        initial_loss = None
        final_loss = None

        for step in range(30):
            logits = x.matmul(W) + b
            log_probs = logits.log_softmax(dim=1)
            # Manual NLL: loss = -mean(sum(one_hot * log_probs))
            selected = targets_oh * log_probs
            loss = -selected.sum() / AutogradTensor.from_data([float(n)], [1])

            loss_val = loss.item()
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val

            loss.backward()

            for p in [W, b]:
                if p.grad is not None:
                    p.data.sub_alpha_(p.grad.data, lr)
                    p.zero_grad()

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )


# =====================================================================
# nll_gather forward/backward
# =====================================================================

class TestNllGather:
    @pytest.mark.xfail(reason="nll_gather not yet on tensor.py AutogradTensor; pending PR #21")
    def test_nll_gather_forward(self):
        """nll_gather picks -log_probs[i, target[i]]."""
        log_probs = AutogradTensor.from_data(
            [-0.5, -1.0, -2.0, -0.3, -0.7, -1.5], [2, 3], requires_grad=True
        )
        targets = AutogradTensor.from_data([1.0, 0.0], [2])
        nll = log_probs.nll_gather(targets)
        result = nll.tolist()
        assert approx(result[0], 1.0)   # -(-1.0)
        assert approx(result[1], 0.3)   # -(-0.3)

    @pytest.mark.xfail(reason="nll_gather not yet on tensor.py AutogradTensor; pending PR #21")
    def test_nll_gather_backward(self):
        """Gradient of nll_gather: -1 at target positions, 0 elsewhere."""
        log_probs = AutogradTensor.from_data(
            [-0.5, -1.0, -2.0, -0.3, -0.7, -1.5], [2, 3], requires_grad=True
        )
        targets = AutogradTensor.from_data([2.0, 1.0], [2])
        nll = log_probs.nll_gather(targets)
        loss = nll.mean()
        loss.backward()
        grad = log_probs.grad.tolist()
        assert approx(grad[2], -0.5)
        assert approx(grad[4], -0.5)


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_scalar_operations(self):
        """Autograd should handle scalar tensors."""
        a = AutogradTensor.from_data([5.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([3.0], [1], requires_grad=True)
        c = a * b + a
        c.backward()
        assert approx(a.grad.tolist()[0], 4.0)  # b + 1
        assert approx(b.grad.tolist()[0], 5.0)  # a

    def test_no_grad_no_gradient(self):
        """Tensors without requires_grad should not get gradients."""
        a = AutogradTensor.from_data([1.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([2.0], [1], requires_grad=False)
        c = a * b
        c.backward()
        assert a.grad is not None
        assert b.grad is None

    def test_detach_stops_gradient(self):
        """Detached tensors should stop gradient flow."""
        a = AutogradTensor.from_data([3.0], [1], requires_grad=True)
        b = a.detach()
        assert not b.requires_grad
        assert b.grad_fn is None

    def test_all_gradients_finite(self):
        """Complex expression should produce all finite gradients."""
        a = AutogradTensor.from_data([1.0, 2.0], [2], requires_grad=True)
        b = AutogradTensor.from_data([3.0, 4.0], [2], requires_grad=True)
        c = (a * b + a) / b
        loss = c.sum()
        loss.backward()
        assert all(math.isfinite(g) for g in a.grad.tolist())
        assert all(math.isfinite(g) for g in b.grad.tolist())

    def test_zero_grad_clears_gradient(self):
        """zero_grad should clear accumulated gradients."""
        a = AutogradTensor.from_data([2.0], [1], requires_grad=True)
        c = a * a
        c.backward()
        assert a.grad is not None
        a.zero_grad()
        assert a.grad is None

    def test_multiple_backward_calls(self):
        """Separate backward calls on new graphs should work."""
        a = AutogradTensor.from_data([3.0], [1], requires_grad=True)

        # First graph
        c1 = a * AutogradTensor.from_data([2.0], [1])
        c1.backward()
        grad1 = a.grad.tolist()[0]
        assert approx(grad1, 2.0)

        # Reset and do second graph
        a.zero_grad()
        c2 = a * AutogradTensor.from_data([5.0], [1])
        c2.backward()
        grad2 = a.grad.tolist()[0]
        assert approx(grad2, 5.0)


# =====================================================================
# nn.Linear integration (xfail until PR #23 merges)
# =====================================================================

class TestLinearIntegration:
    @pytest.mark.xfail(reason="nn.Linear uses plain lists; pending PR #23 for autograd integration")
    def test_linear_creates_autograd_weights(self):
        """Linear layer should use AutogradTensor for weights."""
        from cakelamp.nn import Linear
        fc = Linear(4, 3)
        assert isinstance(fc.weight.data, AutogradTensor)

    @pytest.mark.xfail(reason="nn.Linear uses plain lists; pending PR #23 for autograd integration")
    def test_linear_forward_with_autograd(self):
        """Linear forward should work with AutogradTensor input."""
        from cakelamp.nn import Linear
        fc = Linear(4, 3)
        x = AutogradTensor.from_data(list(range(8)), [2, 4])
        y = fc(x)
        assert y.shape == [2, 3]

    @pytest.mark.xfail(reason="nn.Linear uses plain lists; pending PR #23 for autograd integration")
    def test_linear_backward_produces_gradients(self):
        """Linear backward should produce gradients for weights."""
        from cakelamp.nn import Linear
        fc = Linear(4, 3)
        x = AutogradTensor.from_data([1.0] * 8, [2, 4])
        y = fc(x)
        loss = y.sum()
        loss.backward()
        assert fc.weight.data.grad is not None
