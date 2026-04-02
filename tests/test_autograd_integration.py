"""Integration tests: autograd + nn modules + optimizers.

Verifies that the autograd engine works end-to-end with Linear layers,
loss functions, and optimizers for real training loops.
"""

from __future__ import annotations

import math
import pytest

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.nn.parameter import Parameter
from cakelamp.nn.linear import Linear
from cakelamp.nn.activations import ReLU, Sigmoid, Tanh
from cakelamp.nn.loss import MSELoss, CrossEntropyLoss
from cakelamp.nn.containers import Sequential
from cakelamp.optim.sgd import SGD
from cakelamp.optim.adam import Adam


def approx_equal(a, b, tol=1e-4):
    if isinstance(a, list) and isinstance(b, list):
        return all(abs(x - y) < tol for x, y in zip(a, b))
    return abs(a - b) < tol


# =====================================================================
# Linear layer with autograd
# =====================================================================


class TestLinearAutograd:
    def test_linear_backward(self):
        layer = Linear(3, 2)
        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0], [1, 3]), requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad is not None

    def test_linear_grad_shape(self):
        layer = Linear(4, 2)
        x = AutogradTensor(_C.Tensor([1.0] * 8, [2, 4]), requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad.shape == [2, 4]
        # Bias grad may be [2] or [1, 2] depending on broadcast reduction
        assert layer.bias.grad.numel() == 2


# =====================================================================
# Loss function with autograd
# =====================================================================


class TestLossAutograd:
    def test_mse_backward(self):
        pred = AutogradTensor(_C.Tensor([1.0, 2.0], [2]), requires_grad=True)
        target = AutogradTensor(_C.Tensor([1.5, 2.5], [2]))
        loss_fn = MSELoss()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        # MSE grad = 2*(pred - target)/n
        assert approx_equal(pred.grad.tolist(), [-0.5, -0.5])

    def test_cross_entropy_backward(self):
        logits = AutogradTensor(
            _C.Tensor([2.0, 1.0, 0.1], [1, 3]), requires_grad=True
        )
        target = AutogradTensor(_C.Tensor([0.0], [1]))
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == [1, 3]


# =====================================================================
# Optimizer with autograd
# =====================================================================


class TestSGDAutograd:
    def test_sgd_step(self):
        layer = Linear(2, 1, bias=False)
        x = AutogradTensor(_C.Tensor([1.0, 2.0], [1, 2]))
        target = AutogradTensor(_C.Tensor([3.0], [1, 1]))

        opt = SGD(layer.parameters(), lr=0.01)
        loss_fn = MSELoss()

        initial_w = layer.weight.data.tolist()
        loss = loss_fn(layer(x), target)
        loss.backward()
        opt.step()

        new_w = layer.weight.data.tolist()
        assert new_w != initial_w


class TestAdamAutograd:
    def test_adam_step(self):
        layer = Linear(2, 1, bias=False)
        x = AutogradTensor(_C.Tensor([1.0, 2.0], [1, 2]))
        target = AutogradTensor(_C.Tensor([3.0], [1, 1]))

        opt = Adam(layer.parameters(), lr=0.01)
        loss_fn = MSELoss()

        initial_w = layer.weight.data.tolist()
        loss = loss_fn(layer(x), target)
        loss.backward()
        opt.step()

        new_w = layer.weight.data.tolist()
        assert new_w != initial_w


# =====================================================================
# MLP training loop
# =====================================================================


class TestMLPTraining:
    def test_mlp_forward_backward(self):
        """Full forward + backward through a small MLP."""
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1),
        )
        x = AutogradTensor(_C.Tensor([1.0, 2.0], [1, 2]))
        target = AutogradTensor(_C.Tensor([5.0], [1, 1]))

        loss_fn = MSELoss()
        loss = loss_fn(model(x), target)
        loss.backward()

        for p in model.parameters():
            assert p.grad is not None

    def test_training_loss_decreases(self):
        """Training loop should reduce loss over iterations."""
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1),
        )
        x = AutogradTensor(_C.Tensor([1.0, 2.0], [1, 2]))
        target = AutogradTensor(_C.Tensor([5.0], [1, 1]))

        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()

        losses = []
        for _ in range(20):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            losses.append(loss.item())
            loss.backward()
            opt.step()

        assert losses[-1] < losses[0]

    def test_multi_class_training(self):
        """Training with CrossEntropyLoss should reduce loss."""
        model = Sequential(
            Linear(3, 4),
            ReLU(),
            Linear(4, 3),
        )
        x = AutogradTensor(_C.Tensor([1.0, 0.5, -0.5], [1, 3]))
        target = AutogradTensor(_C.Tensor([1.0], [1]))  # class 1

        opt = Adam(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()

        losses = []
        for _ in range(30):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            losses.append(loss.item())
            loss.backward()
            opt.step()

        assert losses[-1] < losses[0]

    def test_adam_mlp_training(self):
        """Adam optimizer with MLP should converge."""
        model = Sequential(
            Linear(2, 8),
            ReLU(),
            Linear(8, 1),
        )
        x = AutogradTensor(_C.Tensor([1.0, 2.0], [1, 2]))
        target = AutogradTensor(_C.Tensor([3.0], [1, 1]))

        opt = Adam(model.parameters(), lr=0.01)
        loss_fn = MSELoss()

        initial_loss = None
        final_loss = None
        for i in range(50):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            if i == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
            loss.backward()
            opt.step()

        assert final_loss < initial_loss
