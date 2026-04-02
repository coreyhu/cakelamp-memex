"""Tests for MNIST end-to-end training pipeline.

Verifies the complete training stack: autograd, nn modules, loss functions,
optimizer steps, and data loading. Uses synthetic data since real MNIST
may not be downloadable in CI.
"""

from __future__ import annotations

import math
import random

import pytest

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.nn import Module, Linear, ReLU, CrossEntropyLoss, Sequential
from cakelamp.nn.parameter import Parameter
from cakelamp.optim import SGD


# =====================================================================
# CrossEntropyLoss integration tests
# =====================================================================


class TestCrossEntropyLoss:
    def test_cross_entropy_forward(self):
        """CrossEntropyLoss should produce a reasonable loss value."""
        logits = AutogradTensor(
            _C.tensor([2.0, 1.0, 0.1, 0.1, 1.0, 2.0], [2, 3]),
            requires_grad=True,
        )
        targets = AutogradTensor(_C.tensor([0.0, 2.0], [2]))
        criterion = CrossEntropyLoss()
        loss = criterion(logits, targets)
        # Both samples have correct class with highest logit
        assert loss.item() < 1.0

    def test_cross_entropy_backward(self):
        """CrossEntropyLoss backward should produce gradients for logits."""
        logits = AutogradTensor(
            _C.tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3]),
            requires_grad=True,
        )
        targets = AutogradTensor(_C.tensor([2.0, 0.0], [2]))
        criterion = CrossEntropyLoss()
        loss = criterion(logits, targets)
        loss.backward()
        assert logits.grad is not None
        grad = logits.grad.tolist()
        assert len(grad) == 6
        assert all(math.isfinite(g) for g in grad)


# =====================================================================
# MLP training integration
# =====================================================================


class MLP(Module):
    """Simple 2-layer MLP for testing."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TestMLPTraining:
    def test_mlp_forward_backward(self):
        """MLP should do forward and backward without errors."""
        model = MLP(4, 8, 3)
        x = AutogradTensor(_C.tensor([1.0, 2.0, 3.0, 4.0] * 2, [2, 4]))
        targets = AutogradTensor(_C.tensor([0.0, 2.0], [2]))

        criterion = CrossEntropyLoss()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()

        params = list(model.parameters())
        assert len(params) == 4  # fc1.weight, fc1.bias, fc2.weight, fc2.bias
        for p in params:
            assert p.grad is not None, "All params should have gradients"

    def test_loss_decreases_with_sgd(self):
        """Training with SGD should decrease the loss over steps."""
        random.seed(42)
        model = MLP(4, 8, 3)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1)

        x = AutogradTensor(_C.tensor([1.0, 0.0, 0.0, 0.0] * 4, [4, 4]))
        targets = AutogradTensor(_C.tensor([0.0, 0.0, 0.0, 0.0], [4]))

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_training_on_separable_data(self):
        """Train on linearly separable synthetic data and check accuracy."""
        random.seed(123)

        n_samples = 40
        in_dim = 4
        n_classes = 2
        model = MLP(in_dim, 16, n_classes)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.05)

        # Generate separable data
        data = []
        labels = []
        for i in range(n_samples):
            if i < n_samples // 2:
                feat = [random.uniform(0.5, 1.0)] + [random.uniform(-0.1, 0.1) for _ in range(in_dim - 1)]
                data.extend(feat)
                labels.append(0.0)
            else:
                feat = [random.uniform(-1.0, -0.5)] + [random.uniform(-0.1, 0.1) for _ in range(in_dim - 1)]
                data.extend(feat)
                labels.append(1.0)

        x = AutogradTensor(_C.tensor(data, [n_samples, in_dim]))
        targets = AutogradTensor(_C.tensor(labels, [n_samples]))

        for step in range(50):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        # Check accuracy
        logits = model(x)
        preds = logits.argmax(dim=1)
        pred_list = preds.tolist()
        correct = sum(1 for i in range(n_samples) if int(pred_list[i]) == int(labels[i]))
        accuracy = correct / n_samples
        assert accuracy >= 0.8, f"Expected accuracy >= 80%, got {accuracy*100:.1f}%"


# =====================================================================
# MNIST Data Loader tests (synthetic IDX files)
# =====================================================================


class TestMNISTDataLoader:
    def test_dataset_batches(self):
        from cakelamp.data.mnist import MNISTDataset

        images = [[float(i)] * 4 for i in range(10)]
        labels = list(range(10))
        ds = MNISTDataset(images, labels)

        assert len(ds) == 10
        batches = list(ds.batches(batch_size=3, shuffle=False))
        assert len(batches) == 4  # 3+3+3+1
        all_labels = []
        for _, bl in batches:
            all_labels.extend(bl)
        assert sorted(all_labels) == list(range(10))


# =====================================================================
# Full pipeline smoke test (synthetic MNIST-like data)
# =====================================================================


class TestFullPipeline:
    def test_mnist_like_pipeline(self):
        """Full pipeline: create data -> MLP -> train -> loss decreases."""
        random.seed(0)

        n_train = 50
        in_dim = 16
        n_classes = 3

        # Generate separable clusters
        flat_data = []
        label_list = []
        for i in range(n_train):
            cls = i % n_classes
            label_list.append(float(cls))
            feat = [0.0] * in_dim
            feat[cls * (in_dim // n_classes)] = 1.0 + random.uniform(0, 0.3)
            for j in range(in_dim):
                feat[j] += random.uniform(-0.1, 0.1)
            flat_data.extend(feat)

        x = AutogradTensor(_C.tensor(flat_data, [n_train, in_dim]))
        targets = AutogradTensor(_C.tensor(label_list, [n_train]))

        model = MLP(in_dim, 32, n_classes)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1)

        initial_loss = None
        final_loss = None
        for epoch in range(30):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

        # Check accuracy
        logits = model(x)
        preds = logits.argmax(dim=1)
        pred_list = preds.tolist()
        correct = sum(
            1 for i in range(n_train) if int(pred_list[i]) == int(label_list[i])
        )
        accuracy = correct / n_train
        assert accuracy >= 0.6, f"Expected accuracy >= 60%, got {accuracy*100:.1f}%"
