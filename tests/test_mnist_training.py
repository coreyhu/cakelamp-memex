"""Tests for MNIST training pipeline.

Uses synthetic data to test the full training pipeline without
requiring network access to download actual MNIST data.
"""

from __future__ import annotations

import math
import random
import pytest

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.nn.linear import Linear
from cakelamp.nn.activations import ReLU
from cakelamp.nn.loss import CrossEntropyLoss, MSELoss
from cakelamp.nn.containers import Sequential
from cakelamp.optim.sgd import SGD
from cakelamp.optim.adam import Adam
from cakelamp.data.mnist import make_batches


# =====================================================================
# Synthetic data generators
# =====================================================================


def make_separable_data(n_per_class: int = 20, n_classes: int = 3, dim: int = 4):
    """Create linearly separable synthetic data."""
    images = []
    labels = []
    for c in range(n_classes):
        for _ in range(n_per_class):
            # Each class has a different center
            center = [0.0] * dim
            center[c % dim] = 1.0
            # Add noise
            img = [center[j] + random.gauss(0, 0.1) for j in range(dim)]
            images.append(img)
            labels.append(c)
    # Shuffle
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    return list(images), list(labels)


def make_mnist_like_data(n: int = 50, dim: int = 784, n_classes: int = 10):
    """Create MNIST-like random data for pipeline testing."""
    images = []
    labels = []
    for i in range(n):
        img = [random.random() for _ in range(dim)]
        labels.append(i % n_classes)
        images.append(img)
    return images, labels


# =====================================================================
# Tests: make_batches utility
# =====================================================================


class TestMakeBatches:
    def test_basic(self):
        images = [[1.0, 2.0]] * 10
        labels = list(range(10))
        batches = make_batches(images, labels, batch_size=3)
        assert len(batches) == 4  # ceil(10/3) = 4
        assert len(batches[0][0]) == 3
        assert len(batches[-1][0]) == 1  # remainder

    def test_exact_division(self):
        images = [[1.0]] * 6
        labels = [0] * 6
        batches = make_batches(images, labels, batch_size=3)
        assert len(batches) == 2
        assert len(batches[0][0]) == 3
        assert len(batches[1][0]) == 3

    def test_single_batch(self):
        images = [[1.0, 2.0]] * 5
        labels = [0] * 5
        batches = make_batches(images, labels, batch_size=10)
        assert len(batches) == 1
        assert len(batches[0][0]) == 5


# =====================================================================
# Tests: MLP on separable data
# =====================================================================


class TestMLPSeparable:
    def test_sgd_training_converges(self):
        """SGD should achieve high accuracy on linearly separable data."""
        random.seed(42)
        images, labels = make_separable_data(n_per_class=30, n_classes=3, dim=4)

        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 3),
        )
        opt = SGD(model.parameters(), lr=0.05)
        loss_fn = CrossEntropyLoss()

        losses = []
        for epoch in range(30):
            batches = make_batches(images, labels, batch_size=15)
            epoch_loss = 0.0
            for batch_imgs, batch_lbls in batches:
                bs = len(batch_imgs)
                flat = []
                for img in batch_imgs:
                    flat.extend(img)
                x = AutogradTensor(_C.Tensor(flat, [bs, 4]))
                t = AutogradTensor(_C.Tensor([float(l) for l in batch_lbls], [bs]))

                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, t)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(batches))

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_adam_training_converges(self):
        """Adam should converge on separable data."""
        random.seed(42)
        images, labels = make_separable_data(n_per_class=20, n_classes=3, dim=4)

        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 3),
        )
        opt = Adam(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()

        first_loss = None
        last_loss = None
        for epoch in range(20):
            batches = make_batches(images, labels, batch_size=10)
            epoch_loss = 0.0
            for batch_imgs, batch_lbls in batches:
                bs = len(batch_imgs)
                flat = []
                for img in batch_imgs:
                    flat.extend(img)
                x = AutogradTensor(_C.Tensor(flat, [bs, 4]))
                t = AutogradTensor(_C.Tensor([float(l) for l in batch_lbls], [bs]))

                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, t)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(batches)
            if first_loss is None:
                first_loss = avg
            last_loss = avg

        assert last_loss < first_loss

    def test_accuracy_above_threshold(self):
        """After training, accuracy on separable data should be high."""
        random.seed(123)
        images, labels = make_separable_data(n_per_class=40, n_classes=3, dim=4)
        n_train = int(len(images) * 0.8)
        train_imgs, test_imgs = images[:n_train], images[n_train:]
        train_lbls, test_lbls = labels[:n_train], labels[n_train:]

        model = Sequential(
            Linear(4, 16),
            ReLU(),
            Linear(16, 3),
        )
        opt = Adam(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()

        for epoch in range(40):
            batches = make_batches(train_imgs, train_lbls, batch_size=16)
            for batch_imgs, batch_lbls in batches:
                bs = len(batch_imgs)
                flat = []
                for img in batch_imgs:
                    flat.extend(img)
                x = AutogradTensor(_C.Tensor(flat, [bs, 4]))
                t = AutogradTensor(_C.Tensor([float(l) for l in batch_lbls], [bs]))

                opt.zero_grad()
                loss = loss_fn(model(x), t)
                loss.backward()
                opt.step()

        # Evaluate
        model.eval()
        correct = 0
        for img, lbl in zip(test_imgs, test_lbls):
            x = AutogradTensor(_C.Tensor(img, [1, 4]))
            out = model(x)
            pred = out.argmax(dim=1).tolist()[0]
            if int(pred) == lbl:
                correct += 1

        accuracy = correct / len(test_lbls)
        assert accuracy >= 0.70, f"Accuracy {accuracy:.2%} below 70% threshold"


# =====================================================================
# Tests: Full pipeline smoke test
# =====================================================================


class TestPipelineSmoke:
    def test_mnist_like_pipeline(self):
        """Smoke test: full pipeline with MNIST-like dimensions."""
        random.seed(99)
        images, labels = make_mnist_like_data(n=50, dim=784, n_classes=10)

        model = Sequential(
            Linear(784, 32),
            ReLU(),
            Linear(32, 10),
        )
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()

        batches = make_batches(images, labels, batch_size=10)

        # Just verify it runs without errors
        for batch_imgs, batch_lbls in batches:
            bs = len(batch_imgs)
            flat = []
            for img in batch_imgs:
                flat.extend(img)
            x = AutogradTensor(_C.Tensor(flat, [bs, 784]))
            t = AutogradTensor(_C.Tensor([float(l) for l in batch_lbls], [bs]))

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, t)
            loss.backward()
            opt.step()

        # Verify loss is finite
        assert math.isfinite(loss.item())

    def test_eval_mode_no_dropout(self):
        """Model should work in eval mode."""
        from cakelamp.nn.dropout import Dropout

        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Dropout(0.5),
            Linear(8, 3),
        )
        model.eval()

        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0, 4.0], [1, 4]))
        out = model(x)
        assert out.shape == [1, 3]
        # Run twice - should give same result in eval mode
        out2 = model(x)
        assert out.tolist() == out2.tolist()
