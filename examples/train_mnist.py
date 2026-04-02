#!/usr/bin/env python3
"""MNIST end-to-end training with CakeLamp.

Trains a 2-layer MLP (784 -> 128 -> 10) on MNIST to achieve >= 95% accuracy.
Architecture: Linear(784,128) -> ReLU -> Linear(128,10) -> CrossEntropyLoss
Optimizer: SGD with momentum.

Usage:
    python examples/train_mnist.py [--epochs N] [--batch-size N] [--lr FLOAT]
"""

from __future__ import annotations

import argparse
import sys
import time

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.data.mnist import load_mnist
from cakelamp.nn import Module, Linear, ReLU, CrossEntropyLoss
from cakelamp.optim import SGD


class MLP(Module):
    """Simple 2-layer MLP for MNIST classification.

    Architecture: Linear(784, 128) -> ReLU -> Linear(128, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def make_batch_tensors(batch_images, batch_labels):
    """Convert batch of images and labels to AutogradTensors.

    Parameters
    ----------
    batch_images : list[list[float]]
        Batch of flattened 784-pixel images.
    batch_labels : list[int]
        Batch of integer labels 0-9.

    Returns
    -------
    tuple[AutogradTensor, AutogradTensor]
        (images_tensor [N, 784], labels_tensor [N])
    """
    batch_size = len(batch_images)
    flat_data = []
    for img in batch_images:
        flat_data.extend(img)
    images = AutogradTensor(
        _C.tensor(flat_data, [batch_size, 784]), requires_grad=False
    )
    labels = AutogradTensor(
        _C.tensor([float(l) for l in batch_labels], [batch_size]),
        requires_grad=False,
    )
    return images, labels


def compute_accuracy(model, dataset, batch_size=256):
    """Compute accuracy on a dataset.

    Parameters
    ----------
    model : Module
        The model to evaluate.
    dataset : MNISTDataset
        Dataset to evaluate on.
    batch_size : int
        Batch size for evaluation.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    correct = 0
    total = 0

    for batch_images, batch_labels in dataset.batches(batch_size, shuffle=False):
        images, labels = make_batch_tensors(batch_images, batch_labels)
        logits = model(images)

        # Argmax along dim=1 to get predictions
        preds = logits.argmax(dim=1)
        pred_list = preds.tolist()
        for i in range(len(batch_labels)):
            pred_class = int(pred_list[i])
            if pred_class == batch_labels[i]:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def train(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.01,
    momentum: float = 0.9,
    data_dir: str = "./data/mnist",
    log_interval: int = 100,
) -> float:
    """Train the MNIST MLP.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    momentum : float
        SGD momentum.
    data_dir : str
        Directory for MNIST data.
    log_interval : int
        Print loss every N batches.

    Returns
    -------
    float
        Final test accuracy.
    """
    train_data, test_data = load_mnist(data_dir=data_dir)

    model = MLP()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    print(f"\nTraining MLP: 784 -> 128 -> 10")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}, Momentum: {momentum}")
    print(f"Parameters: {sum(1 for _ in model.parameters())}")
    print()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        batch_count = 0

        for batch_images, batch_labels in train_data.batches(batch_size, shuffle=True):
            images, labels = make_batch_tensors(batch_images, batch_labels)

            # Forward pass
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            if batch_count % log_interval == 0:
                avg_loss = running_loss / batch_count
                print(f"  Epoch {epoch}, Batch {batch_count}: loss = {avg_loss:.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / batch_count if batch_count > 0 else 0

        # Evaluate on test set
        test_acc = compute_accuracy(model, test_data)
        print(
            f"Epoch {epoch}/{epochs}: "
            f"loss = {avg_loss:.4f}, "
            f"test_acc = {test_acc:.4f} ({test_acc*100:.1f}%), "
            f"time = {epoch_time:.1f}s"
        )

        if test_acc >= 0.95:
            print(f"\n** Target accuracy (95%) reached at epoch {epoch}! **")
            return test_acc

    final_acc = compute_accuracy(model, test_data)
    print(f"\nFinal test accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
    if final_acc < 0.95:
        print("WARNING: Did not reach 95% target accuracy.")
    return final_acc


def main():
    parser = argparse.ArgumentParser(description="Train MNIST MLP with CakeLamp")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--data-dir", type=str, default="./data/mnist",
                        help="MNIST data dir")
    args = parser.parse_args()

    accuracy = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        data_dir=args.data_dir,
    )

    sys.exit(0 if accuracy >= 0.95 else 1)


if __name__ == "__main__":
    main()
