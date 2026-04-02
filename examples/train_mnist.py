"""Train a 2-layer MLP on MNIST using CakeLamp.

Usage:
    uv run python examples/train_mnist.py [--epochs N] [--lr LR] [--batch-size BS]

This demonstrates the full CakeLamp training pipeline:
    Tensor backend → AutogradTensor → nn.Linear → CrossEntropyLoss → SGD/Adam
"""

from __future__ import annotations

import argparse
import random
import time

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.nn.linear import Linear
from cakelamp.nn.activations import ReLU
from cakelamp.nn.loss import CrossEntropyLoss
from cakelamp.nn.containers import Sequential
from cakelamp.optim.sgd import SGD
from cakelamp.optim.adam import Adam
from cakelamp.data.mnist import load_mnist, make_batches


def build_model(hidden: int = 128) -> Sequential:
    """Build a 2-layer MLP: 784 -> hidden -> 10."""
    return Sequential(
        Linear(784, hidden),
        ReLU(),
        Linear(hidden, 10),
    )


def train_epoch(model, optimizer, loss_fn, batches):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_imgs, batch_lbls in batches:
        bs = len(batch_imgs)
        # Flatten batch into tensor
        flat_data = []
        for img in batch_imgs:
            flat_data.extend(img)
        x = AutogradTensor(_C.Tensor(flat_data, [bs, 784]))
        t = AutogradTensor(_C.Tensor([float(l) for l in batch_lbls], [bs]))

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, t)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, batches):
    """Evaluate accuracy on batches, return accuracy fraction."""
    model.eval()
    correct = 0
    total = 0

    for batch_imgs, batch_lbls in batches:
        bs = len(batch_imgs)
        flat_data = []
        for img in batch_imgs:
            flat_data.extend(img)
        x = AutogradTensor(_C.Tensor(flat_data, [bs, 784]))

        out = model(x)
        # Argmax along dim 1
        preds = out.argmax(dim=1)
        pred_list = preds.tolist()

        for pred, label in zip(pred_list, batch_lbls):
            if int(pred) == label:
                correct += 1
            total += 1

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--data-dir", default="./data/mnist")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit samples (0=all)")
    args = parser.parse_args()

    print("Loading MNIST...")
    train_images, train_labels = load_mnist(
        data_dir=args.data_dir, train=True, limit=args.limit
    )
    test_images, test_labels = load_mnist(
        data_dir=args.data_dir, train=False, limit=args.limit
    )
    print(f"  Train: {len(train_images)} samples")
    print(f"  Test:  {len(test_images)} samples")

    # Shuffle training data
    combined = list(zip(train_images, train_labels))
    random.shuffle(combined)
    train_images, train_labels = zip(*combined)
    train_images, train_labels = list(train_images), list(train_labels)

    train_batches = make_batches(train_images, train_labels, args.batch_size)
    test_batches = make_batches(test_images, test_labels, args.batch_size)

    print(f"\nBuilding model (784 -> {args.hidden} -> 10)...")
    model = build_model(hidden=args.hidden)

    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr)

    loss_fn = CrossEntropyLoss()
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_epoch(model, optimizer, loss_fn, train_batches)
        t1 = time.time()
        acc = evaluate(model, test_batches)
        t2 = time.time()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Test Acc: {acc:.2%} | "
            f"Train: {t1-t0:.1f}s | Eval: {t2-t1:.1f}s"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
