"""MNIST dataset loader for CakeLamp.

Pure-Python IDX format parser. Downloads MNIST from the web if not cached.
Returns data as flat Python lists ready for use with _C.Tensor.
"""

from __future__ import annotations

import gzip
import os
import random
import struct
import urllib.request
from typing import Tuple, List

# Mirror URLs for MNIST (original site is sometimes unreliable)
_MNIST_URLS = {
    "train-images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test-images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test-labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, path: str) -> None:
    """Download a file if it doesn't exist."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def _read_idx_images(path) -> Tuple[List[List[float]], int, int, int]:
    """Read IDX image file, return (images_as_flat_float_lists, n, rows, cols)."""
    with gzip.open(str(path), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic}")
        data = f.read()

    images = []
    img_size = rows * cols
    for i in range(n):
        offset = i * img_size
        pixels = data[offset : offset + img_size]
        # Normalize to [0, 1]
        images.append([p / 255.0 for p in pixels])

    return images, n, rows, cols


def _read_idx_labels(path) -> List[int]:
    """Read IDX label file, return list of int labels."""
    with gzip.open(str(path), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic}")
        data = f.read()

    return [b for b in data[:n]]


class MNISTDataset:
    """MNIST dataset with batching support.

    Parameters
    ----------
    images : list[list[float]]
        Each image is a flat list of floats in [0, 1].
    labels : list[int]
        Integer labels 0-9.
    """

    def __init__(self, images: List[List[float]], labels: List[int]) -> None:
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def batches(self, batch_size: int, shuffle: bool = True):
        """Yield (batch_images, batch_labels) tuples.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        shuffle : bool
            Whether to shuffle indices each time.

        Yields
        ------
        tuple[list[list[float]], list[int]]
        """
        indices = list(range(len(self.images)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_images = [self.images[i] for i in batch_idx]
            batch_labels = [self.labels[i] for i in batch_idx]
            yield batch_images, batch_labels


def load_mnist(
    data_dir: str = "./data/mnist",
    download: bool = True,
    train: bool = True,
    limit: int = 0,
) -> Tuple[List[List[float]], List[int]]:
    """Load MNIST dataset.

    Parameters
    ----------
    data_dir : str
        Directory to cache downloaded files.
    download : bool
        If True, download MNIST if not found locally.
    train : bool
        If True, load training set (60k); otherwise test set (10k).
    limit : int
        If > 0, only return first ``limit`` samples.

    Returns
    -------
    images : list of list of float
        Each image is a flat list of 784 floats in [0, 1].
    labels : list of int
        Integer labels 0-9.
    """
    prefix = "train" if train else "test"
    img_key = f"{prefix}-images"
    lbl_key = f"{prefix}-labels"

    img_path = os.path.join(data_dir, f"{img_key}.gz")
    lbl_path = os.path.join(data_dir, f"{lbl_key}.gz")

    if download:
        _download(_MNIST_URLS[img_key], img_path)
        _download(_MNIST_URLS[lbl_key], lbl_path)

    images, n, rows, cols = _read_idx_images(img_path)
    labels = _read_idx_labels(lbl_path)

    assert len(images) == len(labels), f"Mismatch: {len(images)} images, {len(labels)} labels"

    if limit > 0:
        images = images[:limit]
        labels = labels[:limit]

    return images, labels


def make_batches(
    images: List[List[float]],
    labels: List[int],
    batch_size: int,
) -> List[Tuple[List[List[float]], List[int]]]:
    """Split images and labels into batches.

    Returns list of (batch_images, batch_labels) tuples.
    """
    batches = []
    n = len(images)
    for i in range(0, n, batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_lbls = labels[i : i + batch_size]
        batches.append((batch_imgs, batch_lbls))
    return batches
