"""MNIST dataset loader.

Pure Python parsing of the IDX file format. Downloads MNIST data
from the web if not already cached locally.

IDX file format:
  - 4 bytes: magic number (big-endian)
  - 4 bytes: number of items (big-endian)
  - For images: 4 bytes rows, 4 bytes cols, then row*col bytes per image
  - For labels: 1 byte per label
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple


# MNIST mirror URLs
_MNIST_URLS = {
    "train-images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test-images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test-labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, dest: Path) -> None:
    """Download a file if it doesn't exist."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  -> saved to {dest}")


def _read_idx_images(path: Path) -> Tuple[List[List[float]], int, int, int]:
    """Read IDX image file and return (flat_images, n, rows, cols).

    Each image is a flat list of float32 values normalised to [0, 1].
    """
    with gzip.open(str(path), "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic}")
        n = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]

        images = []
        pixel_count = rows * cols
        for _ in range(n):
            raw = f.read(pixel_count)
            if len(raw) != pixel_count:
                raise ValueError("Unexpected EOF while reading images")
            # Normalise to [0, 1]
            img = [b / 255.0 for b in raw]
            images.append(img)

    return images, n, rows, cols


def _read_idx_labels(path: Path) -> List[int]:
    """Read IDX label file and return list of integer labels."""
    with gzip.open(str(path), "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic}")
        n = struct.unpack(">I", f.read(4))[0]

        raw = f.read(n)
        if len(raw) != n:
            raise ValueError("Unexpected EOF while reading labels")
        labels = list(raw)

    return labels


class MNISTDataset:
    """MNIST dataset with batching support.

    Parameters
    ----------
    images : list[list[float]]
        Each image is a flat list of 784 floats in [0, 1].
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
            Whether to shuffle the dataset each epoch.

        Yields
        ------
        tuple[list[list[float]], list[int]]
            Batch of images (flat lists) and labels.
        """
        import random

        indices = list(range(len(self.images)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_images = [self.images[i] for i in batch_idx]
            batch_labels = [self.labels[i] for i in batch_idx]
            yield batch_images, batch_labels


def load_mnist(
    data_dir: Optional[str] = None,
    download: bool = True,
) -> Tuple[MNISTDataset, MNISTDataset]:
    """Load the MNIST dataset.

    Parameters
    ----------
    data_dir : str, optional
        Directory to store/load data. Defaults to ``./data/mnist``.
    download : bool
        Whether to download data if not present (default: True).

    Returns
    -------
    tuple[MNISTDataset, MNISTDataset]
        (train_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = os.path.join(".", "data", "mnist")
    data_path = Path(data_dir)

    # File paths
    files = {
        "train-images": data_path / "train-images-idx3-ubyte.gz",
        "train-labels": data_path / "train-labels-idx1-ubyte.gz",
        "test-images": data_path / "t10k-images-idx3-ubyte.gz",
        "test-labels": data_path / "t10k-labels-idx1-ubyte.gz",
    }

    if download:
        for key, path in files.items():
            _download_file(_MNIST_URLS[key], path)

    # Parse IDX files
    train_images, n_train, rows, cols = _read_idx_images(files["train-images"])
    train_labels = _read_idx_labels(files["train-labels"])
    test_images, n_test, _, _ = _read_idx_images(files["test-images"])
    test_labels = _read_idx_labels(files["test-labels"])

    assert n_train == len(train_labels), "Train image/label count mismatch"
    assert n_test == len(test_labels), "Test image/label count mismatch"
    assert rows == 28 and cols == 28, f"Unexpected image size: {rows}x{cols}"

    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    print(f"MNIST loaded: {n_train} train, {n_test} test ({rows}x{cols})")
    return train_dataset, test_dataset
