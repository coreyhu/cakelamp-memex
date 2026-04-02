"""Tests for MNIST data loading utilities.

Tests the IDX file parser and dataset batching without downloading
actual MNIST data. Uses synthetic IDX files.
"""

from __future__ import annotations

import gzip
import os
import struct
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from cakelamp.data.mnist import (
    MNISTDataset,
    _read_idx_images,
    _read_idx_labels,
)
from pathlib import Path


def _make_idx_images(path: Path, images: list, rows: int = 2, cols: int = 2) -> None:
    """Write a synthetic IDX images file (gzipped).

    Parameters
    ----------
    path : Path
        Output .gz file path.
    images : list[list[int]]
        List of images, each a flat list of pixel values (0-255).
    rows, cols : int
        Image dimensions.
    """
    n = len(images)
    with gzip.open(str(path), "wb") as f:
        f.write(struct.pack(">I", 2051))  # magic
        f.write(struct.pack(">I", n))
        f.write(struct.pack(">I", rows))
        f.write(struct.pack(">I", cols))
        for img in images:
            f.write(bytes(img))


def _make_idx_labels(path: Path, labels: list) -> None:
    """Write a synthetic IDX labels file (gzipped)."""
    n = len(labels)
    with gzip.open(str(path), "wb") as f:
        f.write(struct.pack(">I", 2049))  # magic
        f.write(struct.pack(">I", n))
        f.write(bytes(labels))


# =====================================================================
# IDX parser tests
# =====================================================================

class TestIDXParser:
    def test_read_images(self, tmp_path):
        """Read a synthetic 2x2 image file."""
        img1 = [0, 128, 255, 64]
        img2 = [255, 0, 128, 32]
        path = tmp_path / "images.gz"
        _make_idx_images(path, [img1, img2], rows=2, cols=2)

        images, n, rows, cols = _read_idx_images(path)
        assert n == 2
        assert rows == 2
        assert cols == 2
        assert len(images) == 2
        assert len(images[0]) == 4
        # Check normalisation to [0, 1]
        assert images[0][0] == 0.0 / 255.0
        assert abs(images[0][1] - 128.0 / 255.0) < 1e-6
        assert images[0][2] == 1.0  # 255/255

    def test_read_labels(self, tmp_path):
        """Read a synthetic labels file."""
        labels = [0, 5, 9, 3]
        path = tmp_path / "labels.gz"
        _make_idx_labels(path, labels)

        result = _read_idx_labels(path)
        assert result == [0, 5, 9, 3]

    def test_invalid_image_magic(self, tmp_path):
        """Wrong magic number should raise."""
        path = tmp_path / "bad_images.gz"
        with gzip.open(str(path), "wb") as f:
            f.write(struct.pack(">I", 9999))
            f.write(struct.pack(">I", 0))
            f.write(struct.pack(">I", 1))
            f.write(struct.pack(">I", 1))
        with pytest.raises(ValueError, match="Invalid magic"):
            _read_idx_images(path)

    def test_invalid_label_magic(self, tmp_path):
        """Wrong magic number should raise."""
        path = tmp_path / "bad_labels.gz"
        with gzip.open(str(path), "wb") as f:
            f.write(struct.pack(">I", 9999))
            f.write(struct.pack(">I", 0))
        with pytest.raises(ValueError, match="Invalid magic"):
            _read_idx_labels(path)

    def test_single_image(self, tmp_path):
        """Single image file."""
        img = [100, 200]
        path = tmp_path / "single.gz"
        _make_idx_images(path, [img], rows=1, cols=2)

        images, n, rows, cols = _read_idx_images(path)
        assert n == 1
        assert rows == 1
        assert cols == 2
        assert abs(images[0][0] - 100 / 255.0) < 1e-6

    def test_larger_images(self, tmp_path):
        """28x28 images (real MNIST size)."""
        import random
        random.seed(42)
        n_imgs = 5
        size = 28 * 28
        imgs = [[random.randint(0, 255) for _ in range(size)] for _ in range(n_imgs)]

        path = tmp_path / "large.gz"
        _make_idx_images(path, imgs, rows=28, cols=28)

        images, n, rows, cols = _read_idx_images(path)
        assert n == 5
        assert rows == 28
        assert cols == 28
        assert len(images[0]) == 784
        # All values in [0, 1]
        for img in images:
            assert all(0.0 <= v <= 1.0 for v in img)


# =====================================================================
# MNISTDataset tests
# =====================================================================

class TestMNISTDataset:
    def test_len(self):
        images = [[0.0] * 4 for _ in range(10)]
        labels = list(range(10))
        ds = MNISTDataset(images, labels)
        assert len(ds) == 10

    def test_batches_no_shuffle(self):
        images = [[float(i)] * 4 for i in range(10)]
        labels = list(range(10))
        ds = MNISTDataset(images, labels)

        batches = list(ds.batches(batch_size=3, shuffle=False))
        assert len(batches) == 4  # 3+3+3+1
        assert len(batches[0][0]) == 3
        assert len(batches[0][1]) == 3
        assert len(batches[-1][0]) == 1  # last batch has 1 item

    def test_batches_cover_all_data(self):
        n = 20
        images = [[0.0] * 4 for _ in range(n)]
        labels = list(range(n))
        ds = MNISTDataset(images, labels)

        all_labels = []
        for _, batch_labels in ds.batches(batch_size=7, shuffle=False):
            all_labels.extend(batch_labels)
        assert sorted(all_labels) == list(range(n))

    def test_batches_shuffle(self):
        """Shuffled batches should still cover all data."""
        import random
        random.seed(42)
        n = 50
        images = [[float(i)] * 4 for i in range(n)]
        labels = list(range(n))
        ds = MNISTDataset(images, labels)

        all_labels = []
        for _, batch_labels in ds.batches(batch_size=10, shuffle=True):
            all_labels.extend(batch_labels)
        assert sorted(all_labels) == list(range(n))

    def test_exact_batch_size(self):
        """When dataset size is divisible by batch size."""
        images = [[0.0] * 4 for _ in range(12)]
        labels = list(range(12))
        ds = MNISTDataset(images, labels)

        batches = list(ds.batches(batch_size=4, shuffle=False))
        assert len(batches) == 3
        assert all(len(b[0]) == 4 for b in batches)

    def test_single_batch(self):
        """Batch size larger than dataset."""
        images = [[0.0] * 4 for _ in range(5)]
        labels = list(range(5))
        ds = MNISTDataset(images, labels)

        batches = list(ds.batches(batch_size=100, shuffle=False))
        assert len(batches) == 1
        assert len(batches[0][0]) == 5


# =====================================================================
# Integration test (synthetic data, no download)
# =====================================================================

class TestIntegration:
    def test_end_to_end_data_flow(self, tmp_path):
        """Parse IDX files -> MNISTDataset -> iterate batches."""
        import random
        random.seed(0)

        # Create synthetic 4x4 images
        n_train = 20
        rows, cols = 4, 4
        imgs = [[random.randint(0, 255) for _ in range(rows * cols)]
                for _ in range(n_train)]
        labels = [random.randint(0, 9) for _ in range(n_train)]

        img_path = tmp_path / "images.gz"
        lbl_path = tmp_path / "labels.gz"
        _make_idx_images(img_path, imgs, rows=rows, cols=cols)
        _make_idx_labels(lbl_path, labels)

        # Parse
        parsed_imgs, n, r, c = _read_idx_images(img_path)
        parsed_lbls = _read_idx_labels(lbl_path)
        assert n == n_train
        assert r == rows
        assert c == cols

        # Create dataset
        ds = MNISTDataset(parsed_imgs, parsed_lbls)
        assert len(ds) == n_train

        # Iterate batches
        total = 0
        for batch_imgs, batch_lbls in ds.batches(batch_size=7, shuffle=False):
            total += len(batch_imgs)
            assert len(batch_imgs) == len(batch_lbls)
            for img in batch_imgs:
                assert len(img) == rows * cols
                assert all(0.0 <= v <= 1.0 for v in img)
        assert total == n_train
