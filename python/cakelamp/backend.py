"""Pure Python backend for tensor operations.

This module provides the low-level tensor operations using pure Python/lists.
It mirrors the Rust cakelamp-core API. When PyO3 bindings are available,
this module can be replaced with the Rust backend for performance.
"""

import math
import random


def compute_strides(shape):
    """Compute contiguous (row-major) strides for a shape."""
    if not shape:
        return []
    ndim = len(shape)
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def flat_index(indices, strides, offset=0):
    """Compute flat index from multi-dimensional indices."""
    idx = offset
    for i, s in zip(indices, strides):
        idx += i * s
    return idx


def numel(shape):
    """Total number of elements."""
    result = 1
    for s in shape:
        result *= s
    return result


def zeros(n):
    return [0.0] * n


def ones(n):
    return [1.0] * n


def rand_data(n):
    return [random.random() for _ in range(n)]


def randn_data(n):
    """Generate n samples from standard normal using Box-Muller."""
    data = []
    for _ in range(n):
        u1 = max(random.random(), 1e-10)
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        data.append(z)
    return data


def broadcast_shape(a_shape, b_shape):
    """Compute broadcast output shape."""
    ndim = max(len(a_shape), len(b_shape))
    result = [0] * ndim
    for i in range(ndim):
        da = a_shape[i - ndim + len(a_shape)] if i >= ndim - len(a_shape) else 1
        db = b_shape[i - ndim + len(b_shape)] if i >= ndim - len(b_shape) else 1
        if da == db:
            result[i] = da
        elif da == 1:
            result[i] = db
        elif db == 1:
            result[i] = da
        else:
            raise ValueError(f"Cannot broadcast shapes {a_shape} and {b_shape}")
    return result


def broadcast_strides(shape, strides, target_shape):
    """Compute strides for broadcasting a tensor to target_shape."""
    ndim = len(target_shape)
    pad = ndim - len(shape)
    result = [0] * ndim
    for i in range(len(shape)):
        ni = i + pad
        if shape[i] == target_shape[ni]:
            result[ni] = strides[i]
        elif shape[i] == 1:
            result[ni] = 0
        else:
            raise ValueError(f"Cannot broadcast dim {i} from {shape[i]} to {target_shape[ni]}")
    return result


def binary_op(a_data, a_shape, a_strides, a_offset,
              b_data, b_shape, b_strides, b_offset, op):
    """Perform binary operation with broadcasting."""
    out_shape = broadcast_shape(a_shape, b_shape)
    a_bc_strides = broadcast_strides(a_shape, a_strides, out_shape)
    b_bc_strides = broadcast_strides(b_shape, b_strides, out_shape)

    n = numel(out_shape)
    ndim = len(out_shape)
    result = [0.0] * n

    if ndim == 0:
        result = [op(a_data[a_offset], b_data[b_offset])]
        return result, out_shape

    coord = [0] * ndim
    for flat_i in range(n):
        ai = a_offset
        bi = b_offset
        for d in range(ndim):
            ai += coord[d] * a_bc_strides[d]
            bi += coord[d] * b_bc_strides[d]
        result[flat_i] = op(a_data[ai], b_data[bi])

        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < out_shape[d]:
                break
            coord[d] = 0

    return result, out_shape


def unary_op(data, shape, strides, offset, op):
    """Perform unary operation."""
    n = numel(shape)
    if not shape:
        return [op(data[offset])], []

    result = [0.0] * n
    ndim = len(shape)
    coord = [0] * ndim
    for flat_i in range(n):
        idx = offset
        for d in range(ndim):
            idx += coord[d] * strides[d]
        result[flat_i] = op(data[idx])

        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < shape[d]:
                break
            coord[d] = 0

    return result, shape[:]


def to_contiguous(data, shape, strides, offset):
    """Return contiguous data from possibly non-contiguous storage."""
    expected_strides = compute_strides(shape)
    if strides == expected_strides and offset == 0 and len(data) == numel(shape):
        return data[:], shape[:]

    n = numel(shape)
    if not shape:
        return [data[offset]], []

    result = [0.0] * n
    ndim = len(shape)
    coord = [0] * ndim
    for flat_i in range(n):
        idx = offset
        for d in range(ndim):
            idx += coord[d] * strides[d]
        result[flat_i] = data[idx]

        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < shape[d]:
                break
            coord[d] = 0

    return result, shape[:]


def matmul(a_data, a_shape, b_data, b_shape):
    """Matrix multiplication (2D)."""
    m, k = a_shape
    k2, n = b_shape
    assert k == k2, f"Inner dims mismatch: {k} vs {k2}"

    result = [0.0] * (m * n)
    for i in range(m):
        for p in range(k):
            a_val = a_data[i * k + p]
            for j in range(n):
                result[i * n + j] += a_val * b_data[p * n + j]

    return result, [m, n]


def sum_dim(data, shape, dim, keepdim):
    """Sum along a dimension."""
    ndim = len(shape)
    dim_size = shape[dim]

    out_shape = list(shape)
    out_shape[dim] = 1
    out_n = numel(out_shape)
    result = [0.0] * out_n

    in_strides = compute_strides(shape)
    out_strides = compute_strides(out_shape)

    coord = [0] * ndim
    for _ in range(out_n):
        out_idx = 0
        for d in range(ndim):
            out_idx += coord[d] * out_strides[d]

        s = 0.0
        for k_val in range(dim_size):
            in_idx = 0
            for d in range(ndim):
                if d == dim:
                    in_idx += k_val * in_strides[d]
                else:
                    in_idx += coord[d] * in_strides[d]
            s += data[in_idx]
        result[out_idx] = s

        for d in range(ndim - 1, -1, -1):
            if d == dim:
                continue
            coord[d] += 1
            if coord[d] < out_shape[d]:
                break
            coord[d] = 0

    if keepdim:
        return result, out_shape
    else:
        squeezed = [s for i, s in enumerate(out_shape) if i != dim]
        if not squeezed:
            return result, []
        return result, squeezed


def argmax_dim(data, shape, dim):
    """Argmax along a dimension."""
    ndim = len(shape)
    dim_size = shape[dim]

    out_shape = [s for i, s in enumerate(shape) if i != dim]
    if not out_shape:
        best_idx = 0
        best_val = data[0]
        for i in range(1, len(data)):
            if data[i] > best_val:
                best_val = data[i]
                best_idx = i
        return [float(best_idx)], []

    out_n = numel(out_shape)
    result = [0.0] * out_n

    in_strides = compute_strides(shape)

    coord = [0] * len(out_shape)
    for out_i in range(out_n):
        best_val = float('-inf')
        best_idx = 0

        for k_val in range(dim_size):
            in_idx = 0
            out_d = 0
            for d in range(ndim):
                if d == dim:
                    in_idx += k_val * in_strides[d]
                else:
                    in_idx += coord[out_d] * in_strides[d]
                    out_d += 1
            if data[in_idx] > best_val:
                best_val = data[in_idx]
                best_idx = k_val
        result[out_i] = float(best_idx)

        for d in range(len(out_shape) - 1, -1, -1):
            coord[d] += 1
            if coord[d] < out_shape[d]:
                break
            coord[d] = 0

    return result, out_shape


def softmax_data(data, shape, dim):
    """Softmax along a dimension."""
    ndim = len(shape)
    n = numel(shape)
    strides = compute_strides(shape)

    out_shape = list(shape)
    out_shape[dim] = 1
    out_n = numel(out_shape)
    out_strides = compute_strides(out_shape)

    # Find max along dim
    maxes = [float('-inf')] * out_n
    coord = [0] * ndim
    for _ in range(n):
        in_idx = sum(coord[d] * strides[d] for d in range(ndim))
        out_idx = sum(coord[d] * out_strides[d] for d in range(ndim) if d != dim)
        if data[in_idx] > maxes[out_idx]:
            maxes[out_idx] = data[in_idx]
        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < shape[d]:
                break
            coord[d] = 0

    # Compute exp(x - max) and sum
    exp_sums = [0.0] * out_n
    result = [0.0] * n
    coord = [0] * ndim
    for flat_i in range(n):
        in_idx = sum(coord[d] * strides[d] for d in range(ndim))
        out_idx = sum(coord[d] * out_strides[d] for d in range(ndim) if d != dim)
        e = math.exp(data[in_idx] - maxes[out_idx])
        result[flat_i] = e
        exp_sums[out_idx] += e
        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < shape[d]:
                break
            coord[d] = 0

    # Normalize
    coord = [0] * ndim
    for flat_i in range(n):
        out_idx = sum(coord[d] * out_strides[d] for d in range(ndim) if d != dim)
        result[flat_i] /= exp_sums[out_idx]
        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < shape[d]:
                break
            coord[d] = 0

    return result, shape[:]
