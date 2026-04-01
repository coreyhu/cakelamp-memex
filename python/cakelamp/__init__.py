"""CakeLamp: A lightweight tensor library with Rust backend."""

from cakelamp._core import (
    Tensor,
    tensor,
    zeros,
    ones,
    full,
    rand,
    randn,
    matmul,
    relu,
    sigmoid,
    tanh,
    softmax,
    log_softmax,
    one_hot,
    arange,
)

__all__ = [
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "full",
    "rand",
    "randn",
    "matmul",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "one_hot",
    "arange",
]
