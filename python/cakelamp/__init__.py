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

from cakelamp import autograd, nn, optim
from cakelamp.autograd import AutogradTensor, Function

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
    "autograd",
    "nn",
    "optim",
    "AutogradTensor",
    "Function",
]
