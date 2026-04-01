"""CakeLamp: A lightweight tensor library with Rust backend and autograd."""

from cakelamp._core import Tensor as _CoreTensor
from cakelamp._core import (
    tensor as _tensor,
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
from cakelamp.autograd.engine import AutogradTensor
from cakelamp.autograd import function

import cakelamp.nn as nn
import cakelamp.optim as optim


def tensor(data, shape, requires_grad=False):
    """Create an AutogradTensor from data and shape."""
    core_t = _tensor(data, shape)
    return AutogradTensor(core_t, requires_grad=requires_grad)


# Convenience aliases
Tensor = AutogradTensor


__all__ = [
    "Tensor",
    "AutogradTensor",
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
    "nn",
    "optim",
]
