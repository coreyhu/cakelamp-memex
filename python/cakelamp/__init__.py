"""CakeLamp: A lightweight tensor library with automatic differentiation."""

from cakelamp.tensor import Tensor
from cakelamp import ops
from cakelamp import nn
from cakelamp import optim
from cakelamp import autograd

__all__ = ["Tensor", "ops", "nn", "optim", "autograd"]

# Convenience constructors
def zeros(*shape, requires_grad=False):
    return Tensor.zeros(list(shape), requires_grad=requires_grad)

def ones(*shape, requires_grad=False):
    return Tensor.ones(list(shape), requires_grad=requires_grad)

def rand(*shape, requires_grad=False):
    return Tensor.rand(list(shape), requires_grad=requires_grad)

def randn(*shape, requires_grad=False):
    return Tensor.randn(list(shape), requires_grad=requires_grad)

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)

def eye(n, requires_grad=False):
    return Tensor.eye(n, requires_grad=requires_grad)

def arange(start, end=None, step=1.0):
    if end is None:
        end = start
        start = 0.0
    return Tensor.arange(float(start), float(end), float(step))
