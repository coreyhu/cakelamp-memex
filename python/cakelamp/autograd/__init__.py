"""Autograd: Reverse-mode automatic differentiation for CakeLamp.

The computation graph is built in Python. Each differentiable op creates
a Function (Node) that records the inputs and knows how to compute
the gradient. Tensor.backward() performs topological sort + reverse
traversal to compute all gradients.
"""

from cakelamp.autograd.engine import no_grad, GradMode
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.autograd.function import Function

__all__ = [
    "AutogradTensor",
    "Function",
    "no_grad",
    "GradMode",
]
