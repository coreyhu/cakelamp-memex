"""CakeLamp Autograd: Reverse-mode automatic differentiation.

The computation graph is built in Python (like PyTorch). Each differentiable
op creates a Function (Node) in the graph that stores references to input
tensors and edges to parent nodes. Tensor.backward() does topological sort
+ reverse traversal.
"""

from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.autograd.function import Function

__all__ = ["AutogradTensor", "Function"]
