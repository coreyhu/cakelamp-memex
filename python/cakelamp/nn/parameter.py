"""Parameter class for CakeLamp nn module.

A Parameter is an AutogradTensor that is automatically registered
as a trainable parameter when assigned as a Module attribute.
"""

from __future__ import annotations
from cakelamp.autograd.tensor import AutogradTensor


class Parameter(AutogradTensor):
    """A tensor that is treated as a module parameter.

    When a Parameter is assigned as an attribute of a Module, it is
    automatically added to the module's list of parameters.

    Parameters
    ----------
    data : tensor
        The parameter tensor data (_core.Tensor).
    requires_grad : bool
        Whether the parameter requires gradient computation (default: True).
    """

    def __init__(self, data, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad)
