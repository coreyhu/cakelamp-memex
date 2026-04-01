"""Parameter class for CakeLamp nn module.

A Parameter is a Tensor that is automatically registered as a trainable
parameter when assigned as a Module attribute.
"""

from __future__ import annotations

from typing import Optional


class Parameter:
    """A tensor that is treated as a module parameter.

    When a :class:`Parameter` is assigned as an attribute of a
    :class:`Module`, it is automatically added to the module's list of
    parameters and will be returned by :meth:`Module.parameters`.

    Parameters
    ----------
    data : tensor
        The parameter tensor.  ``requires_grad`` is set to ``True``
        automatically.
    requires_grad : bool
        Whether the parameter requires gradient computation
        (default: ``True``).
    """

    def __init__(self, data, requires_grad: bool = True) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional[object] = None

    def zero_grad(self) -> None:
        """Zero out the gradient."""
        self.grad = None

    def __repr__(self) -> str:
        return f"Parameter(data={self.data}, requires_grad={self.requires_grad})"
