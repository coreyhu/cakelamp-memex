"""Activation function modules for CakeLamp nn.

Each activation is a stateless Module that wraps the corresponding
tensor operation.
"""

from __future__ import annotations

from cakelamp.nn.module import Module


class ReLU(Module):
    """Applies the rectified linear unit function element-wise.

    ``ReLU(x) = max(0, x)``
    """

    def forward(self, input):
        return input.relu()

    def extra_repr(self) -> str:
        return ""


class Sigmoid(Module):
    """Applies the sigmoid function element-wise.

    ``Sigmoid(x) = 1 / (1 + exp(-x))``
    """

    def forward(self, input):
        return input.sigmoid()


class Tanh(Module):
    """Applies the hyperbolic tangent function element-wise."""

    def forward(self, input):
        return input.tanh()


class Softmax(Module):
    """Applies the softmax function along a dimension.

    Parameters
    ----------
    dim : int
        Dimension along which softmax will be computed.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.softmax(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSoftmax(Module):
    """Applies log(softmax(x)) along a dimension.

    Parameters
    ----------
    dim : int
        Dimension along which log-softmax will be computed.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.log_softmax(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
