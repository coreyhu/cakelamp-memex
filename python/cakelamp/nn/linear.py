"""Linear (fully connected) layer for CakeLamp nn.

Applies an affine transformation: y = x @ W^T + b.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from cakelamp.nn.module import Module
from cakelamp.nn.parameter import Parameter


class Linear(Module):
    """Fully connected linear layer.

    Applies: ``y = x @ weight.T + bias``

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If ``True`` (default), add an additive bias.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialise weight with Kaiming uniform (like PyTorch default).
        # For a Linear layer: fan_in = in_features
        bound = 1.0 / math.sqrt(in_features)
        weight_data = self._uniform(out_features * in_features, -bound, bound)
        self.weight = Parameter(weight_data)

        if bias:
            bias_data = self._uniform(out_features, -bound, bound)
            self.bias: Optional[Parameter] = Parameter(bias_data)
        else:
            self.bias = None

    @staticmethod
    def _uniform(n: int, low: float, high: float) -> list:
        """Generate n uniform random values in [low, high).

        Returns a plain list.  When the real Tensor backend is available,
        this will be replaced with Tensor.uniform_.
        """
        return [random.uniform(low, high) for _ in range(n)]

    def forward(self, input):
        """Apply the linear transformation.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape ``(*, in_features)``.

        Returns
        -------
        Tensor
            Output of shape ``(*, out_features)``.
        """
        # x @ W^T
        output = input.matmul(self.weight.data.t())
        if self.bias is not None:
            output = output.add(self.bias.data)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
