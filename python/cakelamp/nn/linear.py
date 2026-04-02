"""Linear (fully connected) layer for CakeLamp nn.

Applies an affine transformation: y = x @ W^T + b.
"""

from __future__ import annotations

import math
from typing import Optional

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
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

        # Kaiming uniform initialization
        bound = 1.0 / math.sqrt(in_features)
        weight_data = _C.randn([out_features, in_features])
        weight_data.mul_scalar_(bound)
        self.weight = Parameter(weight_data)

        if bias:
            bias_data = _C.randn([out_features])
            bias_data.mul_scalar_(bound)
            self.bias: Optional[Parameter] = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, input: AutogradTensor) -> AutogradTensor:
        """Apply the linear transformation."""
        # x @ W^T + b
        output = input @ self.weight.t()
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
