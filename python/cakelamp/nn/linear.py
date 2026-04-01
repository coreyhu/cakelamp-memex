"""Linear (fully connected) layer."""

from __future__ import annotations
import math
import cakelamp._core as _C
from cakelamp.autograd.engine import AutogradTensor
from cakelamp.nn.module import Module, Parameter


class Linear(Module):
    """Fully connected linear layer: y = x @ W^T + b

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If True, adds a learnable bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
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
            self.bias = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x: AutogradTensor) -> AutogradTensor:
        # x: (batch, in_features), weight: (out_features, in_features)
        # output = x @ weight.T + bias
        output = x @ self.weight.t()
        if self.bias is not None:
            output = output + self.bias
        return output
