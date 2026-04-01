"""Dropout module for CakeLamp nn."""

from __future__ import annotations

import random

from cakelamp.nn.module import Module


class Dropout(Module):
    """During training, randomly zeroes elements with probability ``p``.

    Outputs are scaled by ``1/(1-p)`` so that the expected value is
    preserved.  During evaluation, Dropout is a no-op.

    Parameters
    ----------
    p : float
        Probability of an element being zeroed (default: ``0.5``).
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, input):
        if not self.training or self.p == 0.0:
            return input
        # Generate a mask and apply it.
        # This delegates to the tensor's dropout method.
        return input.dropout(self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"
