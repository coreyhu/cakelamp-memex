"""Loss function modules for CakeLamp nn.

Provides common loss functions needed for training neural networks.
All loss functions are autograd-compatible — they work with AutogradTensor
and support backward pass for gradient computation.
"""

from __future__ import annotations

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.nn.module import Module


class MSELoss(Module):
    """Mean Squared Error loss.

    ``loss = mean((input - target)^2)``
    """

    def forward(self, input, target):
        diff = input - target
        return (diff * diff).mean()

    def extra_repr(self) -> str:
        return ""


class NLLLoss(Module):
    """Negative Log Likelihood loss.

    Expects log-probabilities as input (e.g., from LogSoftmax).
    target: 1D tensor of class indices (as floats).
    """

    def forward(self, log_probs, target):
        batch_size = log_probs.shape[0]
        num_classes = log_probs.shape[1]

        target_data = target.data if isinstance(target, AutogradTensor) else target
        target_oh = AutogradTensor(
            _C.one_hot(target_data, num_classes),
            requires_grad=False,
        )

        selected = log_probs * target_oh
        loss = -selected.sum() / AutogradTensor.from_scalar(float(batch_size))
        return loss


class CrossEntropyLoss(Module):
    """Cross Entropy loss = log_softmax + NLL.

    Expects raw logits as input.
    target: 1D tensor of class indices (as floats).
    """

    def forward(self, logits, target):
        log_probs = logits.log_softmax(dim=1)
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        target_data = target.data if isinstance(target, AutogradTensor) else target
        target_oh = AutogradTensor(
            _C.one_hot(target_data, num_classes),
            requires_grad=False,
        )

        selected = log_probs * target_oh
        loss = -selected.sum() / AutogradTensor.from_scalar(float(batch_size))
        return loss
