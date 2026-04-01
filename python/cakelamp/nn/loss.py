"""Loss functions."""

from __future__ import annotations
import cakelamp._core as _C
from cakelamp.autograd.engine import AutogradTensor
from cakelamp.nn.module import Module


class MSELoss(Module):
    """Mean Squared Error loss."""

    def forward(self, prediction: AutogradTensor, target: AutogradTensor) -> AutogradTensor:
        diff = prediction - target
        return (diff * diff).mean()


class NLLLoss(Module):
    """Negative Log Likelihood loss.

    Expects log-probabilities as input (from log_softmax).
    target: 1D tensor of class indices (as floats).
    """

    def forward(self, log_probs: AutogradTensor, target: AutogradTensor) -> AutogradTensor:
        # log_probs: (N, C), target: (N,) with class indices
        batch_size = log_probs.shape[0]
        num_classes = log_probs.shape[1]

        # Create one-hot from target
        target_oh = AutogradTensor(
            _C.one_hot(target.data, num_classes),
            requires_grad=False,
        )

        # NLL = -sum(one_hot * log_probs) / batch_size
        selected = log_probs * target_oh
        loss = -selected.sum() / AutogradTensor(_C.Tensor.scalar(float(batch_size)))
        return loss


class CrossEntropyLoss(Module):
    """Cross Entropy loss = log_softmax + NLL.

    Expects raw logits as input.
    target: 1D tensor of class indices (as floats).
    """

    def forward(self, logits: AutogradTensor, target: AutogradTensor) -> AutogradTensor:
        log_probs = logits.log_softmax(dim=1)
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        target_oh = AutogradTensor(
            _C.one_hot(target.data, num_classes),
            requires_grad=False,
        )

        selected = log_probs * target_oh
        loss = -selected.sum() / AutogradTensor(_C.Tensor.scalar(float(batch_size)))
        return loss
