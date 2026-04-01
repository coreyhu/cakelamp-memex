"""Loss function modules for CakeLamp nn.

Provides common loss functions needed for training neural networks.
"""

from __future__ import annotations

from cakelamp.nn.module import Module


class MSELoss(Module):
    """Mean Squared Error loss.

    ``loss = mean((input - target)^2)``

    Parameters
    ----------
    reduction : str
        Specifies the reduction to apply: ``'mean'`` (default),
        ``'sum'``, or ``'none'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Invalid reduction mode: {reduction}. "
                "Must be 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction

    def forward(self, input, target):
        diff = input.sub(target)
        sq = diff.mul(diff)
        if self.reduction == "mean":
            return sq.mean()
        elif self.reduction == "sum":
            return sq.sum()
        else:
            return sq

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class NLLLoss(Module):
    """Negative Log Likelihood loss.

    Expects log-probabilities as input (e.g., from LogSoftmax).

    ``loss = -sum(log_probs[i, target[i]]) / N``

    Parameters
    ----------
    reduction : str
        ``'mean'`` (default), ``'sum'``, or ``'none'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Invalid reduction mode: {reduction}. "
                "Must be 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction

    def forward(self, input, target):
        """Compute NLL loss.

        Parameters
        ----------
        input : Tensor
            Log-probabilities of shape ``(N, C)``.
        target : Tensor
            Target class indices of shape ``(N,)`` with integer values.
        """
        # Gather the log-probability for each target class.
        # nll = -input[i, target[i]] for each sample i
        nll = input.nll_gather(target)

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class CrossEntropyLoss(Module):
    """Cross Entropy loss.

    Combines LogSoftmax and NLLLoss in one step for numerical stability.

    ``loss = -sum(log_softmax(input)[i, target[i]]) / N``

    Parameters
    ----------
    reduction : str
        ``'mean'`` (default), ``'sum'``, or ``'none'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Invalid reduction mode: {reduction}. "
                "Must be 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction

    def forward(self, input, target):
        """Compute cross entropy loss.

        Parameters
        ----------
        input : Tensor
            Raw logits of shape ``(N, C)``.
        target : Tensor
            Target class indices of shape ``(N,)`` with integer values.
        """
        log_probs = input.log_softmax(dim=1)
        nll = log_probs.nll_gather(target)

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"
