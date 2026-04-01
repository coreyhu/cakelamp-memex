"""Base class for autograd functions (computation graph nodes)."""

from __future__ import annotations
from typing import Any, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cakelamp.autograd.engine import AutogradTensor


class Function:
    """Base class for differentiable operations.

    Each Function records the forward computation and knows how to
    compute gradients in the backward pass.
    """

    def __init__(self):
        self.saved_tensors: list = []
        self.inputs: list[AutogradTensor] = []

    def save_for_backward(self, *tensors):
        """Save tensors needed for backward computation."""
        self.saved_tensors = list(tensors)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def backward(self, grad_output) -> Tuple:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply this function, building the computation graph."""
        from cakelamp.autograd.engine import AutogradTensor

        fn = cls()
        # Separate tensor inputs from other args
        tensor_inputs = [a for a in args if isinstance(a, AutogradTensor)]
        fn.inputs = tensor_inputs

        # Check if any input requires grad
        needs_grad = any(t.requires_grad for t in tensor_inputs)

        result_data = fn.forward(*args, **kwargs)

        if isinstance(result_data, tuple):
            results = []
            for rd in result_data:
                out = AutogradTensor(rd, requires_grad=needs_grad)
                if needs_grad:
                    out.grad_fn = fn
                results.append(out)
            return tuple(results)
        else:
            out = AutogradTensor(result_data, requires_grad=needs_grad)
            if needs_grad:
                out.grad_fn = fn
            return out
