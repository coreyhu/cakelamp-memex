"""Stochastic Gradient Descent optimizer for CakeLamp.

Implements SGD with optional momentum and weight decay.
Updated to work with AutogradTensor parameters.
"""

from __future__ import annotations

from typing import Any, Dict

import cakelamp._core as _C
from cakelamp.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent (optionally with momentum).

    Parameters
    ----------
    params : iterable
        Parameters to optimise.
    lr : float
        Learning rate.
    momentum : float
        Momentum factor (default: 0).
    weight_decay : float
        L2 penalty (default: 0).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        """Perform a single SGD optimisation step."""
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data.tolist()
                p_data = p.data.tolist()
                n = len(g)

                # L2 weight decay
                if weight_decay != 0:
                    g = [gg + weight_decay * pp for gg, pp in zip(g, p_data)]

                # Momentum
                if momentum != 0:
                    state = self.state[id(p)]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g[:]
                    else:
                        buf = state["momentum_buffer"]
                        for j in range(n):
                            buf[j] = momentum * buf[j] + g[j]
                        g = buf

                # Update: p.data -= lr * grad
                new_data = [pp - lr * gg for pp, gg in zip(p_data, g)]
                p.data = _C.Tensor(new_data, p.data.shape)
