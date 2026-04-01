"""Stochastic Gradient Descent optimizer for CakeLamp.

Implements SGD with optional momentum, dampening, weight decay,
and Nesterov momentum.  Mirrors torch.optim.SGD.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

from cakelamp.optim.optimizer import Optimizer


class SGD(Optimizer):
    r"""Stochastic Gradient Descent (optionally with momentum).

    .. math::
        v_{t+1} &= \mu \, v_t + (1 - \tau) \, g_t  \\
        p_{t+1} &= p_t - \text{lr} \, v_{t+1}

    where :math:`g_t` is the gradient (possibly with weight decay),
    :math:`\mu` is momentum, and :math:`\tau` is dampening.

    With Nesterov momentum the update becomes:

    .. math::
        p_{t+1} = p_t - \text{lr} \, (g_t + \mu \, v_{t+1})

    Parameters
    ----------
    params : iterable
        Parameters to optimise.
    lr : float
        Learning rate.  **Required.**
    momentum : float
        Momentum factor (default: ``0``).
    dampening : float
        Dampening for momentum (default: ``0``).
    weight_decay : float
        L2 penalty (default: ``0``).
    nesterov : bool
        Enable Nesterov momentum (default: ``False``).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )

        defaults: Dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        """Perform a single SGD optimisation step."""
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # L2 weight decay: add weight_decay * p.data to gradient.
                if weight_decay != 0:
                    grad = grad.add(p.data.mul(weight_decay))

                # Momentum.
                if momentum != 0:
                    state = self.state[id(p)]

                    if "momentum_buffer" not in state:
                        # First step: initialise buffer as a clone of grad.
                        buf = grad.clone()
                        state["momentum_buffer"] = buf
                    else:
                        buf = state["momentum_buffer"]
                        # buf = momentum * buf + (1 - dampening) * grad
                        buf.mul_(momentum).add_(grad.mul(1.0 - dampening))

                    if nesterov:
                        # effective_grad = grad + momentum * buf
                        grad = grad.add(buf.mul(momentum))
                    else:
                        grad = buf

                # Update: p.data -= lr * grad
                p.data.add_(grad.mul(-lr))
