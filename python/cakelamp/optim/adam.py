"""Adam optimizer for CakeLamp.

Implements the Adam algorithm (Adaptive Moment Estimation) with optional
weight decay (AdamW-style decoupled weight decay).
Updated to work with AutogradTensor parameters.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import cakelamp._core as _C
from cakelamp.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimiser (Kingma & Ba, 2015).

    Parameters
    ----------
    params : iterable
        Parameters to optimise.
    lr : float
        Learning rate (default: 1e-3).
    betas : tuple[float, float]
        Coefficients for running averages (default: (0.9, 0.999)).
    eps : float
        Term for numerical stability (default: 1e-8).
    weight_decay : float
        Decoupled weight decay (default: 0).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        """Perform a single Adam optimisation step."""
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data.tolist()
                p_data = p.data.tolist()
                n = len(g)

                state = self.state[id(p)]

                # State initialisation
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = [0.0] * n  # first moment
                    state["v"] = [0.0] * n  # second moment

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                # Decoupled weight decay
                if weight_decay != 0:
                    p_data = [pp * (1 - lr * weight_decay) for pp in p_data]

                for j in range(n):
                    m[j] = beta1 * m[j] + (1 - beta1) * g[j]
                    v[j] = beta2 * v[j] + (1 - beta2) * g[j] * g[j]

                    # Bias correction
                    m_hat = m[j] / (1 - beta1 ** t)
                    v_hat = v[j] / (1 - beta2 ** t)

                    p_data[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

                p.data = _C.Tensor(p_data, p.data.shape)
