"""Adam optimizer for CakeLamp.

Implements the Adam algorithm (Adaptive Moment Estimation) with optional
weight decay (AdamW-style decoupled weight decay).  Mirrors torch.optim.Adam.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

from cakelamp.optim.optimizer import Optimizer


class Adam(Optimizer):
    r"""Adam optimiser (Kingma & Ba, 2015).

    .. math::
        m_t &= \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t \\
        v_t &= \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2 \\
        \hat{m}_t &= m_t / (1 - \beta_1^t) \\
        \hat{v}_t &= v_t / (1 - \beta_2^t) \\
        p_t &= p_{t-1} - \text{lr} \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)

    Parameters
    ----------
    params : iterable
        Parameters to optimise.
    lr : float
        Learning rate (default: ``1e-3``).
    betas : tuple[float, float]
        Coefficients for computing running averages of gradient and its
        square (default: ``(0.9, 0.999)``).
    eps : float
        Term added to the denominator for numerical stability
        (default: ``1e-8``).
    weight_decay : float
        Decoupled weight decay (AdamW-style) (default: ``0``).
    amsgrad : bool
        Whether to use the AMSGrad variant (default: ``False``).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
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
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        """Perform a single Adam optimisation step."""
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[id(p)]

                # State initialisation (first step).
                if len(state) == 0:
                    state["step"] = 0
                    # First moment (mean of gradients).
                    state["exp_avg"] = grad.zeros_like()
                    # Second moment (mean of squared gradients).
                    state["exp_avg_sq"] = grad.zeros_like()
                    if amsgrad:
                        state["max_exp_avg_sq"] = grad.zeros_like()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay (AdamW).
                if weight_decay != 0:
                    p.data.add_(p.data.mul(-weight_decay * lr))

                # Update biased first moment estimate.
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad.mul(1.0 - beta1))

                # Update biased second raw moment estimate.
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).add_(grad.mul(grad).mul(1.0 - beta2))

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    # max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq)
                    max_exp_avg_sq = max_exp_avg_sq.maximum(exp_avg_sq)
                    state["max_exp_avg_sq"] = max_exp_avg_sq
                    # Use max for denominator.
                    denom = max_exp_avg_sq.sqrt().add(eps)
                else:
                    denom = exp_avg_sq.sqrt().add(eps)

                # Bias correction.
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                # sqrt of bias_correction2 applied to denominator.
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                # p.data -= step_size * exp_avg / (denom / sqrt(bias_correction2))
                # Equivalent to: p.data -= step_size * exp_avg / denom * sqrt(bias_correction2)
                p.data.add_(
                    exp_avg.div(denom.div(bias_correction2_sqrt)).mul(-step_size)
                )
