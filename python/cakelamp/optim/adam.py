"""Adam optimizer."""

from __future__ import annotations
import math
import cakelamp._core as _C
from cakelamp.autograd.engine import AutogradTensor


class Adam:
    """Adam optimizer (Kingma & Ba, 2015).

    Parameters
    ----------
    params : iterable of Parameter
        Parameters to optimize.
    lr : float
        Learning rate (default: 1e-3).
    betas : tuple of float
        Coefficients for running averages (default: (0.9, 0.999)).
    eps : float
        Term for numerical stability (default: 1e-8).
    weight_decay : float
        Decoupled weight decay (default: 0).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [None] * len(self.params)  # first moment
        self.v = [None] * len(self.params)  # second moment

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.step_count += 1
        t = self.step_count

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.data.tolist()
            p_data = p.data.tolist()
            n = len(g)

            # Initialize moments
            if self.m[i] is None:
                self.m[i] = [0.0] * n
                self.v[i] = [0.0] * n

            m = self.m[i]
            v = self.v[i]

            # Decoupled weight decay
            if self.weight_decay != 0:
                p_data = [pp * (1 - self.lr * self.weight_decay) for pp in p_data]

            for j in range(n):
                # Update moments
                m[j] = self.beta1 * m[j] + (1 - self.beta1) * g[j]
                v[j] = self.beta2 * v[j] + (1 - self.beta2) * g[j] * g[j]

                # Bias correction
                m_hat = m[j] / (1 - self.beta1 ** t)
                v_hat = v[j] / (1 - self.beta2 ** t)

                # Update parameter
                p_data[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            p.data = _C.Tensor(p_data, p.data.shape)
