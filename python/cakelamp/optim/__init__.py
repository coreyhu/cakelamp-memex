"""CakeLamp optimizers module.

Provides optimization algorithms for training neural networks.
Mirrors the torch.optim API.
"""

from cakelamp.optim.optimizer import Optimizer
from cakelamp.optim.sgd import SGD
from cakelamp.optim.adam import Adam

__all__ = ["Optimizer", "SGD", "Adam"]
