"""CakeLamp neural network modules.

Provides the building blocks for defining and training neural networks.
Mirrors the torch.nn API.
"""

from cakelamp.nn.module import Module
from cakelamp.nn.parameter import Parameter
from cakelamp.nn.linear import Linear
from cakelamp.nn.activations import ReLU, Sigmoid, Tanh, Softmax, LogSoftmax
from cakelamp.nn.loss import MSELoss, CrossEntropyLoss, NLLLoss
from cakelamp.nn.containers import Sequential
from cakelamp.nn.dropout import Dropout

__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LogSoftmax",
    "MSELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "Sequential",
    "Dropout",
]
