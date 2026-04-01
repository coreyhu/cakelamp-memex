"""CakeLamp neural network modules, mirroring torch.nn."""

from cakelamp.nn.module import Module, Parameter
from cakelamp.nn.linear import Linear
from cakelamp.nn.activations import ReLU, Sigmoid, Tanh, Softmax, LogSoftmax
from cakelamp.nn.loss import MSELoss, CrossEntropyLoss, NLLLoss

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
]
