"""Functional interface for tensor operations."""

from cakelamp.tensor import Tensor


def relu(x):
    return x.relu()

def sigmoid(x):
    return x.sigmoid()

def tanh(x):
    return x.tanh()

def exp(x):
    return x.exp()

def log(x):
    return x.log()

def softmax(x, dim=-1):
    return x.softmax(dim=dim)

def log_softmax(x, dim=-1):
    return x.log_softmax(dim=dim)

def matmul(a, b):
    return a.mm(b)

def cross_entropy(input, target):
    from cakelamp.nn import CrossEntropyLoss
    return CrossEntropyLoss()(input, target)

def mse_loss(prediction, target):
    from cakelamp.nn import MSELoss
    return MSELoss()(prediction, target)
