"""Autograd Function nodes for backward computation."""

import math


class Function:
    """Base class for autograd function nodes."""

    def saved_tensors(self):
        """Return the input tensors this node depends on."""
        raise NotImplementedError

    def backward(self, grad_output):
        """Compute gradients w.r.t. inputs. Returns list of gradients."""
        raise NotImplementedError


def _reduce_grad(grad, target_shape):
    """Sum grad to match target_shape (undo broadcasting).

    When a smaller tensor is broadcast to a larger shape for an op,
    the gradient must be summed back along the broadcast dimensions.
    """
    from cakelamp.autograd.tensor import AutogradTensor

    if not isinstance(grad, AutogradTensor):
        return grad

    grad_shape = list(grad._shape)
    target = list(target_shape)

    if grad_shape == target:
        return grad

    # Handle scalar target
    if not target:
        return grad.sum()

    # Sum away leading dimensions
    while len(grad_shape) > len(target):
        grad = grad.sum(dim=0)
        grad_shape = list(grad._shape)

    # Sum along dimensions that were broadcast (size 1 in target)
    for i in range(len(target)):
        if target[i] == 1 and grad_shape[i] != 1:
            grad = grad.sum(dim=i, keepdim=True)

    return grad


class AddBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def saved_tensors(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        return [
            _reduce_grad(grad_output, self.a._shape),
            _reduce_grad(grad_output, self.b._shape),
        ]


class SubBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def saved_tensors(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        return [
            _reduce_grad(grad_output, self.a._shape),
            _reduce_grad(-grad_output, self.b._shape),
        ]


class MulBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def saved_tensors(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        return [
            _reduce_grad(grad_output * self.b.detach(), self.a._shape),
            _reduce_grad(grad_output * self.a.detach(), self.b._shape),
        ]


class DivBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def saved_tensors(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        b_det = self.b.detach()
        return [
            _reduce_grad(grad_output / b_det, self.a._shape),
            _reduce_grad(-grad_output * self.a.detach() / (b_det * b_det), self.b._shape),
        ]


class NegBackward(Function):
    def __init__(self, a):
        self.a = a

    def saved_tensors(self):
        return [self.a]

    def backward(self, grad_output):
        return [-grad_output]


class PowBackward(Function):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent

    def saved_tensors(self):
        return [self.base, self.exponent]

    def backward(self, grad_output):
        b = self.base.detach()
        e = self.exponent.detach()
        grad_base = _reduce_grad(
            grad_output * e * (b ** (e - 1)),
            self.base._shape,
        )
        grad_exp = None
        if self.exponent.requires_grad:
            grad_exp = _reduce_grad(
                grad_output * (b ** e) * b.log(),
                self.exponent._shape,
            )
        return [grad_base, grad_exp]


class ExpBackward(Function):
    def __init__(self, input_t, output_t):
        self.input = input_t
        self.output = output_t

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        return [grad_output * self.output.detach()]


class LogBackward(Function):
    def __init__(self, input_t):
        self.input = input_t

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        return [grad_output / self.input.detach()]


class ReluBackward(Function):
    def __init__(self, input_t):
        self.input = input_t

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        mask_data = [1.0 if x > 0 else 0.0 for x in self.input._data]
        mask = AutogradTensor._make(mask_data, list(self.input._shape))
        return [grad_output * mask]


class SigmoidBackward(Function):
    def __init__(self, input_t, output_t):
        self.input = input_t
        self.output = output_t

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        s = self.output.detach()
        ones = AutogradTensor.ones_like(s)
        return [grad_output * s * (ones - s)]


class TanhBackward(Function):
    def __init__(self, input_t, output_t):
        self.input = input_t
        self.output = output_t

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        t = self.output.detach()
        ones = AutogradTensor.ones_like(t)
        return [grad_output * (ones - t * t)]


class SumBackward(Function):
    def __init__(self, input_t, dim=None, keepdim=False):
        self.input = input_t
        self.dim = dim
        self.keepdim = keepdim

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        if self.dim is None:
            return [AutogradTensor.ones(list(self.input._shape)) * grad_output]
        else:
            if not self.keepdim:
                g = grad_output.unsqueeze(self.dim)
            else:
                g = grad_output
            return [g.expand(list(self.input._shape))]


class MeanBackward(Function):
    def __init__(self, input_t, dim=None, keepdim=False):
        self.input = input_t
        self.dim = dim
        self.keepdim = keepdim

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        if self.dim is None:
            n = self.input.numel
            return [AutogradTensor.ones(list(self.input._shape)) * grad_output / n]
        else:
            n = self.input._shape[self.dim]
            if not self.keepdim:
                g = grad_output.unsqueeze(self.dim)
            else:
                g = grad_output
            return [g.expand(list(self.input._shape)) / n]


class MatmulBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def saved_tensors(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        # d(A @ B)/dA = grad @ B^T, d(A @ B)/dB = A^T @ grad
        return [
            grad_output.mm(self.b.detach().transpose(0, 1)),
            self.a.detach().transpose(0, 1).mm(grad_output),
        ]


class ReshapeBackward(Function):
    def __init__(self, input_t):
        self.input = input_t

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        return [grad_output.reshape(list(self.input._shape))]


class TransposeBackward(Function):
    def __init__(self, input_t, dim0, dim1):
        self.input = input_t
        self.dim0 = dim0
        self.dim1 = dim1

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        return [grad_output.transpose(self.dim0, self.dim1)]


class LogSoftmaxBackward(Function):
    def __init__(self, input_t, softmax_output, dim):
        self.input = input_t
        self.softmax = softmax_output
        self.dim = dim

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        sm = self.softmax
        grad_sum = grad_output.sum(dim=self.dim, keepdim=True)
        return [grad_output - sm * grad_sum.expand(list(self.input._shape))]


class CrossEntropyBackward(Function):
    """Backward for cross-entropy loss (combined log_softmax + nll)."""

    def __init__(self, input_t, target, softmax_output):
        self.input = input_t
        self.target = target
        self.softmax = softmax_output

    def saved_tensors(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        batch_size = self.input._shape[0]
        num_classes = self.input._shape[1]

        sm_data = list(self.softmax._data)
        target_data = self.target._data

        grad_data = sm_data[:]
        for i in range(batch_size):
            c = int(target_data[i])
            grad_data[i * num_classes + c] -= 1.0

        go = grad_output.item() if grad_output.numel == 1 else 1.0
        scale = go / batch_size
        grad_data = [x * scale for x in grad_data]

        return [AutogradTensor._make(grad_data, list(self.input._shape))]
