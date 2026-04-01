"""Neural network modules mirroring torch.nn."""

from cakelamp.tensor import Tensor
import math


class Parameter(Tensor):
    """A tensor that is a module parameter (requires_grad=True by default)."""

    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data.tolist(), requires_grad=True)
        else:
            super().__init__(data, requires_grad=True)


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return all parameters (recursive)."""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(self, prefix=''):
        """Return all parameters with names."""
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        for name, module in self._modules.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(full_name)

    def __setattr__(self, name, value):
        # Only register on first assignment (during __init__)
        if name not in ('_modules', '_parameters', 'training') and hasattr(self, '_parameters'):
            if isinstance(value, Parameter):
                self._parameters[name] = value
            elif isinstance(value, Module):
                self._modules[name] = value
        object.__setattr__(self, name, value)

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None


class Linear(Module):
    """Fully connected layer: y = x @ W^T + b"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Kaiming uniform initialization
        k = 1.0 / math.sqrt(in_features)
        w_data = []
        import random
        for _ in range(out_features * in_features):
            w_data.append(random.uniform(-k, k))
        self.weight = Parameter(
            Tensor._make(w_data, [out_features, in_features])
        )
        if bias:
            b_data = [random.uniform(-k, k) for _ in range(out_features)]
            self.bias = Parameter(
                Tensor._make(b_data, [out_features])
            )
        else:
            self.bias = None

    def forward(self, x):
        # x: (batch, in_features), weight: (out_features, in_features)
        # output = x @ weight.T + bias
        out = x.mm(self.weight.transpose(0, 1))
        if self.bias is not None:
            # bias is (out_features,), broadcasting handles (batch, out_features) + (out_features,)
            out = out + self.bias
        return out


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class MSELoss(Module):
    def forward(self, prediction, target):
        diff = prediction - target
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    """Cross-entropy loss from logits.

    input: (N, C) logits
    target: (N,) class indices (as float tensor)
    """
    def forward(self, input, target):
        # Log softmax
        log_probs = input.log_softmax(dim=1)

        # Gather the log probs for correct classes
        batch_size = input._shape[0]
        num_classes = input._shape[1]
        target_data = target._contiguous_data()

        # Manual gather: for each sample, pick log_prob[target_class]
        log_probs_data = log_probs._contiguous_data()
        loss_sum = 0.0
        for i in range(batch_size):
            c = int(target_data[i])
            loss_sum += log_probs_data[i * num_classes + c]

        # NLL loss = -mean(log_probs for correct classes)
        loss_val = -loss_sum / batch_size
        result = Tensor._make([loss_val], [], input.requires_grad)

        if input.requires_grad:
            # Store info for backward
            result.grad_fn = _CrossEntropyBackward(input, target, log_probs)

        return result


class _CrossEntropyBackward:
    """Backward for cross-entropy loss."""

    def __init__(self, input_tensor, target, softmax_output):
        self.input = input_tensor
        self.target = target
        self.softmax = softmax_output

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        import math
        batch_size = self.input._shape[0]
        num_classes = self.input._shape[1]

        # Softmax output
        sm_data = []
        log_probs_data = self.softmax._contiguous_data()
        for x in log_probs_data:
            sm_data.append(math.exp(x))

        target_data = self.target._contiguous_data()

        # grad = softmax - one_hot(target)
        grad_data = sm_data[:]
        for i in range(batch_size):
            c = int(target_data[i])
            grad_data[i * num_classes + c] -= 1.0

        # Scale by 1/batch_size and grad_output
        go = grad_output.item() if grad_output.numel == 1 else 1.0
        scale = go / batch_size
        grad_data = [x * scale for x in grad_data]

        return [Tensor._make(grad_data, list(self.input._shape))]


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        import random
        data = x._contiguous_data()
        mask = [0.0 if random.random() < self.p else 1.0 / (1.0 - self.p)
                for _ in data]
        result_data = [d * m for d, m in zip(data, mask)]
        return Tensor._make(result_data, list(x._shape), x.requires_grad)


class BatchNorm1d(Module):
    """Simplified 1D batch normalization."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(Tensor.ones([num_features]))
        self.bias = Parameter(Tensor.zeros([num_features]))
        self.running_mean = Tensor.zeros([num_features])
        self.running_var = Tensor.ones([num_features])

    def forward(self, x):
        # x: (N, C)
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = ((x - mean.expand(list(x._shape))) ** 2).mean(dim=0, keepdim=True)
        else:
            mean = self.running_mean.unsqueeze(0)
            var = self.running_var.unsqueeze(0)

        x_norm = (x - mean.expand(list(x._shape))) / (var.expand(list(x._shape)) + self.eps) ** 0.5
        out = x_norm * self.weight.unsqueeze(0).expand(list(x._shape)) + self.bias.unsqueeze(0).expand(list(x._shape))
        return out
