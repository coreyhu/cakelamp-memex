"""Autograd engine: gradient computation and context management."""


class GradMode:
    """Global gradient computation mode."""
    _enabled = True

    @classmethod
    def is_enabled(cls):
        return cls._enabled

    @classmethod
    def set_enabled(cls, mode):
        cls._enabled = mode


class no_grad:
    """Context manager to disable gradient computation."""

    def __enter__(self):
        self.prev = GradMode.is_enabled()
        GradMode.set_enabled(False)
        return self

    def __exit__(self, *args):
        GradMode.set_enabled(self.prev)


def backward(tensor, gradient=None):
    """Run backward pass from a tensor through the computation graph.

    Parameters
    ----------
    tensor : AutogradTensor
        The output tensor to differentiate.
    gradient : AutogradTensor, optional
        External gradient. Defaults to ones for scalar tensors.
    """
    from cakelamp.autograd.tensor import AutogradTensor

    if gradient is None:
        if tensor.numel != 1:
            raise RuntimeError(
                "backward() requires scalar output or explicit gradient"
            )
        gradient = AutogradTensor.ones_like(tensor)

    # Topological sort (reverse postorder)
    topo_order = []
    visited = set()

    def _topo_sort(t):
        if id(t) in visited:
            return
        visited.add(id(t))
        if t._grad_fn is not None:
            for parent in t._grad_fn.saved_tensors():
                if isinstance(parent, AutogradTensor) and parent.requires_grad:
                    _topo_sort(parent)
            topo_order.append(t)

    _topo_sort(tensor)
    topo_order.reverse()

    # Set output gradient
    grads = {id(tensor): gradient}

    # Backpropagate
    for t in topo_order:
        grad = grads.get(id(t))
        if grad is None or t._grad_fn is None:
            continue

        input_grads = t._grad_fn.backward(grad)
        parents = t._grad_fn.saved_tensors()

        for parent, parent_grad in zip(parents, input_grads):
            if not isinstance(parent, AutogradTensor):
                continue
            if not parent.requires_grad:
                continue
            if parent_grad is None:
                continue

            if id(parent) in grads:
                grads[id(parent)] = grads[id(parent)] + parent_grad
            else:
                grads[id(parent)] = parent_grad

    # Assign gradients to leaf tensors and all tensors with requires_grad
    all_tensors = set()

    def _collect(t):
        all_tensors.add(id(t))
        if t._grad_fn is not None:
            for parent in t._grad_fn.saved_tensors():
                if isinstance(parent, AutogradTensor) and id(parent) not in all_tensors:
                    _collect(parent)

    _collect(tensor)

    for t in topo_order:
        if id(t) in grads:
            t.grad = grads[id(t)]
        if t._grad_fn is not None:
            for parent in t._grad_fn.saved_tensors():
                if isinstance(parent, AutogradTensor) and id(parent) in grads:
                    parent.grad = grads[id(parent)]
