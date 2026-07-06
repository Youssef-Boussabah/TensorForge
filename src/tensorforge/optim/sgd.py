"""SGD: plain stochastic gradient descent, the simplest optimizer."""


class SGD:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Take one descent step: move each parameter against its gradient."""
        for param in self.parameters:
            # Skip parameters with no gradient and frozen parameters
            # (requires_grad=False), even if a stale grad is present.
            if param.grad is None or not param.requires_grad:
                continue
            param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        """Clear gradients so the next backward() starts fresh."""
        for param in self.parameters:
            param.grad = None

    def state_dict(self):
        """Return the optimizer's restorable state (SGD is stateless
        apart from its hyperparameters)."""
        return {"lr": self.lr}

    def load_state_dict(self, state):
        """Restore hyperparameters from ``state``. Parameters are
        untouched — they belong to the model, not the checkpoint."""
        self.lr = float(state["lr"])
