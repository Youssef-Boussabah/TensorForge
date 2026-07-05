"""SGD: plain stochastic gradient descent, the simplest optimizer."""


class SGD:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Take one descent step: move each parameter against its gradient."""
        for param in self.parameters:
            if param.grad is None:
                continue
            param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        """Clear gradients so the next backward() starts fresh."""
        for param in self.parameters:
            param.grad = None
