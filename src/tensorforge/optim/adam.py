"""Adam: SGD with a per-parameter adaptive step size.

Adam keeps two exponential moving averages per parameter — the gradient
(first moment, "momentum") and the squared gradient (second moment) —
and divides the step by the square root of the second, so parameters
with consistently large gradients take smaller steps and vice versa.
"""

import numpy as np


class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # number of steps taken so far
        # Moment estimates, one pair per parameter, persisted across steps.
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Take one Adam step using each parameter's current gradient."""
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            grad = param.grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

            # Bias correction: m and v start at zero, so for small t the
            # moving averages underestimate the true moments. Dividing by
            # (1 - beta^t) rescales them; the correction fades as t grows.
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Clear gradients so the next backward() starts fresh."""
        for param in self.parameters:
            param.grad = None
