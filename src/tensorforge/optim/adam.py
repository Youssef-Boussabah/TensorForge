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
            # Skip parameters with no gradient and frozen parameters
            # (requires_grad=False), even if a stale grad is present.
            # Skipped parameters keep their m/v state untouched.
            if param.grad is None or not param.requires_grad:
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

    def state_dict(self):
        """Return everything needed to resume Adam exactly: the
        hyperparameters, the step count, and copies of the moment
        estimates."""
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "t": self.t,
            "m": [m.copy() for m in self.m],
            "v": [v.copy() for v in self.v],
        }

    def load_state_dict(self, state):
        """Restore state produced by ``state_dict``.

        The moment estimates must match the optimizer's parameter list
        in length and shapes. Arrays are copied in, and the parameter
        list itself is untouched.
        """
        m, v = state["m"], state["v"]
        if len(m) != len(self.parameters) or len(v) != len(self.parameters):
            raise ValueError(
                f"optimizer state holds {len(m)}/{len(v)} moment arrays "
                f"but the optimizer has {len(self.parameters)} parameters"
            )
        for i, param in enumerate(self.parameters):
            for label, moments in (("m", m), ("v", v)):
                shape = np.asarray(moments[i]).shape
                if shape != param.data.shape:
                    raise ValueError(
                        f"shape mismatch for {label}[{i}]: parameter is "
                        f"{param.data.shape}, state has {shape}"
                    )

        self.lr = float(state["lr"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.t = int(state["t"])
        self.m = [np.array(x, dtype=np.float64) for x in m]
        self.v = [np.array(x, dtype=np.float64) for x in v]
