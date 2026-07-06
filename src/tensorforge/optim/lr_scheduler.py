"""Learning-rate schedulers: change the optimizer's lr over time."""

import numbers


class StepLR:
    """Multiply the optimizer's lr by ``gamma`` every ``step_size`` epochs.

    Call ``scheduler.step()`` once per epoch, after ``optimizer.step()``.
    The scheduler only ever touches ``optimizer.lr`` — it never steps
    the optimizer or looks at gradients or parameters.
    """

    def __init__(self, optimizer, step_size, gamma=0.1):
        if not isinstance(step_size, int) or isinstance(step_size, bool) or step_size <= 0:
            raise ValueError(f"step_size must be a positive int, got {step_size!r}")
        if not isinstance(gamma, numbers.Real) or gamma <= 0:
            raise ValueError(f"gamma must be a positive number, got {gamma!r}")
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = float(gamma)
        self.last_epoch = 0

    def step(self):
        """Advance one epoch; decay the lr on step boundaries.

        Returns the (possibly updated) learning rate.
        """
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr = self.optimizer.lr * self.gamma
        return float(self.optimizer.lr)

    def state_dict(self):
        """Return the scheduler's restorable state (not the optimizer)."""
        return {
            "step_size": self.step_size,
            "gamma": self.gamma,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state):
        """Restore state produced by ``state_dict``. The optimizer this
        scheduler drives is left untouched."""
        self.step_size = int(state["step_size"])
        self.gamma = float(state["gamma"])
        self.last_epoch = int(state["last_epoch"])
