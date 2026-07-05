"""Loss functions.

Losses are ordinary Tensor expressions, so they need no backward logic
of their own — autograd differentiates through them automatically.
"""


def mse_loss(prediction, target):
    """Mean squared error: mean((prediction - target) ** 2).

    ``target`` may be a Tensor or a plain Python/NumPy value; the
    subtraction wraps it automatically.
    """
    return ((prediction - target) ** 2).mean()
