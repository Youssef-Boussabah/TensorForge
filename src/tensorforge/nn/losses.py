"""Loss functions.

Losses are ordinary Tensor expressions, so they need no backward logic
of their own — autograd differentiates through them automatically.
"""

import numpy as np

from tensorforge.tensor import Tensor


def mse_loss(prediction, target):
    """Mean squared error: mean((prediction - target) ** 2).

    ``target`` may be a Tensor or a plain Python/NumPy value; the
    subtraction wraps it automatically.
    """
    return ((prediction - target) ** 2).mean()


def cross_entropy(logits, targets):
    """Cross-entropy loss for multi-class classification.

    ``logits`` is a Tensor of shape (batch, num_classes) with raw,
    unnormalized scores. ``targets`` holds the correct class index for
    each row (Tensor, NumPy array, or plain list of ints).

    Returns the mean negative log likelihood of the correct classes.
    """
    if isinstance(targets, Tensor):
        targets = targets.data
    targets = np.asarray(targets, dtype=int)

    probs = logits.softmax(axis=-1)

    # Select each row's correct-class probability. A one-hot mask keeps
    # the selection inside Tensor ops (multiply + sum), so gradients
    # flow back to the logits.
    num_rows = logits.data.shape[0]
    one_hot = np.zeros_like(logits.data)
    one_hot[np.arange(num_rows), targets] = 1.0
    correct_class_probs = (probs * one_hot).sum(axis=-1)

    return -(correct_class_probs.log().mean())
