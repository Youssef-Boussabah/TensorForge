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

    Returns a scalar Tensor: the mean negative log likelihood of the
    correct classes. Unlike naive softmax-then-log, this is a single
    fused op with a custom backward, so it stays numerically stable
    even for huge logits (log(softmax) never underflows to log(0)).
    """
    if isinstance(targets, Tensor):
        targets = targets.data
    targets = np.asarray(targets, dtype=int)

    batch_size = logits.data.shape[0]
    rows = np.arange(batch_size)

    # Stable log-softmax: log p = (x - max) - log(sum(exp(x - max))).
    # Shifting by the row max keeps exp() from overflowing without
    # changing the result (softmax is shift-invariant).
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    out = Tensor(
        -log_probs[rows, targets].mean(),
        requires_grad=logits.requires_grad,
        _children=(logits,),
        _op="cross_entropy",
    )

    def _backward():
        # The classic closed form: d(loss)/d(logits) is
        #   (softmax(logits) - one_hot(targets)) / batch_size
        # — push each row's probabilities toward the one-hot target.
        grad = np.exp(log_probs)  # softmax, recovered from log-softmax
        grad[rows, targets] -= 1.0
        logits._accumulate_grad(grad / batch_size * out.grad)

    out._backward = _backward
    return out
