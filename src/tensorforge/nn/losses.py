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


def _align_binary_targets(logits_np, targets):
    """Validate 0/1 targets and align their shape to the logits.

    Allows targets shaped (batch,) against logits shaped (batch, 1);
    otherwise the shapes must match exactly.
    """
    if logits_np.ndim > 2 or (logits_np.ndim == 2 and logits_np.shape[1] != 1):
        raise ValueError(
            f"binary logits must be scalar, (batch,), or (batch, 1), "
            f"got shape {logits_np.shape}"
        )
    targets = np.asarray(targets, dtype=np.float64)
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("binary targets must contain only 0 and 1")
    if targets.shape == logits_np.shape:
        return targets
    if logits_np.ndim == 2 and targets.shape == (logits_np.shape[0],):
        return targets.reshape(-1, 1)
    raise ValueError(
        f"targets shape {targets.shape} is incompatible with "
        f"logits shape {logits_np.shape}"
    )


def binary_cross_entropy(logits, targets):
    """Binary cross-entropy on raw logits (no Sigmoid layer needed).

    ``logits`` is a Tensor of raw scores — scalar, (batch,), or
    (batch, 1). ``targets`` holds 0/1 labels (Tensor, array, or list).
    Returns a scalar Tensor: the mean binary cross-entropy.

    Working on logits instead of sigmoid probabilities is what keeps
    this stable: the naive -y*log(p) - (1-y)*log(1-p) hits log(0) once
    a probability saturates, while the logits form
        max(x, 0) - x*y + log(1 + exp(-|x|))
    is algebraically identical and stays finite for any x.
    """
    if isinstance(targets, Tensor):
        targets = targets.data
    x = logits.data
    y = _align_binary_targets(x, targets)

    values = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
    out = Tensor(
        values.mean(),
        requires_grad=logits.requires_grad,
        _children=(logits,),
        _op="binary_cross_entropy",
    )

    def _backward():
        # The closed form: d(loss)/d(logits) = (sigmoid(x) - y) / N.
        # Computed via exp(-|x|) so the exponential never overflows.
        e = np.exp(-np.abs(x))
        sig = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        logits._accumulate_grad((sig - y) / x.size * out.grad)

    out._backward = _backward
    return out


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
