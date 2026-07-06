"""Gradient clipping: limit gradient size before the optimizer step.

Both helpers modify gradients in place and never touch parameter data.
Parameters with no gradient, or frozen ones (requires_grad=False), are
skipped.
"""

import numbers

import numpy as np


def _clippable_grads(parameters):
    return [p.grad for p in parameters if p.grad is not None and p.requires_grad]


def clip_grad_norm(parameters, max_norm, eps=1e-6):
    """Scale gradients so their combined L2 norm is at most ``max_norm``.

    The norm is computed over *all* included gradients together, and if
    it exceeds ``max_norm`` every gradient is scaled by the same factor
    — the update direction is preserved, only its length shrinks.

    Returns the total norm measured *before* clipping, as a float.
    """
    if not isinstance(max_norm, numbers.Real) or max_norm <= 0:
        raise ValueError(f"max_norm must be a positive number, got {max_norm!r}")
    if not isinstance(eps, numbers.Real) or eps <= 0:
        raise ValueError(f"eps must be a positive number, got {eps!r}")

    grads = _clippable_grads(parameters)
    if not grads:
        return 0.0

    total_norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for grad in grads:
            grad *= scale
    return total_norm


def clip_grad_value(parameters, clip_value):
    """Clamp every gradient value into [-clip_value, clip_value] in place.

    Unlike clip_grad_norm this treats each value independently, so it
    can change the update direction, not just its length.
    """
    if not isinstance(clip_value, numbers.Real) or clip_value < 0:
        raise ValueError(f"clip_value must be a non-negative number, got {clip_value!r}")

    for grad in _clippable_grads(parameters):
        np.clip(grad, -clip_value, clip_value, out=grad)
