"""NativeCrossEntropyLoss — the native classification loss module
(Phase E, milestone E7; see docs/native_classification_design.md §4.5
and §8).

A **parameter-free**, **buffer-free** ``NativeModule`` that is a thin
wrapper around one public operation::

    loss = logits.cross_entropy(targets, reduction=self.reduction)

That is the whole forward. There is deliberately **no second
cross-entropy implementation here**: no Core call, no C ABI call, no
NumPy, no ``softmax``/``log_softmax`` composition, and no Python
formula. Every behavior of the E6 operation is therefore inherited
rather than reimplemented, and cannot drift from it:

* rank-2 ``(batch_size, num_classes)`` logits with the class axis fixed
  at axis 1 (no ``axis`` argument exists at either layer);
* the strict E5 target contract — a one-dimensional sequence of integer
  class labels copied into independently owned, contiguous, read-only
  ``int64`` metadata before anything is allocated, with ``bool`` and
  floating-point labels (``1.0`` included) rejected outright, so caller
  mutation after the forward cannot reach the gradient;
* the fused, numerically stable forward (one kernel: row maximum,
  log-sum-exp, saved probabilities, and loss in a single pass) and the
  **scalar** ``()`` output that ``backward()`` seeds by default;
* graph-owned private saved probabilities, retained under
  ``retain_graph=True`` and across a failed retryable backward, released
  exactly once with the graph history, and closed immediately when
  nothing requires gradients;
* a backward that **never rereads the logits** and records **no
  expected parameter version**;
* Policy-B copy-then-compute for strided logits, and full failure
  atomicity across validation, allocation, graph construction, and
  backward.

**Reduction contract**: exactly ``"mean"`` and ``"sum"``, validated in
the **constructor** through the same private helper the operation uses,
so the module's accepted set, its error types, and its messages cannot
diverge from ``NativeTensor.cross_entropy``'s — a non-string raises
TypeError, an unrecognized string (``"none"``, ``"Mean"``, ``""``, ...)
raises ValueError, and an invalid reduction can never reach the
operation because the module cannot be constructed with one. The
validated string is **constructor configuration, not model state**: it
never appears in ``state_dict()`` and never becomes a buffer.

**Input contract**: ``forward(logits, targets)`` requires an open
``NativeTensor`` (a ``NativeParameter`` is accepted as the subclass it
is; the stable framework's ``Tensor``, NumPy arrays, lists, scalars, and
closed tensors are rejected with clear errors) — the ``NativeMSELoss``
convention. ``targets`` is **not** validated here: it is passed straight
through, so exactly one target-validation implementation exists in the
repository. Inputs are never mutated, reshaped, cast, or copied, and the
module stores no tensors between calls.

Parameter-free module behavior: no parameters, children, or buffers;
``state_dict()`` is empty (loading ``{}`` succeeds; unexpected keys
follow the existing strict/non-strict rules); the inherited ``training``
flag propagates normally but never affects numerics. E7 adds **no**
kernel, C ABI export, ctypes symbol, tensor operation, or checkpoint
change — the native checkpoint format stays version 1.

Fully separate from the stable framework's
``tensorforge.nn.cross_entropy``. It takes **no** dtype argument
and must not gain one: it is a thin delegate that inherits the
dtype of the logits it is handed, and the fused kernel underneath
it has been dtype-general since Phase I milestone I6. CPU only.
"""

from ..backends import cpp
from .native_module import NativeModule
from .native_tensor import NativeTensor


class NativeCrossEntropyLoss(NativeModule):
    """Multi-class cross-entropy over raw native logits:
    ``logits.cross_entropy(targets, reduction)``. Parameter-free,
    buffer-free, scalar output; forward and backward are supplied
    entirely by the E5/E6 stack. See the module docstring for the full
    contract."""

    def __init__(self, reduction="mean"):
        # The operation's own reduction validator — not a second copy of
        # the rule — so "mean"/"sum", the TypeError/ValueError split, and
        # the message text stay identical at both layers. Pure Python:
        # constructing this module never needs the compiled backend.
        reduction, _ = cpp._normalize_reduction(
            reduction, "NativeCrossEntropyLoss"
        )
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """The scalar cross-entropy of ``logits`` against ``targets``.

        ``logits`` must be an open ``NativeTensor`` of shape
        ``(batch_size, num_classes)``; ``targets`` is a one-dimensional
        sequence of integer class labels, validated and copied by the
        operation itself. Delegates in full — the module contributes no
        arithmetic, no validation of the labels, and no graph state."""
        if not isinstance(logits, NativeTensor):
            raise TypeError(
                f"NativeCrossEntropyLoss.forward requires a NativeTensor "
                f"logits argument, got {type(logits).__name__}"
            )
        if logits.closed:
            raise RuntimeError(
                "NativeCrossEntropyLoss.forward: the logits tensor has been "
                "closed"
            )
        return logits.cross_entropy(targets, reduction=self.reduction)

    def __repr__(self):
        return f"NativeCrossEntropyLoss(reduction={self.reduction!r})"
