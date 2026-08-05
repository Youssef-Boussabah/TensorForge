"""Native reporting metrics (Phase E, milestone E7; see
docs/native_classification_design.md §8).

One function today: ``native_accuracy(logits, targets) -> float``.

**This is a reporting helper, and the name is the whole honest claim.**
It is *not* native C++ compute, *not* a ``NativeTensorCore`` operation,
*not* a differentiable ``NativeTensor`` operation, and *not* a module.
There is no accuracy kernel, no C ABI export, and no ctypes symbol — E7
added none.

**A native ``argmax`` exists.** Phase K, milestone K3 shipped it:
``NativeTensorCore.argmax`` and ``NativeTensor.argmax``, over the
``tf_core_argmax`` export. It takes a floating tensor at either dtype and
returns a fresh owning ``int64`` index tensor.

**This metric deliberately does not use it, and that is a choice rather
than a gap.** Two earlier versions of this paragraph have expired, and the
history is recorded so the correction is legible rather than silent:
through K1 it explained the operation's non-existence by the runtime having
no integer dtype, and K2 supplied exactly that dtype; from K2 to K3 it
called the same non-existence deliberate, and K3 shipped the operation.
Both explanations have retired with the facts under them. What survives is
the conclusion each was attached to: this helper still reports through the
explicit host boundary below.

The reasons it stays that way:

* Substituting the native ``argmax`` would not make this a native runtime
  operation, a differentiable operation, or a module. It would still be a
  Python function returning a Python ``float``, and every claim this
  docstring declines to make would still have to be declined.
* Accuracy is ``mean(predictions == targets)``, so the comparison needs an
  integer **equality** reduction against the host targets. The native
  runtime has none and Phase K ships none, so the values would have to come
  back to the host regardless — one operation later, one native allocation
  heavier, with the single explicit conversion harder to see.
* **The two behaviors are not universally equivalent, and must never be
  documented as though they were.** This metric inherits ``numpy.argmax``'s
  tie and exceptional-value conventions unchanged, while TensorForge's
  native ``argmax`` has its own committed rule — in particular a
  **first-NaN** rule, under which a NaN beats every finite value and either
  infinity and the lowest-indexed NaN wins. Swapping the implementation
  would therefore be a silent behavior change for NaN logits, not a
  refactor. The native rule is normative for ``NativeTensor.argmax`` and is
  specified in ``docs/native_integer_tensors_design.md`` §17.5; the rule
  this metric follows is NumPy's, stated below.

Because it is outside training, autograd, and native numerical
execution, it is **allowed and required** to leave native memory through
the explicit public conversion boundary::

    values = logits.to_numpy()          # one deliberate copy out
    predictions = np.argmax(values, axis=1)

That is a real NumPy round trip, stated plainly rather than hidden. It
is exactly the boundary the training path forbids — every native
operation keeps tensor data in native storage, and the cross-entropy
tripwire tests prove it — and the distinction is the point: a metric
reports on a finished forward pass, so a host copy costs nothing that
matters and buys a one-line implementation.

The metric shares the classification stack's **strict target contract**
(§6) exactly, by calling the same private preparation helper the E5 Core
forward calls: a one-dimensional sequence of integer class labels of
length ``batch_size``, every label in ``[0, num_classes)``, with
``bool`` (Python and NumPy), floating-point values including integral
ones like ``1.0``, complex values, strings, bytes, nested and ragged
sequences, rank-2 arrays, scalars, and out-of-int64 values all rejected
before anything is materialized. There is no second, more permissive
metric path.

It creates **no** autograd graph, parents, or backward callback; it
touches no ``.grad``, no graph history, no ``requires_grad`` flag, and
no ``NativeParameter`` version; it allocates no native output; and it
retains nothing after returning — not the target copy, not the host
array. A graph built before the call is still fully usable after it.

Accuracy is the plain fraction ``mean(argmax(logits, axis=1) ==
targets)`` in ``[0.0, 1.0]``, returned as a built-in ``float`` — no
percentage, no rounding. Ties and exceptional values follow
``numpy.argmax`` unchanged: a tie goes to the **first** maximal index,
and no special NaN or infinity semantics are invented here — which is
**not** the same contract as ``NativeTensor.argmax``'s first-NaN rule, as
the note above records. This is a
reporting helper, not a numerically stable loss — for training, use
``NativeCrossEntropyLoss`` or ``NativeTensor.cross_entropy``, which are
fused and stable and never form probabilities at all.
"""

import numpy as np

from ..backends import cpp
from .native_tensor import NativeTensor


def native_accuracy(logits, targets):
    """The fraction of rows whose highest logit is the target class.

    ``logits`` is an open ``NativeTensor`` of shape
    ``(batch_size, num_classes)`` — rank exactly 2, with dimension 1 the
    class axis. ``targets`` is a one-dimensional sequence of integer
    class labels under the strict Phase-E contract (see the module
    docstring); it is validated and copied by the same private helper
    the cross-entropy forward uses, so the accepted and rejected forms
    are identical at both call sites.

    Returns a built-in ``float`` in ``[0.0, 1.0]``. Reporting-only: no
    graph is built, no gradient or parameter is touched, and nothing is
    retained. The logits are materialized once through the explicit
    public ``to_numpy()`` boundary and the winners are found with
    ``numpy.argmax``, whose first-maximal-index tie rule this function
    adopts unchanged."""
    if not isinstance(logits, NativeTensor):
        raise TypeError(
            f"native_accuracy requires a NativeTensor logits argument, got "
            f"{type(logits).__name__}"
        )
    if logits.closed:
        raise RuntimeError("native_accuracy: the logits tensor has been closed")
    shape = logits.shape
    if len(shape) != 2:
        raise ValueError(
            f"native_accuracy requires 2-D (batch_size, num_classes) logits, "
            f"got shape {shape}"
        )
    batch_size, num_classes = shape
    # The same strict target validator the E5 Core forward calls — one
    # implementation of the rule, one owned read-only int64 copy, and it
    # runs *before* anything is materialized, so a rejected call converts
    # nothing.
    labels = cpp._prepare_class_targets(
        targets, batch_size, num_classes, "native_accuracy"
    )
    # The deliberate conversion boundary: one explicit copy out of native
    # storage. Every native *training* operation refuses to do this; a
    # reporting metric is exactly the case where it is honest.
    values = logits.to_numpy()
    predictions = np.argmax(values, axis=1)
    return float(np.mean(predictions == labels))
