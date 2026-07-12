"""NativeMSELoss — the first native loss module (Advanced C++ v3.6;
see docs/backend_experiments.md).

Mean-squared-error as a **parameter-free** ``NativeModule`` composed
entirely from existing differentiable ``NativeTensor`` operations::

    difference = prediction.subtract(target)
    squared    = difference.multiply(difference)
    loss       = squared.mean()   # reduction="mean" (default)
    loss       = squared.sum()    # reduction="sum"

Both reductions produce a **scalar** (shape ``()``), so
``loss.backward()`` works with the existing default seed; an explicit
scalar upstream gradient scales both input gradients exactly as the
engine already defines. There is **no manual or fused loss backward**:
the analytical gradients — ``dL/dprediction = 2·(prediction − target)/N``
and ``dL/dtarget = −2·(prediction − target)/N`` under ``mean`` (drop the
``/N`` under ``sum``) — fall out of the existing graph: ``multiply``'s
duplicate-parent accumulation on the shared ``difference`` node yields
the factor 2, ``subtract``'s backward supplies the sign split, and
``mean``'s existing native backward supplies the ``1/N`` scaling (no
division operation exists or is needed). Graph lifetime is untouched:
one-shot by default, ``retain_graph=True`` reuse, deterministic
freed-history errors.

**Reduction contract** (deliberately small): exactly ``"mean"`` and
``"sum"``, validated in the constructor by exact string match — no case
or whitespace normalization, no coercion, no ``"none"`` (both supported
reductions are scalar, which is sufficient for the first native
training loop; unreduced losses are a later, broader API). The
validated string is constructor configuration, **not** model state — it
never appears in ``state_dict()``.

**Input contract**: ``forward(prediction, target)`` requires two open
``NativeTensor``s (a ``NativeParameter`` is accepted as the subclass it
is; the stable framework's ``Tensor``, NumPy arrays, lists, scalars,
and closed tensors are rejected with errors naming which argument is
invalid) of **exactly equal shape** — broadcasting is deliberately
forbidden even though ``subtract`` supports it (a silently broadcast
loss hides target-shape bugs), and the check runs *before* any graph
node is built. dtype/device must match exactly (float64/cpu today,
validated through the existing metadata contract). Every rank the
participating operations support works; zero-element tensors cannot be
constructed by the native runtime (``NativeStorage`` requires a
positive size), and NativeMSELoss simply inherits that limitation.
Inputs are never mutated, reshaped, cast, or copied, and the module
stores no temporary tensors.

Parameter-free module behavior: no parameters, children, buffers, or
storage; ``state_dict()`` is empty (loading ``{}`` succeeds; unexpected
keys follow the v3.3 strict/non-strict rules); the inherited
``training`` flag propagates normally but never affects numerics.

The v3.3–v3.5 mutation boundary is unchanged: the supported sequence is
model forward → loss → backward → parameter/state updates only after
the graph completes (no version counters yet — that is v3.7's scope).
Fully separate from the stable framework's ``mse_loss``; float64/cpu
only; no optimizer or training loop yet.
"""

from .native_module import NativeModule
from .native_tensor import NativeTensor

_REDUCTIONS = ("mean", "sum")


class NativeMSELoss(NativeModule):
    """Mean squared error over existing native operations:
    ``mean((prediction - target)^2)`` (default) or the ``sum``
    variant. Parameter-free; scalar output; backward supplied entirely
    by the existing autograd. See the module docstring for the full
    contract."""

    def __init__(self, reduction="mean"):
        if not isinstance(reduction, str):
            raise TypeError(
                f"reduction must be a str, got {type(reduction).__name__}"
            )
        if reduction not in _REDUCTIONS:
            # Exact match only: "Mean", " mean ", "SUM", ... all land
            # here — nothing is normalized or coerced.
            raise ValueError(
                f"reduction must be one of {list(_REDUCTIONS)}, got "
                f"{reduction!r}"
            )
        super().__init__()
        self.reduction = reduction

    def _validate_operand(self, value, role):
        """``value`` must be an open NativeTensor; errors name whether
        the prediction or the target is at fault."""
        if not isinstance(value, NativeTensor):
            raise TypeError(
                f"NativeMSELoss.forward requires a NativeTensor "
                f"{role}, got {type(value).__name__}"
            )
        if value.closed:
            raise RuntimeError(
                f"NativeMSELoss.forward: the {role} tensor has been closed"
            )

    def forward(self, prediction, target):
        """The scalar MSE of ``prediction`` against ``target`` —
        exact-shape, exact-dtype/device, no broadcasting. Composed as
        subtract → multiply(diff, diff) → mean/sum, so the existing
        engine provides every gradient."""
        self._validate_operand(prediction, "prediction")
        self._validate_operand(target, "target")
        if prediction.shape != target.shape:
            raise ValueError(
                f"NativeMSELoss requires exactly equal shapes (no "
                f"broadcasting): prediction has shape {prediction.shape}, "
                f"target has shape {target.shape}"
            )
        if (
            prediction.dtype != target.dtype
            or prediction.device != target.device
        ):
            raise ValueError(
                f"NativeMSELoss requires matching dtype/device: prediction "
                f"is {prediction.dtype}/{prediction.device}, target is "
                f"{target.dtype}/{target.device}"
            )
        difference = prediction.subtract(target)
        squared = difference.multiply(difference)
        if self.reduction == "mean":
            return squared.mean()
        return squared.sum()

    def __repr__(self):
        return f"NativeMSELoss(reduction={self.reduction!r})"
