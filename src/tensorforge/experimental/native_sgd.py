"""NativeSGD — the first native optimizer (Advanced C++ v3.8, the
eighth Phase C milestone; see docs/backend_experiments.md and
docs/native_autograd_design.md §19).

``NativeSGD(parameters, lr)`` is deliberately minimal: plain stochastic
gradient descent over ``NativeParameter`` objects — ``value ← value -
lr * grad`` — committed through the v3.7 controlled mutation contract.
No momentum, dampening, Nesterov, weight decay, parameter groups,
per-parameter learning rates, schedulers, or checkpointing, and no
training loop (the multi-iteration MLP training proof is v3.9). As of
Advanced C++ v3.13 it carries the in-memory optimizer state contract —
``state_dict()``/``load_state_dict()``, pure Python metadata since SGD
owns no tensor state (see the two method docstrings) — but no file
serialization. It is fully separate from the stable
``tensorforge.optim`` package: neither accepts the other's objects.

The contract (all tested in tests/test_native_sgd.py):

- **Parameters are stored by reference, keyed by identity.** The
  constructor materializes the iterable exactly once (lists,
  ``model.parameters()``, and generators all work), validates every
  entry as an open ``NativeParameter`` (errors name the offending
  position), deduplicates strictly by object identity in
  first-occurrence order (duplicate references and shared-module
  aliases become one stored entry — one update, one version increment
  per step; never value equality), and rejects an empty effective
  collection. The optimizer holds strong references but owns nothing:
  it never copies, replaces, closes, or mutates parameter storage
  outside the sanctioned update path — lifetime stays with the caller.
- **The learning rate is validated, never repaired.** ``lr`` must be a
  real number (``numbers.Real``; ``bool`` is explicitly rejected even
  though it subclasses ``int``, as are strings and arbitrary
  float-coercible objects), finite, and strictly positive — NaN,
  ±infinity, zero, and negatives raise. The accepted value is
  normalized to a Python float only after validation.
- **``step()`` is two-phase and mutation-atomic on its public failure
  surface.** Phase 1 preflights every stored parameter as still open,
  selects the active set (``requires_grad=False`` parameters are
  skipped *before* their gradients are examined — a frozen parameter
  with a stale gradient never updates; ``grad is None`` parameters are
  skipped), validates every active gradient (an open ``NativeTensor``
  of exactly the parameter's shape/dtype/device — no broadcasting,
  reshaping, casting, or device movement; errors name the parameter
  index), and stages every updated value **natively at the
  NativeTensorCore level** — the same autograd-unaware layer the
  engine's own backward math uses, so no graph node, parent, or
  backward callback can possibly be created, no NumPy touches the
  update, and the staged values are fresh owning tensors independent
  of every parameter. The broadcast ``lr`` scalar is the same value for
  every parameter in a step, so it is built **once per step** rather
  than once per parameter (Phase H, milestone H4) — lazily, so a step
  with no active parameter still allocates nothing, and released before
  the commit begins, so the optimizer still owns no native storage
  between steps and ``NativeSGD`` still keeps no tensor state. That is
  an allocation change only: the same operands, the same operations, in
  the same order, and bit-identical results.
  Any phase-1 failure releases all staged
  temporaries and changes no value, version, or gradient. Phase 2
  commits each staged value through ``NativeParameter.copy_value_()``
  in stored order — parameter identity, registration, aliases,
  ``requires_grad``, and the gradient (by identity and value) are all
  preserved, and each updated parameter's ``version`` increments
  exactly once (a numerically unchanged update — e.g. a zero gradient
  — still increments: the owned value was replaced). Staged
  temporaries are always released, on success and on failure alike.
  One narrow, honest limitation: after a fully successful preflight
  the commit loop's ``copy_value_`` calls cannot fail through any
  public surface, but an asynchronous interruption (e.g.
  KeyboardInterrupt) landing between two commits would leave the
  earlier parameters updated and the later ones not — each individual
  commit is still atomic and version-consistent, and no private
  rollback is manufactured to paper over that window.
- **Gradients are retained.** ``step()`` never clears, replaces,
  mutates, or closes a gradient — gradients persist until
  ``zero_grad()``, which preflights every stored parameter as open
  (failing before any clearing — no partial clears), then delegates to
  each parameter's own ``zero_grad()`` (grad → ``None``; values,
  versions, identity, ``requires_grad``, and registrations untouched;
  frozen parameters included harmlessly).
- **v3.7 staleness applies naturally.** A value-sensitive graph
  (multiply/matmul/relu edges over a parameter) built before
  ``step()`` becomes stale after it: the next ``backward()`` raises
  the existing deterministic stale-value error, gradients unchanged,
  and a fresh forward/backward uses the updated values. The optimizer
  neither weakens nor extends that classification.

The intended v3.9 pattern — forward → loss → backward → ``step()`` →
``zero_grad()`` → fresh forward — composes entirely from these pieces;
this milestone deliberately stops at the single verified step.
"""

import math
import numbers

from ..backends import cpp
from ._native_state_lock import state_transaction
from .native_optimizer_state import (
    FORMAT_VERSION,
    parameter_metadata,
    validate_parameter_metadata,
    validate_state_schema,
)
from .native_parameter import NativeParameter
from .native_tensor import NativeTensor

# The exact key set of a NativeSGD state dict (format version 1).
_STATE_KEYS = ("format_version", "optimizer", "lr", "parameters")


def _validated_lr(lr):
    """Validate a learning rate — real, non-bool, finite, strictly
    positive — and return it as a Python float. The constructor's
    contract, shared with ``load_state_dict`` so a loaded lr can never
    be weaker than a constructed one."""
    if isinstance(lr, bool) or not isinstance(lr, numbers.Real):
        raise TypeError(
            f"lr must be a real number, got {type(lr).__name__}"
        )
    lr = float(lr)
    if not math.isfinite(lr):
        raise ValueError(f"lr must be finite, got {lr}")
    if lr <= 0.0:
        raise ValueError(f"lr must be strictly positive, got {lr}")
    return lr


class NativeSGD:
    """Minimal native stochastic gradient descent:
    ``value ← value - lr * grad`` for every unique open trainable
    ``NativeParameter`` with a gradient, committed through
    ``copy_value_()``. See the module docstring for the full contract.
    """

    __slots__ = ("_parameters", "_lr")

    def __init__(self, parameters, lr):
        # Validate the learning rate first — cheap, and a bad lr must
        # never depend on whether the parameter iterable was consumable.
        lr = _validated_lr(lr)

        # Materialize the iterable exactly once (model.parameters()
        # lists and one-shot generators alike), then validate every
        # entry before storing anything.
        try:
            entries = list(parameters)
        except TypeError:
            raise TypeError(
                f"parameters must be an iterable of NativeParameter, "
                f"got {type(parameters).__name__}"
            )
        seen = set()
        unique = []
        for position, entry in enumerate(entries):
            if not isinstance(entry, NativeParameter):
                # Covers plain NativeTensor and the stable framework's
                # Tensor/Parameter alike — nothing is wrapped.
                raise TypeError(
                    f"parameters[{position}] must be a NativeParameter, "
                    f"got {type(entry).__name__}"
                )
            if entry.closed:
                raise RuntimeError(
                    f"parameters[{position}] has been closed"
                )
            if id(entry) in seen:
                continue  # identity dedup: aliases update once
            seen.add(id(entry))
            unique.append(entry)
        if not unique:
            raise ValueError(
                "NativeSGD requires at least one parameter"
            )
        self._parameters = tuple(unique)
        self._lr = lr

    @property
    def lr(self):
        """The validated learning rate, as a Python float. Read-only —
        this milestone has no schedulers and no mutation surface."""
        return self._lr

    def parameters(self):
        """The stored parameters: unique by identity, first-occurrence
        order — the exact objects the caller supplied, never copies.
        Returns a new list (mutating it cannot affect the optimizer)."""
        return list(self._parameters)

    def step(self):
        """One SGD update: for every stored parameter that is trainable
        (``requires_grad=True``) and has a gradient, replace its value
        with ``value - lr * grad`` via ``copy_value_()`` (one version
        increment each). Frozen and gradient-less parameters are
        skipped untouched. Two-phase — every validation and every
        staged native computation completes before the first parameter
        mutates, so the public failure surface (closed parameter,
        closed/mismatched gradient, staging failure) changes no value,
        version, or gradient. Returns None."""
        # Phase 1a: every stored parameter must still be open — checked
        # for the whole collection first, so a closed parameter fails
        # deterministically regardless of earlier entries' grad state.
        for index, parameter in enumerate(self._parameters):
            if parameter.closed:
                raise RuntimeError(
                    f"NativeSGD.step(): parameters[{index}] has been "
                    f"closed"
                )

        # Phase 1b: select the active set and validate every gradient
        # completely before anything is staged or mutated. Frozen
        # parameters are skipped before their grad is examined.
        active = []
        for index, parameter in enumerate(self._parameters):
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            if grad is None:
                continue
            if not isinstance(grad, NativeTensor):
                raise TypeError(
                    f"NativeSGD.step(): parameters[{index}].grad must "
                    f"be a NativeTensor, got {type(grad).__name__}"
                )
            if grad.closed:
                raise RuntimeError(
                    f"NativeSGD.step(): parameters[{index}].grad has "
                    f"been closed"
                )
            if grad.shape != parameter.shape:
                raise ValueError(
                    f"NativeSGD.step(): parameters[{index}].grad shape "
                    f"{grad.shape} does not match the parameter shape "
                    f"{parameter.shape}"
                )
            if grad.dtype != parameter.dtype or grad.device != parameter.device:
                raise ValueError(
                    f"NativeSGD.step(): parameters[{index}].grad "
                    f"dtype/device {grad.dtype}/{grad.device} does not "
                    f"match the parameter "
                    f"{parameter.dtype}/{parameter.device}"
                )
            active.append((parameter, grad))

        # Phase 1c: stage every updated value natively at the core
        # level — the autograd-unaware layer, so no graph can exist —
        # reading each parameter's *current* value. The transient
        # scale/scaled cores are closed immediately; the staged results
        # are fresh owning tensors independent of every parameter and
        # gradient. A failure here releases everything staged so far
        # and has mutated nothing.
        #
        # H4: the ``lr`` scalar is identical for every parameter in a
        # step, so it is built at most once per step (lazily, per
        # dtype/device, so a step with no active parameter still allocates
        # nothing) instead of once per parameter. It is a read-only
        # broadcast operand, it is released before the commit begins, and
        # it is never stored on the optimizer — SGD keeps no tensor state,
        # and H4 did not give it any. The learning rate is read once here,
        # which is the value the whole step observes; it cannot change
        # mid-step, and a value changed between steps is picked up by the
        # next step.
        learning_rate = self._lr
        scales = {}   # (dtype, device) -> the shared lr scalar core
        staged = []
        try:
            try:
                for parameter, grad in active:
                    parameter_core = parameter._require_open()
                    grad_core = grad._require_open()
                    key = (grad_core.dtype, grad_core.device)
                    scale = scales.get(key)
                    if scale is None:
                        scale = cpp.NativeTensorCore.full(
                            (), learning_rate,
                            dtype=key[0], device=key[1],
                        )
                        scales[key] = scale
                    scaled = grad_core.multiply(scale)  # lr * grad
                    try:
                        updated = parameter_core.subtract(scaled)
                    finally:
                        scaled.close()
                    staged.append(
                        (parameter, NativeTensor._from_core(updated))
                    )
            except BaseException:
                for _, update in staged:
                    update.close()
                raise
        finally:
            for scale in scales.values():
                scale.close()

        # Phase 2: commit in stored order through the one sanctioned
        # mutation path — identity, gradients, requires_grad, and
        # registrations preserved; version +1 per commit. After the
        # completed preflight these calls cannot fail through any
        # public surface (see the module docstring for the honest
        # asynchronous-interruption caveat). Staged temporaries are
        # released on every exit path.
        try:
            for parameter, update in staged:
                parameter.copy_value_(update)
        finally:
            for _, update in staged:
                update.close()

    def zero_grad(self):
        """Clear every stored parameter's gradient to ``None`` via the
        parameter's own ``zero_grad()`` (frozen parameters included
        harmlessly; values, versions, identities, ``requires_grad``,
        and registrations untouched; grad objects dropped, never
        closed). Preflights that every stored parameter is still open
        and fails before clearing anything otherwise — never a partial
        clear. Returns None."""
        for index, parameter in enumerate(self._parameters):
            if parameter.closed:
                raise RuntimeError(
                    f"NativeSGD.zero_grad(): parameters[{index}] has "
                    f"been closed"
                )
        for parameter in self._parameters:
            parameter.zero_grad()

    # -- optimizer state (v3.13: in-memory only; files are v3.14) ----------

    def state_dict(self):
        """The optimizer's restorable in-memory state, as a plain
        independent dict (format version 1)::

            {"format_version": 1, "optimizer": "NativeSGD",
             "lr": <float>,
             "parameters": ({"shape", "dtype", "device"}, ...)}

        NativeSGD keeps no tensor-valued optimizer state, so the dict
        is pure Python metadata: the validated learning rate plus one
        shape/dtype/device entry per unique stored parameter, in the
        optimizer's deterministic first-occurrence order (shared
        aliases appear once — the optimizer already deduplicated).
        Restoration across instances is purely positional; no object
        identity, name, parameter value, or gradient is included, and
        no parameter object is exposed. Every stored parameter is
        preflighted as open (a closed parameter raises naming its
        index); nothing is created, mutated, or graphed. Each call
        returns fresh containers the caller may mutate freely."""
        for index, parameter in enumerate(self._parameters):
            if parameter.closed:
                raise RuntimeError(
                    f"NativeSGD.state_dict(): parameters[{index}] has "
                    f"been closed"
                )
        return {
            "format_version": FORMAT_VERSION,
            "optimizer": "NativeSGD",
            "lr": self._lr,
            "parameters": parameter_metadata(self._parameters),
        }

    def load_state_dict(self, state):
        """Restore optimizer behavior from a ``state_dict()`` dict —
        for NativeSGD, the learning rate.

        The complete state is validated before anything changes: every
        stored parameter open, ``state`` a plain dict with exactly the
        schema keys, ``format_version == 1``, the ``"NativeSGD"`` tag,
        the lr under the constructor's full contract, and the ordered
        parameter metadata matching this optimizer's stored parameters
        position by position (exact count and exact shape/dtype/device
        — no casting, reshaping, broadcasting, or device movement;
        sequence fields accept tuple or list). Only then is ``lr``
        replaced — a single assignment, so ordinary failure leaves the
        learning rate and every parameter (identity, value, version,
        gradient, alias, registration) exactly unchanged, and success
        touches parameters not at all (loading the same lr is still a
        successful load and changes no version). The supplied state is
        caller-owned and read-only: never mutated, retained, or
        consumed. Returns None.

        Runs under the shared native state-transaction guard (Phase G,
        milestone G5), so a checkpoint load that replaces the model, this
        optimizer, and the generators cannot be interleaved by another
        participating state load. The guard is reentrant, so the
        checkpoint transaction holds it and this method re-enters it; this
        path takes no generator lock, so it cannot invert the universal
        order."""
        where = "NativeSGD.load_state_dict()"
        with state_transaction():
            for index, parameter in enumerate(self._parameters):
                if parameter.closed:
                    raise RuntimeError(
                        f"{where}: parameters[{index}] has been closed"
                    )
            validate_state_schema(state, "NativeSGD", _STATE_KEYS, where)
            lr = _validated_lr(state["lr"])
            validate_parameter_metadata(
                state["parameters"], self._parameters, where
            )
            # Commit: one attribute assignment, after all validation.
            self._lr = lr

    def __repr__(self):
        return (
            f"NativeSGD(lr={self._lr}, "
            f"parameters={len(self._parameters)})"
        )
