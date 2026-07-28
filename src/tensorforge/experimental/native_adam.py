"""NativeAdam — the native adaptive optimizer (Advanced C++ v3.12, the
twelfth Phase C milestone; see docs/backend_experiments.md and
docs/native_autograd_design.md §19).

``NativeAdam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8)`` is
minimal, correct Adam over ``NativeParameter`` objects: per-parameter
first- and second-moment estimates with bias correction, every update
computed graph-free at the ``NativeTensorCore`` level and committed
through the v3.7 controlled mutation contract (``copy_value_``). No
weight decay, AMSGrad, parameter groups, per-parameter learning rates,
schedulers, or checkpointing (in-memory
``state_dict``/``load_state_dict`` landed in v3.13 — see below; file
checkpoint archives are v3.14), and no general
tensor division — ``reciprocal`` + ``multiply`` compose the bias
corrections and the denominator. Fully separate from the stable
``tensorforge.optim.Adam``: neither accepts the other's objects.

The contract (all tested in tests/test_native_adam.py):

- **Parameters follow the NativeSGD contract exactly.** The
  constructor materializes the iterable exactly once, validates every
  entry as an open ``NativeParameter`` (position-named errors),
  deduplicates strictly by object identity in first-occurrence order
  (duplicate references and shared aliases: one entry, one state slot,
  one update, one version increment per step; equal values never
  merge), and rejects an empty effective collection. Strong references
  are stored; parameters are never copied, replaced, closed, or
  mutated outside the sanctioned commit path.
- **Hyperparameters are validated, never repaired.** ``lr`` and
  ``eps`` must each be a real number (``numbers.Real``; ``bool``,
  strings, and arbitrary float-coercible objects rejected), finite,
  and strictly positive. ``betas`` must be a tuple or list of exactly
  two real numbers (same type rules per element), each finite and
  satisfying ``0.0 <= beta < 1.0``. Accepted values are normalized to
  Python floats only after validation and exposed read-only.
- **Optimizer state is owned and explicit.** After all validation, one
  state entry per unique parameter is allocated **eagerly**: first
  moment ``m`` and second moment ``v`` as plain graph-free
  ``NativeTensor`` zeros (never ``NativeParameter``; fresh owning
  contiguous storage of exactly the parameter's shape/dtype/device;
  never registered in a module, so never in ``model.state_dict()``),
  plus a per-parameter integer step count starting at 0. A constructor
  failure mid-allocation closes every buffer created so far and
  touches no user parameter or gradient. The optimizer owns exactly
  its m/v buffers and nothing else.
- **Per-parameter step counts drive bias correction.** A skipped
  parameter (frozen, or ``grad is None``) never ages its moments or
  its counter, so a parameter that becomes active later takes its
  first bias-corrected update at ``t = 1``. A present zero-valued
  gradient is active and advances state, counter, and version.
  ``step_counts`` exposes the counters as an immutable tuple aligned
  with ``parameters()``.
- **``step()`` is two-phase and mutation-atomic on its public failure
  surface.** Phase 1 preflights the optimizer and every stored
  parameter as open, validates every entry's m/v state (open, exact
  shape/dtype/device), selects the active set (frozen parameters are
  skipped *before* their gradients are examined; ``grad is None``
  skipped), validates every active gradient (an open ``NativeTensor``
  of exactly the parameter's shape/dtype/device), and stages — per
  active entry, entirely at the autograd-unaware core level, with
  Python scalar exponentiation only for the bias-correction
  coefficients::

      t      = previous_step + 1
      m_new  = beta1 * m + (1 - beta1) * g
      v_new  = beta2 * v + (1 - beta2) * (g * g)
      m_hat  = m_new * reciprocal(1 - beta1 ** t)
      v_hat  = v_new * reciprocal(1 - beta2 ** t)
      update = lr * m_hat * reciprocal(sqrt(v_hat) + eps)
      parameter_new = parameter - update

  No graph node, no NumPy, no division operation; ``eps > 0``
  guarantees a positive denominator even when ``v_hat`` is zero. Any
  phase-1 failure closes every staged temporary and changes no value,
  version, moment, counter, or gradient — the same optimizer recovers
  on a later valid step. Phase 2 commits each active entry in stored
  order: ``copy_value_(parameter_new)`` (version +1), install the
  staged ``m_new``/``v_new`` as the new optimizer-owned state, commit
  the step count, then close the replaced old moment buffers; the
  staged ``parameter_new`` never persists and is always closed. Two
  narrow, honest limitations, documented rather than papered over
  with private rollback: an asynchronous interruption (e.g.
  KeyboardInterrupt) landing **between two commits** leaves earlier
  entries advanced and later ones not (each committed entry stays
  internally consistent); and **within one entry** an interruption
  between the ``copy_value_`` commit and the state installation —
  two Python operations that cannot be made indivisible — would
  advance the parameter but not its moments and count.
- **Gradients are retained.** ``step()`` never clears, replaces,
  mutates, or closes a gradient — gradients persist until
  ``zero_grad()``, which requires an open optimizer, preflights every
  stored parameter as open (never a partial clear), then delegates to
  each parameter's own ``zero_grad()`` (frozen parameters included
  harmlessly; values, versions, moments, counters, identities, and
  registrations untouched). No ``set_to_none`` option.
- **v3.7 staleness applies naturally.** A value-sensitive graph built
  before ``step()`` becomes stale after it — the existing
  deterministic error, gradients untouched — and a fresh
  forward/backward trains on the updated values. No classification
  changes.
- **Optimizer state is snapshot-able and restorable in memory
  (v3.13).** ``state_dict()`` returns a plain versioned dict — scalar
  hyperparameters, ordered positional parameter metadata, per-parameter
  step counts, and independent caller-owned NativeTensor moment
  snapshots — and ``load_state_dict(state)`` restores it into a
  compatible optimizer through full validation, staged independent
  copies, and an ordered commit, never touching a parameter's value,
  version, or gradient and never aliasing or consuming caller state.
  No file format exists (native checkpoint archives are v3.14). See
  the two method docstrings for the complete contract.
- **Lifetime is explicit.** ``close()`` (idempotent; ``with`` blocks
  work) releases every owned m/v buffer exactly once and makes
  ``step()``/``zero_grad()`` raise deterministically. It never closes
  or mutates a parameter or gradient. After close, ``parameters()``,
  ``lr``/``betas``/``eps``, and ``step_counts`` stay readable as
  plain-Python introspection (the documented choice — none of them
  can reach released native storage). There is no ``__del__``:
  each owned buffer carries the NativeTensor GC safety net, and
  correctness never depends on it — call ``close()``.
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
    validate_step_counts,
)
from .native_parameter import NativeParameter
from .native_tensor import NativeTensor, _native_copy

# The exact key set of a NativeAdam state dict (format version 1).
_STATE_KEYS = (
    "format_version", "optimizer", "lr", "betas", "eps",
    "parameters", "step_counts", "m", "v",
)


def _validated_positive_real(value, name):
    """Validate ``value`` as a real, non-bool, finite, strictly
    positive number and return it as a Python float — the NativeSGD
    learning-rate rules, shared by ``lr`` and ``eps``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value}")
    return value


def _validated_betas(betas):
    """Validate ``betas`` — a tuple or list of exactly two real,
    non-bool, finite values in ``[0.0, 1.0)`` — and return it as a
    tuple of Python floats. The constructor's contract, shared with
    ``load_state_dict`` so loaded betas can never be weaker than
    constructed ones."""
    if isinstance(betas, bool) or not isinstance(betas, (tuple, list)):
        raise TypeError(
            f"betas must be a tuple or list of two real numbers, "
            f"got {type(betas).__name__}"
        )
    if len(betas) != 2:
        raise ValueError(
            f"betas must contain exactly two values, got {len(betas)}"
        )
    validated = []
    for index, beta in enumerate(betas):
        if isinstance(beta, bool) or not isinstance(beta, numbers.Real):
            raise TypeError(
                f"betas[{index}] must be a real number, got "
                f"{type(beta).__name__}"
            )
        beta = float(beta)
        if not math.isfinite(beta):
            raise ValueError(f"betas[{index}] must be finite, got {beta}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(
                f"betas[{index}] must satisfy 0.0 <= beta < 1.0, "
                f"got {beta}"
            )
        validated.append(beta)
    return tuple(validated)


class NativeAdam:
    """Minimal native Adam: per-parameter first/second moments with
    bias correction, staged graph-free at the core level and committed
    through ``copy_value_()``, with explicit optimizer-owned state
    lifetime (``close()``). See the module docstring for the full
    contract.
    """

    __slots__ = (
        "_parameters", "_lr", "_betas", "_eps",
        "_m", "_v", "_steps", "_closed",
    )

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        # Validate every hyperparameter first — cheap, and a bad value
        # must never depend on whether the parameter iterable was
        # consumable, and must never allocate native state.
        lr = _validated_positive_real(lr, "lr")
        betas = _validated_betas(betas)
        eps = _validated_positive_real(eps, "eps")

        # Materialize the iterable exactly once, validate every entry,
        # and deduplicate by object identity in first-occurrence order
        # — the NativeSGD parameter contract, unchanged.
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
                raise TypeError(
                    f"parameters[{position}] must be a NativeParameter, "
                    f"got {type(entry).__name__}"
                )
            if entry.closed:
                raise RuntimeError(
                    f"parameters[{position}] has been closed"
                )
            if id(entry) in seen:
                continue  # identity dedup: aliases share one state entry
            seen.add(id(entry))
            unique.append(entry)
        if not unique:
            raise ValueError(
                "NativeAdam requires at least one parameter"
            )

        # Eager state allocation, only after everything above passed:
        # one m/v pair per unique parameter, plain graph-free
        # NativeTensor zeros of exactly the parameter's metadata, owned
        # exclusively by this optimizer. A mid-allocation failure
        # releases every buffer created so far and touches nothing else.
        m_buffers = []
        v_buffers = []
        try:
            for parameter in unique:
                m_buffers.append(NativeTensor.zeros(
                    parameter.shape,
                    dtype=parameter.dtype, device=parameter.device,
                ))
                v_buffers.append(NativeTensor.zeros(
                    parameter.shape,
                    dtype=parameter.dtype, device=parameter.device,
                ))
        except BaseException:
            for buffer in m_buffers:
                buffer.close()
            for buffer in v_buffers:
                buffer.close()
            raise

        self._parameters = tuple(unique)
        self._lr = lr
        self._betas = betas
        self._eps = eps
        self._m = m_buffers
        self._v = v_buffers
        self._steps = [0] * len(unique)
        self._closed = False

    # -- read-only introspection (plain Python; readable after close) ------

    @property
    def lr(self):
        """The validated learning rate, as a Python float. Read-only."""
        return self._lr

    @property
    def betas(self):
        """The validated ``(beta1, beta2)`` coefficients, as a tuple of
        Python floats. Read-only."""
        return self._betas

    @property
    def eps(self):
        """The validated denominator epsilon, as a Python float.
        Read-only."""
        return self._eps

    @property
    def closed(self):
        """True once ``close()`` has run. Readable even after close."""
        return self._closed

    @property
    def step_counts(self):
        """The per-parameter step counters, as an immutable tuple of
        Python ints aligned with ``parameters()``. Each counter starts
        at 0 and increments exactly once per committed update of its
        parameter; skipped (frozen / ``grad=None``) parameters never
        advance. Readable after close (plain ints, no native state)."""
        return tuple(self._steps)

    def parameters(self):
        """The stored parameters: unique by identity, first-occurrence
        order — the exact objects the caller supplied, never copies.
        Returns a new list (mutating it cannot affect the optimizer).
        Remains an introspection-only snapshot after ``close()``."""
        return list(self._parameters)

    # -- lifetime -----------------------------------------------------------

    def _require_open(self):
        if self._closed:
            raise RuntimeError("this NativeAdam optimizer has been closed")

    def _require_parameters_open(self, where):
        """Preflight every stored parameter as still open — checked for
        the whole collection first, so a closed parameter fails
        deterministically regardless of earlier entries' state."""
        for index, parameter in enumerate(self._parameters):
            if parameter.closed:
                raise RuntimeError(
                    f"{where}: parameters[{index}] has been closed"
                )

    def _validate_state_buffers(self, where):
        """Preflight every entry's persistent m/v as open and still
        matching its parameter's metadata, and every step count as a
        non-bool non-negative int — the optimizer-owned invariants,
        verified before anything mutates or is exposed. The caller has
        already preflighted the parameters as open."""
        for index, parameter in enumerate(self._parameters):
            for label, state in (("m", self._m[index]), ("v", self._v[index])):
                if state.closed:
                    raise RuntimeError(
                        f"{where}: the {label} state for "
                        f"parameters[{index}] has been closed"
                    )
                if (
                    state.shape != parameter.shape
                    or state.dtype != parameter.dtype
                    or state.device != parameter.device
                ):
                    raise ValueError(
                        f"{where}: the {label} state for "
                        f"parameters[{index}] is "
                        f"{state.shape}/{state.dtype}/{state.device}, "
                        f"the parameter is "
                        f"{parameter.shape}/{parameter.dtype}/"
                        f"{parameter.device}"
                    )
            count = self._steps[index]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"{where}: the step count for parameters[{index}] "
                    f"is not a non-negative int: {count!r}"
                )

    def close(self):
        """Release every optimizer-owned m/v buffer exactly once and
        mark the optimizer closed: ``step()`` and ``zero_grad()``
        reject deterministically afterwards. Idempotent. Never closes
        or mutates a parameter or a gradient — caller-owned objects
        stay exactly as they were. Returns None."""
        if self._closed:
            return
        self._closed = True
        for buffer in self._m:
            buffer.close()
        for buffer in self._v:
            buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    # -- the optimizer surface ----------------------------------------------

    def step(self):
        """One Adam update for every stored parameter that is trainable
        and has a gradient (see the module docstring for the exact
        equations). Frozen and gradient-less parameters are skipped
        untouched — values, versions, moments, and counters all
        preserved. Two-phase: every validation and every staged native
        computation completes before the first parameter mutates, so
        the public failure surface changes no value, version, moment,
        counter, or gradient. Returns None."""
        self._require_open()

        # Phase 1a: every stored parameter must still be open.
        self._require_parameters_open("NativeAdam.step()")

        # Phase 1b (state): every entry's persistent m/v must be open
        # and still match its parameter's metadata — optimizer-owned
        # invariants, verified before anything mutates.
        self._validate_state_buffers("NativeAdam.step()")

        # Phase 1b (gradients): select the active set and validate every
        # gradient completely before anything is staged or mutated.
        # Frozen parameters are skipped before their grad is examined.
        active = []
        for index, parameter in enumerate(self._parameters):
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            if grad is None:
                continue
            if not isinstance(grad, NativeTensor):
                raise TypeError(
                    f"NativeAdam.step(): parameters[{index}].grad must "
                    f"be a NativeTensor, got {type(grad).__name__}"
                )
            if grad.closed:
                raise RuntimeError(
                    f"NativeAdam.step(): parameters[{index}].grad has "
                    f"been closed"
                )
            if grad.shape != parameter.shape:
                raise ValueError(
                    f"NativeAdam.step(): parameters[{index}].grad shape "
                    f"{grad.shape} does not match the parameter shape "
                    f"{parameter.shape}"
                )
            if grad.dtype != parameter.dtype or grad.device != parameter.device:
                raise ValueError(
                    f"NativeAdam.step(): parameters[{index}].grad "
                    f"dtype/device {grad.dtype}/{grad.device} does not "
                    f"match the parameter "
                    f"{parameter.dtype}/{parameter.device}"
                )
            active.append((index, parameter, grad))

        # Phase 1c: stage every entry's next values natively at the
        # core level — the autograd-unaware layer, so no graph can
        # exist. A failure here releases everything staged so far and
        # has mutated nothing.
        staged = []
        try:
            for index, parameter, grad in active:
                staged.append(self._stage_entry(index, parameter, grad))
        except BaseException:
            for _, _, m_new, v_new, parameter_new, _ in staged:
                m_new.close()
                v_new.close()
                parameter_new.close()
            raise

        # Phase 2: commit each active entry in stored order — the
        # parameter through the one sanctioned mutation path (version
        # +1), then the optimizer-owned state references, then the step
        # count, then close the replaced old moments. After the
        # completed preflight these calls cannot fail through any
        # public surface (see the module docstring for the honest
        # asynchronous-interruption caveats). The staged parameter_new
        # is always released; a staged moment is released only if it
        # was never installed.
        try:
            for index, parameter, m_new, v_new, parameter_new, next_step in staged:
                parameter.copy_value_(parameter_new)
                old_m = self._m[index]
                old_v = self._v[index]
                self._m[index] = m_new
                self._v[index] = v_new
                self._steps[index] = next_step
                old_m.close()
                old_v.close()
        finally:
            for index, _, m_new, v_new, parameter_new, _ in staged:
                parameter_new.close()
                if self._m[index] is not m_new:
                    m_new.close()
                if self._v[index] is not v_new:
                    v_new.close()

    def _stage_entry(self, index, parameter, grad):
        """Stage one active entry's next values, entirely at the
        NativeTensorCore level: fresh owning ``m_new``/``v_new``/
        ``parameter_new`` tensors (independent of every parameter,
        gradient, and existing state buffer) plus the next step count.
        Transient cores are closed on every path; on failure the
        partially built m_new/v_new are closed too and nothing has
        been mutated."""
        parameter_core = parameter._require_open()
        grad_core = grad._require_open()
        m_core = self._m[index]._require_open()
        v_core = self._v[index]._require_open()
        beta1, beta2 = self._betas
        next_step = self._steps[index] + 1
        dtype, device = parameter_core.dtype, parameter_core.device

        def scalar(value):
            # A broadcast scalar core — the same composition NativeSGD
            # and the engine's own backward math use.
            return cpp.NativeTensorCore.full((), value, dtype=dtype,
                                             device=device)

        m_new_core = None
        v_new_core = None
        transients = []
        try:
            # m_new = beta1 * m + (1 - beta1) * g
            beta1_scalar = scalar(beta1)
            transients.append(beta1_scalar)
            decayed_m = m_core.multiply(beta1_scalar)
            transients.append(decayed_m)
            beta1_complement = scalar(1.0 - beta1)
            transients.append(beta1_complement)
            fresh_m = grad_core.multiply(beta1_complement)
            transients.append(fresh_m)
            m_new_core = decayed_m.add(fresh_m)

            # v_new = beta2 * v + (1 - beta2) * (g * g)
            beta2_scalar = scalar(beta2)
            transients.append(beta2_scalar)
            decayed_v = v_core.multiply(beta2_scalar)
            transients.append(decayed_v)
            grad_squared = grad_core.multiply(grad_core)
            transients.append(grad_squared)
            beta2_complement = scalar(1.0 - beta2)
            transients.append(beta2_complement)
            fresh_v = grad_squared.multiply(beta2_complement)
            transients.append(fresh_v)
            v_new_core = decayed_v.add(fresh_v)

            # Bias correction: the coefficients 1 - beta ** t are the
            # one place Python scalar exponentiation is used; their
            # reciprocals are taken natively (no division operation).
            correction1 = scalar(1.0 - beta1 ** next_step)
            transients.append(correction1)
            inverse_correction1 = correction1.reciprocal()
            transients.append(inverse_correction1)
            m_hat = m_new_core.multiply(inverse_correction1)
            transients.append(m_hat)

            correction2 = scalar(1.0 - beta2 ** next_step)
            transients.append(correction2)
            inverse_correction2 = correction2.reciprocal()
            transients.append(inverse_correction2)
            v_hat = v_new_core.multiply(inverse_correction2)
            transients.append(v_hat)

            # update = lr * m_hat * reciprocal(sqrt(v_hat) + eps);
            # eps > 0 keeps the denominator positive even at v_hat = 0.
            root = v_hat.sqrt()
            transients.append(root)
            eps_scalar = scalar(self._eps)
            transients.append(eps_scalar)
            denominator = root.add(eps_scalar)
            transients.append(denominator)
            inverse_denominator = denominator.reciprocal()
            transients.append(inverse_denominator)
            lr_scalar = scalar(self._lr)
            transients.append(lr_scalar)
            scaled_m_hat = m_hat.multiply(lr_scalar)
            transients.append(scaled_m_hat)
            update = scaled_m_hat.multiply(inverse_denominator)
            transients.append(update)

            parameter_new_core = parameter_core.subtract(update)
        except BaseException:
            if m_new_core is not None:
                m_new_core.close()
            if v_new_core is not None:
                v_new_core.close()
            raise
        finally:
            for core in transients:
                core.close()

        return (
            index,
            parameter,
            NativeTensor._from_core(m_new_core),
            NativeTensor._from_core(v_new_core),
            NativeTensor._from_core(parameter_new_core),
            next_step,
        )

    # -- optimizer state (v3.13: in-memory only; files are v3.14) ----------

    def state_dict(self):
        """The optimizer's restorable in-memory state, as a plain
        independent dict (format version 1)::

            {"format_version": 1, "optimizer": "NativeAdam",
             "lr": <float>, "betas": (<float>, <float>), "eps": <float>,
             "parameters": ({"shape", "dtype", "device"}, ...),
             "step_counts": (<int>, ...),
             "m": [<NativeTensor>, ...], "v": [<NativeTensor>, ...]}

        Everything is positional, in the optimizer's deterministic
        identity-deduplicated first-occurrence parameter order (shared
        aliases appear once); no object identity, name, parameter
        value, gradient, or graph data is included. The ``m``/``v``
        entries are **caller-owned snapshots**: plain graph-free
        ``requires_grad=False`` NativeTensors (never NativeParameter),
        each an independent owning contiguous native copy of the
        optimizer's moment — sharing no storage with the optimizer, the
        parameters, the gradients, or each other. Closing a snapshot
        never affects the optimizer; the caller releases the snapshots
        (``close()`` each) when done, and repeated calls return
        independently owned snapshots.

        Preflight: the optimizer must be open, every stored parameter
        open, every internal m/v open and metadata-matched, and every
        step count a valid non-negative int — a violation raises
        deterministically before anything is created. If snapshotting
        fails partway, every snapshot created by this call is closed
        before the error propagates (never left to garbage collection),
        internal state and parameters are untouched, and the optimizer
        remains usable. No autograd graph is built and no NumPy touches
        the copies (the native copy path)."""
        where = "NativeAdam.state_dict()"
        self._require_open()
        self._require_parameters_open(where)
        self._validate_state_buffers(where)
        snapshots = {"m": [], "v": []}
        try:
            for label, buffers in (("m", self._m), ("v", self._v)):
                for buffer in buffers:
                    snapshots[label].append(NativeTensor._from_core(
                        _native_copy(buffer._require_open())
                    ))
        except BaseException:
            for label in ("m", "v"):
                for snapshot in snapshots[label]:
                    snapshot.close()
            raise
        return {
            "format_version": FORMAT_VERSION,
            "optimizer": "NativeAdam",
            "lr": self._lr,
            "betas": self._betas,
            "eps": self._eps,
            "parameters": parameter_metadata(self._parameters),
            "step_counts": tuple(self._steps),
            "m": snapshots["m"],
            "v": snapshots["v"],
        }

    def _validate_moment_entries(self, entries, label, where):
        """Validate one input moment collection for loading: a
        tuple/list of exactly one **open plain NativeTensor** per
        stored parameter — a NativeParameter (or anything else,
        the stable framework's Tensor included) is rejected — each
        exactly matching its parameter's shape/dtype/device, with no
        casting, reshaping, broadcasting, or device transfer. Raises
        naming the failing entry; touches nothing."""
        if not isinstance(entries, (tuple, list)):
            raise TypeError(
                f"{where}: state[{label!r}] must be a tuple or list of "
                f"NativeTensor, got {type(entries).__name__}"
            )
        if len(entries) != len(self._parameters):
            raise ValueError(
                f"{where}: state[{label!r}] holds {len(entries)} "
                f"tensors, this optimizer stores "
                f"{len(self._parameters)} parameters"
            )
        for index, (entry, parameter) in enumerate(
            zip(entries, self._parameters)
        ):
            if not isinstance(entry, NativeTensor) or isinstance(
                entry, NativeParameter
            ):
                raise TypeError(
                    f"{where}: state[{label!r}][{index}] must be a plain "
                    f"NativeTensor, got {type(entry).__name__}"
                )
            if entry.closed:
                raise RuntimeError(
                    f"{where}: state[{label!r}][{index}] has been closed"
                )
            if entry.shape != parameter.shape:
                raise ValueError(
                    f"{where}: state[{label!r}][{index}] has shape "
                    f"{entry.shape}, the stored parameter is "
                    f"{parameter.shape}"
                )
            if (
                entry.dtype != parameter.dtype
                or entry.device != parameter.device
            ):
                raise ValueError(
                    f"{where}: state[{label!r}][{index}] is "
                    f"{entry.dtype}/{entry.device}, the stored parameter "
                    f"is {parameter.dtype}/{parameter.device}"
                )

    def load_state_dict(self, state):
        """Restore optimizer state from a ``state_dict()`` dict:
        hyperparameters, per-parameter step counts, and both moment
        collections — mapped positionally onto this optimizer's stored
        parameters.

        **Phase 1 — validation, no mutation.** The optimizer must be
        open (loading never reopens a closed one), every stored
        parameter open, and the current internal state intact; then the
        complete input is validated: a plain dict with exactly the
        schema keys, ``format_version == 1``, the ``"NativeAdam"`` tag,
        lr/betas/eps under the constructors' full contracts, parameter
        metadata matching position by position, step counts (non-bool
        non-negative ints, one per parameter), and every ``m``/``v``
        entry an open plain NativeTensor of exactly the parameter's
        shape/dtype/device (sequence fields accept tuple or list).

        **Phase 2 — staging.** Every input moment is copied into a
        fresh optimizer-owned NativeTensor by the native copy path —
        the caller's tensors are never adopted, retained, closed, or
        mutated, and the supplied dict is read-only throughout. A
        staging failure closes every staged copy and changes nothing.

        **Phase 3 — commit.** lr, betas, eps, the step counters, and
        the m/v state are replaced, and the replaced old moment buffers
        are closed only after the new state is installed. Parameters
        are untouched at every phase: no value, version, gradient
        (identity or value), ``requires_grad``, registration, or alias
        changes, and no retained autograd graph becomes stale — the
        v3.7 guard keys on parameter versions, which this method never
        moves. After success the caller's snapshots remain open and
        share no storage with the optimizer (closing either side never
        affects the other).

        One narrow, honest limitation: the commit is several Python
        attribute assignments that cannot be made indivisible, so an
        asynchronous interruption (e.g. KeyboardInterrupt) landing
        mid-commit could leave the scalars replaced but the moments
        not (each installed piece stays internally consistent, and the
        GC safety net eventually reclaims any not-yet-closed old
        buffer). No ordinary public failure reaches the commit.

        All three phases run under the shared native state-transaction
        guard (Phase G, milestone G5), so a checkpoint load that replaces
        the model, this optimizer, and the generators cannot be
        interleaved by another participating state load. The guard is
        reentrant, so the checkpoint transaction holds it and this method
        re-enters it; this path takes no generator lock, so it cannot
        invert the universal order.
        Returns None."""
        where = "NativeAdam.load_state_dict()"
        with state_transaction():
            self._load_state_dict_locked(state, where)

    def _load_state_dict_locked(self, state, where):
        """The validate → stage → commit body, with the shared
        state-transaction guard already held."""
        self._require_open()
        self._require_parameters_open(where)
        self._validate_state_buffers(where)
        validate_state_schema(state, "NativeAdam", _STATE_KEYS, where)
        lr = _validated_positive_real(state["lr"], "lr")
        betas = _validated_betas(state["betas"])
        eps = _validated_positive_real(state["eps"], "eps")
        validate_parameter_metadata(
            state["parameters"], self._parameters, where
        )
        validate_step_counts(
            state["step_counts"], len(self._parameters), where
        )
        self._validate_moment_entries(state["m"], "m", where)
        self._validate_moment_entries(state["v"], "v", where)
        step_counts = [int(count) for count in state["step_counts"]]

        # Phase 2: stage independent optimizer-owned copies of every
        # input moment. Nothing has mutated yet, so a failure only has
        # staged copies to release.
        staged = {"m": [], "v": []}
        try:
            for label in ("m", "v"):
                for entry in state[label]:
                    staged[label].append(NativeTensor._from_core(
                        _native_copy(entry._require_open())
                    ))
        except BaseException:
            for label in ("m", "v"):
                for copy in staged[label]:
                    copy.close()
            raise

        # Phase 3: commit, then release the replaced old buffers.
        old_m = self._m
        old_v = self._v
        self._lr = lr
        self._betas = betas
        self._eps = eps
        self._steps = step_counts
        self._m = staged["m"]
        self._v = staged["v"]
        for buffer in old_m:
            buffer.close()
        for buffer in old_v:
            buffer.close()

    def zero_grad(self):
        """Clear every stored parameter's gradient to ``None`` via the
        parameter's own ``zero_grad()`` (frozen parameters included
        harmlessly; values, versions, moments, step counts,
        identities, ``requires_grad``, and registrations untouched;
        grad objects dropped, never closed). Requires an open
        optimizer, and preflights that every stored parameter is still
        open — failing before clearing anything otherwise, never a
        partial clear. Returns None."""
        self._require_open()
        for index, parameter in enumerate(self._parameters):
            if parameter.closed:
                raise RuntimeError(
                    f"NativeAdam.zero_grad(): parameters[{index}] has "
                    f"been closed"
                )
        for parameter in self._parameters:
            parameter.zero_grad()

    def __repr__(self):
        if self._closed:
            return "NativeAdam(closed)"
        return (
            f"NativeAdam(lr={self._lr}, betas={self._betas}, "
            f"eps={self._eps}, parameters={len(self._parameters)})"
        )
