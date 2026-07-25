"""Native batch normalization — ``NativeBatchNorm1d`` (Phase F,
milestone F3: the first **stateful** native numerical module) and
``NativeBatchNorm2d`` (milestone F4: the NCHW shape), over **one**
shared private implementation. See
docs/native_normalization_design.md §6-§10.

Batch normalization normalizes each feature across the *batch*, so —
unlike ``NativeLayerNorm`` — other samples participate, the layer carries
**persistent running statistics**, and its behavior differs between
training and evaluation mode. That makes it the first native module that
owns mutable numerical state, and the whole point of this file is to make
that state safe: graph-free, atomically updated, identity-preserving, and
never readable by an autograd graph that could observe it changing.

Composed, never primitive
-------------------------

Like LayerNorm, the whole layer is **composed from existing
differentiable ``NativeTensor`` operations** — ``mean``, ``subtract``,
``multiply``, ``add``, ``sqrt``, ``reciprocal``, ``reshape``,
``contiguous_copy``, and ``detach``. F3 adds **no** C++ code, no
normalization kernel, no C ABI symbol, no ctypes declaration, no
``NativeTensorCore`` method, no ``NativeTensor.batch_norm`` operation, and
no hand-written BatchNorm backward: because forward is built entirely
from existing differentiable operations, the existing Python-managed
native autograd engine **is** the backward implementation, and gradients
flow to the input (and to ``gamma``/``beta``) **through the batch mean
and the batch variance** for free. That is not an optimization to be
traded away later — detaching the training statistics would give a
different, wrong gradient.

**Training forward** (``self.training is True``)::

    batch_mean = mean(input, axis=0, keepdims=True)          # (1, C)
    centered   = input - batch_mean
    batch_var  = mean(centered * centered, axis=0, keepdims=True)
    inv_std    = reciprocal(sqrt(batch_var + eps))
    normalized = centered * inv_std
    output     = normalized * gamma + beta

The variance is the **population** variance (divide by the batch size, no
Bessel correction), matching the stable ``tensorforge.nn.BatchNorm1d``,
and **epsilon is added inside the square root** — ``sqrt(var + eps)``,
never ``sqrt(var) + eps``. There is no ``divide`` operation in the native
line, so the inverse standard deviation is built from ``reciprocal``.

**Running-statistics update.** The *same* population ``batch_var`` that
normalized the batch also drives the update — nothing is recomputed::

    running_mean <- (1 - momentum) * running_mean + momentum * batch_mean
    running_var  <- (1 - momentum) * running_var  + momentum * batch_var

Both replacement values are built as **independent, graph-free, owning
``(num_features,)`` native state**: the batch statistics enter through
``detach()`` (a native storage-to-storage copy — no NumPy round trip), so
no gradient path can reach the update and no autograd graph is built for
it. At the boundaries the convention is exact: ``momentum=0.0`` leaves
both running values numerically unchanged, and ``momentum=1.0`` makes them
exactly the current population batch statistics.

**Evaluation forward** (``self.training is False``) uses the stored
statistics instead of the batch's own, so a single sample normalizes
consistently, and updates nothing::

    normalized = (input - running_mean_snapshot) * inv_std_snapshot
    output     = normalized * gamma + beta

Mutable-buffer graph safety (§7)
--------------------------------

**A live, registered ``running_mean``/``running_var`` buffer is never a
graph operand.** A buffer is a plain ``NativeTensor`` with no value
version, so the stale-graph guard cannot see it change; and ``multiply``'s
backward rereads its *other* operand's **current** core. If an eval graph
held the registered buffer, a later training step, ``load_state_dict()``,
or checkpoint load would silently change that graph's gradient — or make
it read a closed core. So evaluation first takes **independent, owning,
contiguous, graph-free, gradient-free** ``(1, C)`` snapshots of both
buffers (through the native copy path — no NumPy, no borrowing view), and
the graph reads only those. A later **running-buffer** mutation therefore
cannot reach it, which is exactly why buffers stay unversioned.

**What that does and does not promise.** A training step, a buffer-only
``load_state_dict()``, and a buffer-only ``load_native_checkpoint()``
over these same registered objects each leave an already-built eval
graph valid and numerically unchanged. A load that *also* replaces
``gamma``/``beta`` is a different matter: those are ``NativeParameter``s
that the eval graph does hold directly as ``multiply`` operands, so their
versions move and the pre-existing v3.7 stale-value guard
**intentionally** rejects the old graph — the parameter contract working
exactly as it does for every other layer, never a running-buffer effect.
BatchNorm neither bypasses nor weakens it.

The snapshot values a backward must read live **exactly as long as the
graph that reads them**: they are adopted as the output node's
``graph_resources`` — the same D9 contract that owns MaxPool2d's winner
buffer and Phase E's saved probabilities — so ``retain_graph=True`` keeps
them, an abandoned graph still frees them, and they are released exactly
once when the graph history is. No second lifetime system and no custom
backward is introduced to achieve that.

Atomic running-statistics transaction (§8)
------------------------------------------

The two buffers describe **one** statistical state, so they advance
together or not at all, through the private F1 primitive
(``_native_state.replace_native_state``) — no fake state dictionary, no
internal ``load_state_dict()``, no attribute reassignment, and no second
transaction system. Both replacement values are staged before anything
mutates; both buffers' Python identities survive; the old cores stay valid
until the commit succeeds and are then closed exactly once; a failure
before the commit boundary restores both buffers and closes every staged
core; and **no parameter version moves**, so a running update never makes
an existing graph stale.

Forward ordering is what makes a *failed* forward harmless: validate all
live state, build the complete differentiable output graph, prepare both
graph-free replacement values, commit them atomically, and only then
return the already-built output. A failure anywhere before the commit
leaves the running statistics exactly as they were, and every unadopted
native temporary is closed immediately — not left to cyclic garbage
collection, which the ``sqrt``/``reciprocal`` result-capturing closures
would otherwise require.

Construction — ``NativeBatchNorm1d(num_features, eps=1e-5, momentum=0.1)``
-------------------------------------------------------------------------

- ``num_features`` must be a positive plain ``int`` (``bool``, floats,
  strings, ``None``, and NumPy integer objects rejected, as everywhere in
  the native line).
- ``eps`` must be a positive real (``bool`` and non-real rejected); stored
  as a Python ``float``.
- ``momentum`` must be a real in ``[0, 1]`` (``bool``, non-real, and NaN
  rejected); stored as a Python ``float``.
- Every argument is validated **before** the first native allocation, so a
  rejected construction leaves the native live-storage count unchanged.
- ``gamma`` (ones) and ``beta`` (zeros) are always created as
  ``NativeParameter``s of shape ``(num_features,)``, registered in that
  order, both requiring gradients at version 0. There is deliberately no
  ``affine=False`` mode.
- ``running_mean`` (zeros) and ``running_var`` (ones) are always created
  as plain owning contiguous gradient-free ``NativeTensor``s of shape
  ``(num_features,)`` and registered in that order through
  ``register_buffer(name, tensor, persistent=True)``. They are never
  parameters: ``parameters()`` never yields them, no optimizer sees them,
  and no gradient flows through them. There is deliberately no
  ``track_running_stats`` option and no ``num_batches_tracked`` counter.
- State order is therefore exactly ``gamma``, ``beta``, ``running_mean``,
  ``running_var`` — parameters first, persistent buffers second.
- If any later allocation or registration fails, every native object this
  constructor already created is closed exactly once, deterministically,
  and the original exception propagates.

NumPy appears only in the constructor, as host-side data preparation
feeding ``NativeParameter`` (the ``from_array`` entry boundary) — the
``NativeLinear``/``NativeLayerNorm`` precedent. Nothing numerical in
forward, backward, or the running-state update touches NumPy or
``to_numpy()``.

**Input contract**: ``forward(input)`` requires an open ``NativeTensor``
(a ``NativeParameter`` is accepted as the subclass it is) of exactly rank
2 and shape ``(N, num_features)``; the stable framework's ``Tensor``,
NumPy arrays, lists, tuples, scalars, arbitrary objects, closed tensors,
and any other rank or feature count are rejected — naming the configured
feature count and the received shape — before any graph node is built.
``gamma``, ``beta``, ``running_mean``, and ``running_var`` are checked
open, correctly shaped, and dtype/device-matched first, and the running
buffers are re-checked as owning, contiguous, gradient-free, registered
persistent buffers. Validation mutates nothing.

The output is a **fresh, owning, row-major contiguous** ``NativeTensor``
of shape ``(N, C)`` in both modes, independent of the input, the
parameters, and the buffer storage, and never a ``NativeParameter`` or a
borrowing view. The module stores no per-forward tensor attributes.
Fully separate from ``tensorforge.nn.BatchNorm1d``; float64/cpu only.

One implementation, two public shapes
-------------------------------------

``_NativeBatchNorm`` below holds every piece of behavior — validation,
both modes' mathematics, the snapshot rule, the running update, the
transaction, and the cleanup. The two public classes declare **only**
shape configuration and inherit every method by function identity:

===========================  =================  ============================
                             ``NativeBatchNorm1d``  ``NativeBatchNorm2d``
===========================  =================  ============================
``_INPUT_NDIM``              2                  4
``_REDUCTION_AXES``          ``(0,)``           ``(0, 2, 3)``
``_TRAILING_DIMS``           0                  2
``_LAYOUT``                  ``(N, C)``         ``(N, C, H, W)``
``_CHANNELS_LAST``           ``None``           ``(0, 2, 3, 1)``
===========================  =================  ============================

So the 4-D shape reduces over the batch **and both spatial axes** and
never over channels — each channel gets one population mean and one
population variance over all ``N * H * W`` of its values — while its
per-channel broadcast layout is ``(1, C, 1, 1)``. The persistent running
buffers stay ``(C,)`` for both shapes. ``_CHANNELS_LAST`` is the only
piece of configuration that is not purely about ranks and axes; see
``_affine`` for what it is for and why it is not a public layout mode.

``_NativeBatchNorm`` is private: it is not exported, not a public
normalization subsystem, and not a stable base class.
"""

import numbers

import numpy as np

from . import _native_state
from .native_module import NativeModule
from .native_parameter import NativeParameter
from .native_tensor import NativeTensor, _native_copy


def _validate_num_features(num_features):
    """``num_features`` must be a real positive ``int`` — ``bool`` and
    integer-like objects (NumPy integers included) are rejected, matching
    the strict count validation ``NativeLinear`` established. A wrong
    *type* raises ``TypeError``; a right type with a bad *value* raises
    ``ValueError``."""
    if not isinstance(num_features, int) or isinstance(num_features, bool):
        raise TypeError(
            f"num_features must be an int, got {type(num_features).__name__}"
        )
    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features}")


def _validate_eps(eps):
    """``eps`` must be a positive real (``bool`` and non-real rejected)."""
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError(f"eps must be a real number, got {type(eps).__name__}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}")


def _validate_momentum(momentum):
    """``momentum`` must be a real in the inclusive range ``[0, 1]``.

    NaN needs no special case: ``0.0 <= nan <= 1.0`` is False, so a NaN
    momentum falls out of the range check as a ``ValueError`` rather than
    silently poisoning every future running statistic."""
    if isinstance(momentum, bool) or not isinstance(momentum, numbers.Real):
        raise TypeError(
            f"momentum must be a real number, got {type(momentum).__name__}"
        )
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be in [0, 1], got {momentum!r}")


def _inverse_permutation(permutation):
    """The permutation that undoes ``permutation``.

    Derived rather than configured so the "move channels last" and "put
    them back" halves can never drift apart — a silent transposition bug
    would be invisible in the output's *shape* and wrong only in its
    values."""
    inverse = [0] * len(permutation)
    for position, axis in enumerate(permutation):
        inverse[axis] = position
    return tuple(inverse)


def _adopt_graph_resources(node, resources):
    """Give ``node``'s graph *history* ownership of ``resources``.

    This is the D9 ``graph_resources`` contract, reached directly because
    the final node of a composed forward is produced by an ordinary
    operation (``add``) that takes no resource argument. It is a private
    adoption detail and deliberately **not** a new public operation, a new
    lifetime system, or a second autograd implementation: the resources
    are released at exactly the points the engine already releases graph
    history — a one-shot ``backward()``'s cleanup or ``close()`` — exactly
    once, and are retained under ``retain_graph=True``.

    Returns ``True`` when a graph was actually built (so the resources are
    now graph-owned) and ``False`` when the forward produced a plain
    no-grad leaf, in which case nothing will ever read them and the caller
    must release them itself.
    """
    if node._is_leaf or not node._requires_grad:
        return False
    node._graph_resources = node._graph_resources + tuple(resources)
    return True


class _NativeBatchNorm(NativeModule):
    """The shared, private batch-normalization implementation.

    Every piece of behavior lives here — constructor validation, parameter
    and buffer creation, forward-state validation, the training and
    evaluation mathematics, the graph-safe evaluation snapshots, the
    graph-free running-statistics update, the atomic two-buffer
    transaction, and the deterministic failure cleanup. A public subclass
    supplies only the *shape* configuration below, so the rank it accepts
    stays part of its own contract instead of being hidden behind silent
    rank-polymorphism.

    Private on purpose: it is not exported, not a public normalization
    subsystem, and not a stable base class.
    """

    # -- shape configuration, supplied privately by the public subclass --
    #
    # ``_INPUT_NDIM``      the one input rank this shape accepts.
    # ``_REDUCTION_AXES``  the axes the batch statistics reduce over, in
    #                      the order the sequential single-axis
    #                      ``NativeTensor.mean`` calls apply them. Every
    #                      reduction uses ``keepdims=True``, so the axis
    #                      numbers stay valid across the whole sequence
    #                      (no tuple-axis reduction is added).
    # ``_TRAILING_DIMS``   how many trailing size-1 dimensions the
    #                      per-feature broadcast shape carries after
    #                      ``(1, C)``.
    # ``_LAYOUT``          the accepted layout, for error messages.
    # ``_CHANNELS_LAST``   the axis permutation that moves the channel
    #                      axis to the *trailing* position for the affine
    #                      application, or ``None`` when it is already
    #                      trailing. See ``_affine`` for why this exists.
    #                      The inverse is derived, never configured.
    _INPUT_NDIM = None
    _REDUCTION_AXES = ()
    _TRAILING_DIMS = 0
    _LAYOUT = ""
    _CHANNELS_LAST = None

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        # Validate every Python argument before any native allocation, so
        # a rejected construction never creates storage it abandons — the
        # live-storage count is unchanged on rejection.
        _validate_num_features(num_features)
        _validate_eps(eps)
        _validate_momentum(momentum)
        super().__init__()
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        # The per-feature broadcast layout: (1, C) for the 2-D shape,
        # (1, C, 1, 1) for NCHW.
        self._stat_shape = (1, num_features) + (1,) * self._TRAILING_DIMS
        # The affine round trip's return leg, derived from its outbound
        # permutation so the two can never disagree.
        self._channels_first = (
            None if self._CHANNELS_LAST is None
            else _inverse_permutation(self._CHANNELS_LAST)
        )

        # A stateful module owns up to four native objects. Track each one
        # the moment it exists so a later failure closes exactly the
        # objects *this* constructor created, exactly once, most-recent
        # first — never something it did not create, and never left to
        # garbage collection. The original exception is preserved by the
        # bare re-raise, and the half-built module is discarded with it.
        created = []
        try:
            # gamma=ones, beta=zeros. NumPy here is host-side data
            # preparation feeding NativeParameter (the from_array entry
            # boundary), the NativeLinear/NativeLayerNorm precedent; no
            # graph is built and no native compute runs. Registration
            # order is gamma, then beta.
            gamma = NativeParameter(np.ones(num_features), requires_grad=True)
            created.append(gamma)
            self.gamma = gamma
            beta = NativeParameter(np.zeros(num_features), requires_grad=True)
            created.append(beta)
            self.beta = beta
            # running_mean=zeros, running_var=ones — plain owning
            # contiguous gradient-free NativeTensors built natively (no
            # NumPy at all), registered as *persistent* buffers in that
            # order so state_dict()/checkpoints carry them after the
            # parameters. They are never NativeParameters.
            running_mean = NativeTensor.zeros((num_features,))
            created.append(running_mean)
            self.register_buffer("running_mean", running_mean, persistent=True)
            running_var = NativeTensor.full((num_features,), 1.0)
            created.append(running_var)
            self.register_buffer("running_var", running_var, persistent=True)
        except BaseException:
            for tensor in reversed(created):
                tensor.close()
            raise

    # -- forward ---------------------------------------------------------

    def forward(self, input):
        """Normalize ``input`` over the batch and apply the affine
        ``gamma``/``beta``.

        Training mode uses this batch's own differentiable statistics and
        atomically advances both running buffers once; evaluation mode
        uses immutable graph-free snapshots of those buffers and changes
        nothing. See the module docstring for the full contract."""
        gamma, beta, running_mean, running_var = self._validate_forward(input)

        # Two tracking lists with different lifetimes.
        #
        # ``graph_temporaries`` — everything the *returned graph* needs,
        # plus the output itself. On the success path nothing here is
        # closed and the list is simply dropped with the frame; on failure
        # every entry is closed exactly once, most-recent first, because
        # nothing adopted them: the grad-building path puts each
        # ``sqrt``/``reciprocal`` result node into a reference cycle (its
        # backward closure captures the node itself), which reference
        # counting alone cannot reclaim, so a failure after those exist
        # would otherwise leak native storage until cyclic GC.
        #
        # ``scratch`` — pure working values no one needs after this call
        # (the eps constant in eval mode, the detached statistics, and the
        # momentum-blend intermediates). Closed on *every* path in the
        # finally, so a training step leaves no per-iteration growth.
        #
        # The caller's input, gamma, beta, and both running buffers are
        # never tracked, so they are never closed.
        graph_temporaries = []
        scratch = []

        def keep(tensor):
            graph_temporaries.append(tensor)
            return tensor

        def temp(tensor):
            scratch.append(tensor)
            return tensor

        try:
            if self.training:
                return self._training_forward(
                    input, gamma, beta, running_mean, running_var, keep, temp
                )
            return self._eval_forward(
                input, gamma, beta, running_mean, running_var, keep, temp
            )
        except BaseException:
            # close() is idempotent and releases only each tensor's own
            # owning core (and any resource its graph history adopted), so
            # this can never touch caller-owned state. The original
            # exception is preserved by the bare re-raise.
            for tensor in reversed(graph_temporaries):
                tensor.close()
            raise
        finally:
            for tensor in reversed(scratch):
                tensor.close()

    def _training_forward(self, input, gamma, beta, running_mean, running_var,
                          keep, temp):
        """Normalize with this batch's own statistics and advance both
        running buffers atomically.

        The ordering is load-bearing: the **complete** differentiable
        output graph is built first, then both graph-free replacement
        values, then the atomic commit, and only then is the already-built
        output returned. So a failure while building the output, while
        preparing the update values, or anywhere before the transaction's
        commit boundary leaves the running statistics exactly as they
        were — and once the update has committed, returning the output
        requires no further numerical operation that could fail."""
        # -- 1. the differentiable output graph.
        batch_mean = self._mean_over(input, keep)
        centered = keep(input.subtract(batch_mean))
        squared = keep(centered.multiply(centered))
        # The population variance: the mean of the squared deviations,
        # dividing by the batch size with no Bessel correction.
        batch_var = self._mean_over(squared, keep)
        inverse_std = keep(self._inverse_std(batch_var, input, keep))
        normalized = keep(centered.multiply(inverse_std))
        output = self._affine(normalized, gamma, beta, keep)

        # -- 2. the graph-free replacement values, from the *same* batch
        # statistics the normalization used — nothing is recomputed, and
        # detach() copies them out of the graph natively.
        new_mean = self._blend(running_mean, batch_mean, input, temp)
        new_var = self._blend(running_var, batch_var, input, temp)

        # -- 3. one atomic two-buffer commit through the F1 transaction.
        self._commit_running_state(
            running_mean, running_var, new_mean, new_var
        )
        return output

    def _eval_forward(self, input, gamma, beta, running_mean, running_var,
                      keep, temp):
        """Normalize with immutable snapshots of the stored running
        statistics, updating nothing.

        The registered buffers are read exactly once here, into
        independent owning graph-free copies; the graph that leaves this
        method holds **only** those copies, so no later mutation of the
        running statistics — a training step, a buffer-only state load, or
        a buffer-only checkpoint load — can change its gradient or
        invalidate its operands. (Replacing ``gamma``/``beta`` is a
        parameter mutation and still stales the graph under the existing
        version rule; see the module docstring.)"""
        mean_snapshot = keep(self._snapshot(running_mean))
        # The variance snapshot is consumed immediately: sqrt/reciprocal
        # each produce fresh independent owning storage, so the value the
        # graph keeps is the inverse standard deviation, not the variance.
        var_snapshot = temp(self._snapshot(running_var))
        inverse_std = keep(self._inverse_std(var_snapshot, input, temp))

        centered = keep(input.subtract(mean_snapshot))
        normalized = keep(centered.multiply(inverse_std))
        output = self._affine(normalized, gamma, beta, keep)

        # The two values a backward could read are handed to the graph's
        # history, so they live exactly as long as it does and are
        # released with it, exactly once. When no graph was built nothing
        # will ever read them, so they are released now.
        if not _adopt_graph_resources(output, (mean_snapshot, inverse_std)):
            mean_snapshot.close()
            inverse_std.close()
        return output

    # -- composed pieces --------------------------------------------------

    def _mean_over(self, value, track):
        """The mean of ``value`` over every reduction axis, as a sequence
        of existing single-axis ``NativeTensor.mean`` calls with
        ``keepdims=True``. Because each reduced dimension is retained at
        size 1, the axis numbers stay valid across the whole sequence — no
        tuple-axis reduction is added to ``NativeTensor``. Every reduction
        is differentiable, materializes nothing to NumPy, and never
        mutates ``value``."""
        for axis in self._REDUCTION_AXES:
            value = track(value.mean(axis=axis, keepdims=True))
        return value

    def _affine(self, normalized, gamma, beta, track):
        """Scale by ``gamma`` and shift by ``beta`` **per channel**,
        returning the fresh owning contiguous output.

        The rank-1 parameters are ``(C,)``, and NumPy-style broadcasting
        aligns from the *trailing* axis. For ``(N, C)`` the channel axis
        already is the trailing one, so ``normalized * gamma + beta`` is
        exactly right — the F3 path, kept unchanged.

        For NCHW it is exactly wrong: ``(N, C, H, W) * (C,)`` would line
        ``gamma`` up with **W**, silently scaling by spatial position (and
        only even running when ``W == C``). The fix deliberately keeps the
        parameters rank-1 rather than reshaping them to ``(1, C, 1, 1)``,
        because ``multiply`` records a stale-value guard entry **only for
        a direct operand carrying a value version** — a reshaped
        ``gamma`` would be an ordinary unversioned view, and mutating
        ``gamma`` after a forward would then silently change that graph's
        gradient instead of raising. That is precisely the §7 hazard, and
        it must not be reintroduced for parameters.

        So the *activation* moves instead of the parameter: a borrowing
        ``transpose`` carries the channel axis to the trailing position
        (NCHW → NHWC), ``gamma`` and ``beta`` apply as direct rank-1
        operands there, and a second borrowing ``transpose`` carries the
        result back before it is materialized. Both transposes are
        metadata-only and already differentiable, so this adds no
        gradient logic: ``multiply``'s existing broadcast-aware backward
        reduces ``gamma``'s gradient over N, H, and W, ``add``'s does the
        same for ``beta``, and ``transpose``'s backward applies the
        inverse permutation. Channels-last is an internal step, never a
        public layout mode."""
        if self._CHANNELS_LAST is None:
            scaled = track(normalized.multiply(gamma))
            return track(scaled.add(beta))
        # NCHW -> NHWC (metadata only; the result borrows ``normalized``).
        channels_last = track(normalized.transpose(self._CHANNELS_LAST))
        # gamma and beta stay *direct* rank-1 operands, so both keep the
        # existing direct-parameter version guard.
        scaled = track(channels_last.multiply(gamma))
        shifted = track(scaled.add(beta))
        # NHWC -> NCHW, then materialize: a transpose result is a
        # borrowing strided view, and this layer's contract is a fresh
        # owning contiguous output.
        channels_first = track(shifted.transpose(self._channels_first))
        return track(channels_first.contiguous_copy())

    def _inverse_std(self, variance, like, track):
        """``reciprocal(sqrt(variance + eps))`` — epsilon **inside** the
        root, never ``sqrt(var) + eps``, from a native graph-free scalar
        (no NumPy). ``track`` records the three intermediates; the
        returned inverse standard deviation is deliberately untracked so
        the caller decides its lifetime (the training graph keeps it as an
        ordinary temporary; the evaluation graph adopts it)."""
        eps_tensor = track(NativeTensor.full(
            (), self.eps, dtype=like.dtype, device=like.device,
            requires_grad=False,
        ))
        var_plus_eps = track(variance.add(eps_tensor))
        std = track(var_plus_eps.sqrt())
        return std.reciprocal()

    def _snapshot(self, buffer):
        """An independent, owning, contiguous, graph-free, gradient-free
        native copy of a registered running buffer, already shaped for
        broadcasting.

        The rank-1 buffer is reshaped to the broadcast layout through a
        *borrowing* metadata view, which is then materialized by the
        native storage-to-storage copy path — so the value the graph ends
        up holding owns its own storage and depends on neither the
        buffer's lifetime nor its future value. No NumPy, no host
        materialization, and no borrowing view escapes."""
        view = buffer.reshape(self._stat_shape)
        try:
            return view.contiguous_copy()
        finally:
            # A borrowing view owns nothing; closing it releases only the
            # wrapper, never the buffer's storage.
            view.close()

    def _blend(self, current, statistic, like, track):
        """``(1 - momentum) * current + momentum * statistic`` as an
        independent **graph-free** owning ``(num_features,)`` value.

        ``statistic`` is the live differentiable batch statistic, so it
        enters through ``detach()`` — a native contiguous copy with no
        graph history and no NumPy round trip. Every operand from there on
        has ``requires_grad=False``, so no autograd node is built and no
        gradient path can reach the result. Reading ``current`` (the
        registered buffer) is safe for exactly that reason: this
        arithmetic builds no graph, so the buffer is never captured as a
        rereadable operand.

        At the boundaries the convention is exact: ``momentum=0.0`` gives
        ``1.0 * current + 0.0 * statistic``, numerically the previous
        value; ``momentum=1.0`` gives ``0.0 * current + 1.0 * statistic``,
        exactly the current population batch statistic."""
        detached = track(statistic.detach())
        # The statistics carry the broadcast layout ((1, C) here); the
        # buffers are rank-1. A borrowing reshape lines them up without
        # copying — it is only an operand, never the stored result.
        flat = track(detached.reshape((self.num_features,)))
        keep_old = track(NativeTensor.full(
            (), 1.0 - self.momentum, dtype=like.dtype, device=like.device,
            requires_grad=False,
        ))
        take_new = track(NativeTensor.full(
            (), self.momentum, dtype=like.dtype, device=like.device,
            requires_grad=False,
        ))
        old_part = track(current.multiply(keep_old))
        new_part = track(flat.multiply(take_new))
        return track(old_part.add(new_part))

    def _commit_running_state(self, running_mean, running_var,
                              new_mean, new_var):
        """Advance both running buffers as **one** atomic transaction,
        through the private F1 primitive.

        Two entries, one call: staging produces both replacement cores
        before anything mutates, the commit swaps both while preserving
        each buffer's Python identity, the replaced cores are closed
        exactly once afterwards, and any failure before the commit
        boundary restores both buffers and closes every staged core. No
        state dictionary is fabricated, ``load_state_dict`` is not called
        internally, no attribute is reassigned, and no second transaction
        system exists. Buffers carry no value version, so this moves no
        version at all — ``gamma`` and ``beta`` are untouched and no
        existing graph becomes stale."""
        _native_state.replace_native_state((
            _native_state.NativeStateEntry(
                label="running_mean",
                destination=running_mean,
                # The transaction takes ownership of whatever a factory
                # returns, so it gets an independent copy: the prepared
                # value is this call's scratch and is released with it.
                make_core=lambda: _native_copy(new_mean._require_open()),
                source=new_mean,
            ),
            _native_state.NativeStateEntry(
                label="running_var",
                destination=running_var,
                make_core=lambda: _native_copy(new_var._require_open()),
                source=new_var,
            ),
        ))

    # -- validation -------------------------------------------------------

    def _validate_forward(self, input):
        """Check every live tensor this forward will touch — the input,
        both affine parameters, and both running buffers — **before** any
        graph node is built, and return the four state tensors. Mutates
        nothing on any path."""
        if not isinstance(input, NativeTensor):
            raise TypeError(
                f"{type(self).__name__}.forward requires a NativeTensor "
                f"input, got {type(input).__name__}"
            )
        if input.closed:
            raise RuntimeError(
                f"{type(self).__name__}.forward: the input tensor has been "
                f"closed"
            )
        # ``or`` short-circuits, so the channel lookup only runs once the
        # rank is known correct.
        if (input.ndim != self._INPUT_NDIM
                or input.shape[1] != self.num_features):
            raise ValueError(
                f"{type(self).__name__}({self.num_features}) expects input "
                f"shaped {self._LAYOUT} with C={self.num_features}, got "
                f"shape {input.shape}"
            )

        expected = (self.num_features,)
        gamma = self.gamma
        beta = self.beta
        for name, parameter in (("gamma", gamma), ("beta", beta)):
            if parameter.closed:
                raise RuntimeError(
                    f"{type(self).__name__}.forward: {name} has been closed"
                )
            if parameter.shape != expected:
                raise ValueError(
                    f"{type(self).__name__}({self.num_features}) affine "
                    f"parameter {name!r} must have shape {expected}, got "
                    f"{parameter.shape}"
                )
        running_mean = self._registered_running("running_mean")
        running_var = self._registered_running("running_var")

        for name, tensor in (("gamma", gamma), ("beta", beta),
                             ("running_mean", running_mean),
                             ("running_var", running_var)):
            if tensor.dtype != input.dtype or tensor.device != input.device:
                raise ValueError(
                    f"{type(self).__name__} expects input dtype/device "
                    f"{tensor.dtype}/{tensor.device} to match {name}, got "
                    f"{input.dtype}/{input.device}"
                )
        return gamma, beta, running_mean, running_var

    def _registered_running(self, name):
        """The named running buffer, proved to still be a live, owning,
        contiguous, gradient-free **registered persistent buffer** of
        shape ``(num_features,)``. Everything the running update and the
        evaluation snapshot assume is checked here, before either runs."""
        entry = self._buffers.get(name)
        if entry is None or not entry.persistent:
            raise RuntimeError(
                f"{type(self).__name__}.forward: {name!r} is no longer a "
                f"registered persistent buffer"
            )
        tensor = entry.tensor
        if not isinstance(tensor, NativeTensor) or isinstance(
            tensor, NativeParameter
        ):
            raise RuntimeError(
                f"{type(self).__name__}.forward: {name!r} must be a plain "
                f"NativeTensor buffer, got {type(tensor).__name__}"
            )
        if tensor.closed:
            raise RuntimeError(
                f"{type(self).__name__}.forward: {name} has been closed"
            )
        expected = (self.num_features,)
        if tensor.shape != expected:
            raise ValueError(
                f"{type(self).__name__}({self.num_features}) running "
                f"statistic {name!r} must have shape {expected}, got "
                f"{tensor.shape}"
            )
        if not tensor.owns_core or not tensor.contiguous:
            raise RuntimeError(
                f"{type(self).__name__}.forward: {name!r} must be an owning "
                f"contiguous buffer"
            )
        if tensor.requires_grad:
            raise RuntimeError(
                f"{type(self).__name__}.forward: {name!r} must not require "
                f"grad (running statistics are never trainable)"
            )
        return tensor

    def __repr__(self):
        return (
            f"{type(self).__name__}(num_features={self.num_features}, "
            f"eps={self.eps}, momentum={self.momentum})"
        )


class NativeBatchNorm1d(_NativeBatchNorm):
    """Native batch normalization over ``(N, C)`` activations.

    ``NativeBatchNorm1d(num_features, eps=1e-5, momentum=0.1)`` — the
    first stateful native numerical module. Training normalizes with this
    batch's own differentiable population statistics and advances the
    persistent ``running_mean``/``running_var`` buffers atomically;
    evaluation normalizes with immutable graph-free snapshots of those
    buffers and updates nothing. ``gamma`` and ``beta`` always exist. See
    the module docstring for the full contract."""

    _INPUT_NDIM = 2
    # Reduce over the batch: each feature gets one mean and one variance
    # over the N values in its column.
    _REDUCTION_AXES = (0,)
    _TRAILING_DIMS = 0
    _LAYOUT = "(N, C)"
    # The channel axis is already trailing, so the rank-1 affine
    # parameters broadcast directly and no transposition is needed.
    _CHANNELS_LAST = None


class NativeBatchNorm2d(_NativeBatchNorm):
    """Native batch normalization over NCHW ``(N, C, H, W)`` activations.

    ``NativeBatchNorm2d(num_features, eps=1e-5, momentum=0.1)`` — the same
    stateful module as ``NativeBatchNorm1d`` over the Phase-D activation
    layout, and **the same implementation**: this class supplies nothing
    but the shape configuration below. Each channel gets one population
    mean and one population variance over all ``N * H * W`` of its values,
    the persistent ``running_mean``/``running_var`` buffers stay ``(C,)``,
    and ``gamma``/``beta`` scale and shift **per channel** at every
    spatial position. See the module docstring for the full contract."""

    _INPUT_NDIM = 4
    # Reduce over the batch and both spatial axes, never over channels:
    # each channel gets one mean and one variance over N * H * W values.
    # Applied as sequential single-axis means with keepdims=True, so
    # (N, C, H, W) -> (1, C, H, W) -> (1, C, 1, W) -> (1, C, 1, 1) and the
    # axis numbers stay valid throughout.
    _REDUCTION_AXES = (0, 2, 3)
    # ...giving the (1, C, 1, 1) per-channel broadcast layout.
    _TRAILING_DIMS = 2
    _LAYOUT = "(N, C, H, W)"
    # NCHW -> NHWC for the affine application only, so the rank-1 gamma
    # and beta line up with the *channel* axis instead of W — see
    # ``_NativeBatchNorm._affine`` for why the activation moves rather
    # than the parameters. The return leg is derived from this.
    _CHANNELS_LAST = (0, 2, 3, 1)
