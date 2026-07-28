"""Private, reusable atomic replacement of registered native state
(Phase F, milestone F1 — see docs/native_normalization_design.md §8).

**This module is private.** It is deliberately absent from
``tensorforge.experimental.__all__`` and must stay that way: it is not a
public in-place mutation API for ``NativeTensor``, and nothing outside
``tensorforge.experimental`` may depend on it. The one public controlled
mutation primitive in the native line remains
``NativeParameter.copy_value_``.

What it does
------------

One narrow thing: replace the ``_core`` of one or more **already
registered** native state objects — ``NativeParameter`` leaves and
persistent buffers (plain owning ``NativeTensor``\\ s) — as a single
all-or-nothing transaction that preserves every destination's Python
identity.

This behavior already existed, inline, inside
``NativeModule.load_state_dict``. F1 extracts it verbatim in semantics so
that a second caller can reuse it without inventing a parallel
implementation of the subtle parts (staging, the commit boundary,
rollback, exactly-once closing, and parameter-version movement), which is
where a state-corruption bug would most likely be introduced.

Intended callers
----------------

1. ``NativeModule.load_state_dict`` — parameters and persistent buffers
   in one transaction (live as of F1).
2. **Future (F3/F4):** ``NativeBatchNorm1d`` / ``NativeBatchNorm2d``
   committing ``running_mean`` **and** ``running_var`` as one atomic
   running-statistics update. That caller builds two entries directly and
   calls :func:`replace_native_state` — it does **not** construct a fake
   state dictionary and does **not** go through the public
   ``load_state_dict``. No normalization code exists or calls this yet;
   F1 only makes the primitive available and proves its semantics.

The transaction model
---------------------

Three phases, with one explicit, documented commit boundary:

**Plan (no mutation).** Every destination and entry is validated, and
entries are deduplicated by destination *object identity*. Each
destination's current core is recorded. Any failure here leaves
everything untouched and nothing has been allocated.

**Stage (no mutation).** Every replacement core is produced by its
entry's factory and validated. The transaction **takes ownership of every
core a factory returns**: it will either install it or close it, exactly
once, on every path. A staging failure closes every core staged so far,
exactly once, and leaves every destination untouched.

All four phases run under the shared **state-transaction guard**
(``_native_state_lock``, Phase G milestone G5), which is item 1 of the
universal state-replacement lock order. Individually atomic is not the
same as serializable: two concurrent transactions could each be
all-or-nothing and still interleave into a state assembled from both. The
guard gives every participating replacement a valid serial order. It is
reentrant, so the whole-checkpoint transaction holds it and then calls
``NativeModule.load_state_dict``, which arrives here and re-enters it.
This path takes no generator lock, so it cannot invert the order.

**Commit (reversible until the boundary).** Each destination's core is
swapped, then every affected parameter's value version is incremented.
Both steps live inside one rollback guard, so a failure at *either*
restores every already-swapped core and every already-moved version and
closes every staged core.

    THE COMMIT BOUNDARY is the point at which every swap **and** every
    version increment has succeeded. Before it the transaction is fully
    reversible. After it, it is irreversible: the replaced cores are
    released, and there is no way back (a released core cannot be
    un-closed, and callers may already observe the new values).

**Finalize (irreversible).** Each replaced original core is closed
exactly once. Every replaced core is closed even if one close raises, so
ownership is never left ambiguous and nothing leaks; the first such
failure is then re-raised, wrapped, making clear that the state change
itself succeeded. In practice ``NativeTensorCore.close()`` is idempotent
and does not raise, so this path is defensive only.

Failure-injection seams
-----------------------

The five module-level ``_stage_entry`` / ``_install_core`` /
``_restore_core`` / ``_bump_version`` / ``_release_core`` functions exist
so tests can simulate a failure at each step by monkeypatching this
module. They are private seams, not production flags: nothing in the
library ever changes them, and there is no user-facing failure control.
"""

from collections import namedtuple

from tensorforge.backends import cpp

from ._native_state_lock import state_transaction
from .native_parameter import NativeParameter
from .native_tensor import NativeTensor


# One requested replacement.
#
# ``label``       a short caller-supplied name used verbatim in error
#                 messages (``load_state_dict`` passes the canonical
#                 state-dict key, so its errors keep naming the key).
# ``destination`` the registered NativeParameter or persistent-buffer
#                 NativeTensor whose value is being replaced. The object
#                 itself is preserved; only its core changes.
# ``make_core``   a zero-argument callable returning the replacement
#                 NativeTensorCore. Called once, during staging, before
#                 any destination is mutated. The transaction takes
#                 ownership of whatever it returns.
# ``source``      the object the replacement value came from, used only
#                 for duplicate-entry reconciliation (see
#                 ``_plan_entries``). Two entries for the same
#                 destination are compatible **only** when they name the
#                 same source object; anything else is a conflict and is
#                 rejected before any mutation. ``None`` means "no source
#                 identity", which never deduplicates.
NativeStateEntry = namedtuple(
    "NativeStateEntry", ("label", "destination", "make_core", "source")
)
NativeStateEntry.__new__.__defaults__ = (None,)


# One planned replacement, after validation and deduplication. Carries
# everything rollback needs, captured before anything is staged.
_Planned = namedtuple(
    "_Planned",
    ("label", "destination", "make_core", "source",
     "original_core", "is_parameter", "original_version"),
)


# ---------------------------------------------------------------------------
# Private seams. Each is one indivisible step of the transaction, factored
# out so a test can monkeypatch exactly one of them. Keep them tiny: the
# transaction's correctness argument assumes each either fully happens or
# raises without a partial effect.
# ---------------------------------------------------------------------------

def _stage_entry(planned):
    """Produce the replacement core for one planned entry."""
    return planned.make_core()


def _install_core(planned, new_core):
    """Swap one destination's core, returning the core it replaced.

    A ``NativeParameter`` goes through its validated ``_adopt_value_core``
    (which re-checks shape/dtype/device defensively); a buffer is a plain
    owning ``NativeTensor`` whose ``_core`` is assigned directly. Either
    way the destination object, its registrations, its ``requires_grad``
    and gradient, and its ownership are untouched — only the value moves.
    """
    destination = planned.destination
    if planned.is_parameter:
        return destination._adopt_value_core(new_core)
    old_core = destination._require_open()
    destination._core = new_core
    return old_core


def _restore_core(planned, original_core):
    """Put one destination's original core back during rollback.

    A plain assignment on purpose: ``_adopt_value_core`` would re-validate
    and could itself raise, which is precisely what a rollback path must
    never do. The core being restored is the one this destination owned a
    moment ago, so it is known good.
    """
    planned.destination._core = original_core


def _bump_version(planned):
    """Increment one parameter destination's monotonic value version."""
    planned.destination._version += 1


def _restore_version(planned):
    """Put one parameter destination's value version back during
    rollback, so a failed transaction moves no version at all."""
    planned.destination._version = planned.original_version


def _release_core(core):
    """Release one replaced original core, after the commit boundary."""
    core.close()


# ---------------------------------------------------------------------------
# Validation and planning
# ---------------------------------------------------------------------------

def _validate_destination(entry):
    """Check one destination is a legal replacement target, and classify
    it. Returns ``(is_parameter, original_core)``."""
    label = entry.label
    destination = entry.destination
    if not isinstance(destination, NativeTensor):
        raise TypeError(
            f"cannot replace the value of {label!r}: the destination must "
            f"be a NativeTensor or NativeParameter, got "
            f"{type(destination).__name__}"
        )
    if destination.closed:
        raise RuntimeError(f"cannot load into {label!r}: it has been closed")
    if not destination.owns_core:
        raise ValueError(
            f"cannot replace the value of {label!r}: the destination does "
            f"not own its core (a borrowing view is never registered state)"
        )
    if not destination._is_leaf:
        raise ValueError(
            f"cannot replace the value of {label!r}: the destination is a "
            f"graph node, not registered leaf state"
        )
    is_parameter = isinstance(destination, NativeParameter)
    if not is_parameter and destination.requires_grad:
        raise ValueError(
            f"cannot replace the value of {label!r}: a non-parameter "
            f"destination must not require grad"
        )
    return is_parameter, destination._require_open()


def _plan_entries(entries):
    """Validate every entry and reduce it to a deduplicated plan.

    Deduplication is by destination **object identity**, not by label:
    a shared parameter or buffer reachable under several registered names
    is one destination and must be replaced (and version-bumped, and
    released) exactly once. Two entries for one destination are treated as
    the same request only when they name the same ``source`` object; any
    other duplicate is a genuine conflict — two different values for one
    object — and is rejected here, before anything is staged or mutated.
    """
    planned = []
    by_destination = {}
    for entry in entries:
        if not isinstance(entry, NativeStateEntry):
            raise TypeError(
                f"native state entries must be NativeStateEntry, got "
                f"{type(entry).__name__}"
            )
        if not callable(entry.make_core):
            raise TypeError(
                f"the replacement factory for {entry.label!r} must be "
                f"callable, got {type(entry.make_core).__name__}"
            )
        is_parameter, original_core = _validate_destination(entry)
        key = id(entry.destination)
        if key in by_destination:
            first = by_destination[key]
            if entry.source is None or first.source is not entry.source:
                raise ValueError(
                    f"conflicting replacement values for one destination: "
                    f"{first.label!r} and {entry.label!r} name the same "
                    f"registered object but supply different values"
                )
            # Same object, same source: one request stated twice.
            continue
        record = _Planned(
            label=entry.label,
            destination=entry.destination,
            make_core=entry.make_core,
            source=entry.source,
            original_core=original_core,
            is_parameter=is_parameter,
            original_version=(
                entry.destination._version if is_parameter else None
            ),
        )
        by_destination[key] = record
        planned.append(record)
    return planned


def _reject_aliasing_core(planned, new_core, live_storage_ids,
                          staged_storage_ids):
    """Reject a replacement core that aliases something already alive:
    any destination's current storage, or a core already staged in this
    transaction.

    Kept separate from :func:`_validate_staged_core` because the caller
    must treat the two failures differently. A core rejected *here* is
    **not** the transaction's to release — it is either live registered
    state or a core the transaction already owns through an earlier
    staged entry — so closing it would corrupt live state or double-free.
    """
    label = planned.label
    storage_id = id(new_core.storage)
    if storage_id == id(planned.original_core.storage):
        raise ValueError(
            f"the replacement for {label!r} shares storage with the value "
            f"it would replace; replacements must be independent"
        )
    if storage_id in live_storage_ids:
        raise ValueError(
            f"the replacement for {label!r} shares storage with another "
            f"destination's live value; replacements must be independent"
        )
    if storage_id in staged_storage_ids:
        raise ValueError(
            f"the replacement for {label!r} shares storage with another "
            f"destination's replacement; each destination must receive an "
            f"independent core"
        )


def _validate_staged_core(planned, new_core):
    """Check one freshly staged replacement core before it can be
    installed: open, owning, contiguous, and metadata-matched.

    A core rejected here **is** the transaction's to release: the factory
    produced it for this transaction and nothing else refers to it.
    """
    label = planned.label
    if new_core._closed:
        raise RuntimeError(
            f"the replacement core for {label!r} has been closed"
        )
    if not new_core._owns_storage:
        raise ValueError(
            f"the replacement core for {label!r} must own its storage "
            f"(a borrowing view cannot become registered state)"
        )
    if not new_core.contiguous:
        raise ValueError(
            f"the replacement core for {label!r} must be contiguous"
        )
    original = planned.original_core
    if (new_core.shape != original.shape
            or new_core.dtype != original.dtype
            or new_core.device != original.device):
        raise ValueError(
            f"metadata mismatch for {label!r}: the destination is "
            f"{original.shape}/{original.dtype}/{original.device}, the "
            f"replacement is "
            f"{new_core.shape}/{new_core.dtype}/{new_core.device}"
        )


# ---------------------------------------------------------------------------
# The transaction
# ---------------------------------------------------------------------------

def replace_native_state(entries):
    """Atomically replace the values of the given registered native
    state objects, preserving every destination's Python identity.

    ``entries`` is an iterable of :class:`NativeStateEntry`. Returns the
    number of unique destinations replaced.

    On success: every destination owns a fresh independent core holding
    the new value; every affected parameter's version has advanced by
    exactly one (once per unique parameter, however many names alias it);
    every buffer's version state is untouched (buffers have none); and
    every replaced core has been closed exactly once.

    On any failure before the commit boundary: nothing changed anywhere —
    no core, no version — and every staged core has been closed exactly
    once. The original exception propagates unchanged.

    The whole transaction — plan, stage, commit, and release — runs under
    the shared state-transaction guard (``_native_state_lock``), which is
    item 1 of the universal state-replacement lock order. That is what
    makes two concurrent replacements *serializable* rather than merely
    individually atomic: planning captures each destination's current core
    and the commit re-checks it, so without the guard two overlapping
    transactions could each be all-or-nothing and still leave a state
    assembled from both. This path takes **no** generator lock, so it can
    never invert the order; a checkpoint transaction that already holds
    the guard re-enters it here through the ``RLock``.

    See the module docstring for the commit boundary and the ownership
    rules for factory-produced cores.
    """
    with state_transaction():
        return _replace_native_state_locked(entries)


def _replace_native_state_locked(entries):
    """The transaction body, with the shared guard already held."""
    planned = _plan_entries(entries)
    if not planned:
        return 0

    # --- Stage. Nothing is mutated yet, so a failure here only has to
    # release what it already created.
    staged = []
    staged_storage_ids = set()
    live_storage_ids = {id(record.original_core.storage) for record in planned}
    try:
        for record in planned:
            new_core = _stage_entry(record)
            if not isinstance(new_core, cpp.NativeTensorCore):
                # Nothing was produced that this transaction could own,
                # so there is nothing to release for this entry.
                raise TypeError(
                    f"the replacement for {record.label!r} must be a "
                    f"NativeTensorCore, got {type(new_core).__name__}"
                )
            # Aliasing is checked first and *never* closes the offending
            # core: it belongs to live state or to an earlier staged
            # entry, so releasing it here would corrupt or double-free.
            _reject_aliasing_core(
                record, new_core, live_storage_ids, staged_storage_ids
            )
            try:
                _validate_staged_core(record, new_core)
            except BaseException:
                # This core is genuinely ours and never joined ``staged``,
                # so close it here or it is the one leak the handler
                # below cannot see.
                new_core.close()
                raise
            staged.append((record, new_core))
            staged_storage_ids.add(id(new_core.storage))
    except BaseException:
        for _, new_core in staged:
            new_core.close()
        raise

    # --- Commit. Swaps and version increments live under one rollback
    # guard, so the transaction stays fully reversible until *both* have
    # completed for every destination. BaseException is deliberate: a
    # KeyboardInterrupt between two swaps must roll back too.
    installed = []
    bumped = []
    try:
        for record, new_core in staged:
            # Defensive re-check: nothing may have replaced this
            # destination's core between planning and now (a factory with
            # a side effect, a signal handler, a reentrant caller).
            if record.destination._core is not record.original_core:
                raise RuntimeError(
                    f"the value of {record.label!r} changed while its "
                    f"replacement was being prepared; the transaction was "
                    f"abandoned before any change was made"
                )
            old_core = _install_core(record, new_core)
            installed.append((record, old_core))
        # Versions move only once every swap has succeeded, and still
        # inside the guard, so a failed transaction moves no version.
        for record, _ in installed:
            if record.is_parameter:
                _bump_version(record)
                bumped.append(record)
    except BaseException:
        for record in reversed(bumped):
            _restore_version(record)
        for record, old_core in reversed(installed):
            _restore_core(record, old_core)
        for _, new_core in staged:
            new_core.close()
        raise

    # --- COMMIT BOUNDARY CROSSED. Everything below is irreversible.
    #
    # Release each replaced core exactly once. Every one is attempted even
    # if an earlier close raises, so a failure can never leave a replaced
    # core in limbo; the first failure is then reported, wrapped, so it is
    # unmistakable that the state change itself succeeded.
    released = set()
    cleanup_error = None
    for _, old_core in installed:
        if id(old_core) in released:
            continue
        released.add(id(old_core))
        try:
            _release_core(old_core)
        except BaseException as error:      # pragma: no cover - defensive
            if cleanup_error is None:
                cleanup_error = error
    if cleanup_error is not None:           # pragma: no cover - defensive
        raise RuntimeError(
            "the native state replacement committed successfully, but "
            "releasing the replaced storage failed; the new values are "
            "installed and every version has moved"
        ) from cleanup_error
    return len(installed)
