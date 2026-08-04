"""NativeDataLoader — the native mini-batch iteration surface (Phase J,
milestones J3 and J4; see docs/native_data_pipeline_design.md §3.5, §3.6,
§3.7, §9, §10, §11.3, §11.4, §11.5, §12.5, §15, §16, §17.3, §17.5).

The third piece of Phase-J runtime, and the one that finally *delivers* a
batch. It turns each index group the sampler plans into a
``(NativeTensor, numpy.ndarray)`` pair and hands ownership of both to the
caller — one owning contiguous feature batch at the dataset's dtype, and
one fresh read-only host ``int64`` target array.

The one invariant everything here exists to provide
------------------------------------------------------

**The committed sampler position advances if and only if a batch was
successfully delivered to the caller.** A batch that is not delivered
consumes no logical position, at every failure point, without exception.

A naive "materialize, advance the cursor, return" sequence cannot provide
that: an exception between the advance and the return would leave a
position consumed and a batch unreachable, silently breaking both the
exact-resume proof and the promise that a saved state describes the
*next* batch. So every ``__next__`` is an explicit transaction, adapted
from ``NativeGenerator``'s reservation, with a full rollback where the
generator has a cancellation:

1. **Claim** — validate lifecycle, snapshot the committed position,
   compute the candidate indices and the candidate post-delivery position
   through the sampler's pure planner, mint a **never-reused serial**, and
   publish *only* the claim. Nothing has moved.
2. **Construct** — build the feature ``NativeTensor`` and then the target
   array. Both are **iterator-owned** from this moment. Nothing has moved.
3. **Publish** — turn the claim into a pending-delivery record. Nothing
   has moved, and every reentrant state operation is now refused.
4. **Commit and deliver** — apply the candidate position through the
   sampler's non-failing write seam, then pass the record through the
   private ``_deliver_batch`` seam. Only if it returns, and only if the
   exact same transaction is still live and still owns the batch, does the
   handoff complete.
5. **Rollback** — anything that raises after the candidate was applied
   restores the exact pre-delivery position first (through the same
   non-failing seam, so it cannot itself fail), clears the record on both
   owners, closes the undelivered tensor, and releases the target
   reference. It runs from an unconditional ``finally``, and it is
   exact-match, so it is idempotent against a racing ``close()``.

**Commit-then-deliver, not deliver-then-commit**, deliberately: if the
position advanced *after* the seam returned, a failure in between would
hand the caller a batch the loader still considered unconsumed, and the
next call would deliver it a second time. Committing first makes the two
error directions asymmetric in the safe direction — the only recoverable
state is "not yet delivered", and it is fully recoverable.

Who owns what
-------------

The transaction record is **split by ownership**. The sampler holds the
integer half (serial, owning token, the two positions, the indices),
because that is what its own state guards must see. The iterator holds
the resource half (the feature tensor, the target array), because an
owned resource belongs on the object whose ``close()`` releases it — and
because a sampler that transiently owned native storage would contradict
its own "owns nothing releasable, so it has no ``close()``" contract.

**After a successful delivery neither the loader nor the iterator retains
any reference to the batch.** No close path can reach a delivered batch,
and none tries: keeping delivered batches alive so they could be closed
would hold a whole epoch's native storage, which is a leak in the shape
of a convenience. **The caller closes every feature batch**; the target
array is ordinary host memory and needs none.

Loader state (J4)
-----------------

``state_dict()`` returns a compact **tagged wrapper** around the sampler's
own state — exactly three keys, ``format`` / ``format_version`` /
``sampler`` — and ``load_state_dict()`` restores one transactionally. The
loader owns no epoch, cursor, seed, shuffle, batch size, or drop-last
field of its own, so duplicating any of them at the wrapper's root would
create a second authority on a fact the sampler already owns.

**Why a wrapper rather than the sampler state itself:** the loader is
what a caller checkpoints, and the two objects' states must be
distinguishable in metadata. Without its own ``format`` tag a loader
state and a sampler state would be the same JSON, so handing one where
the other was meant would be accepted silently. The wrapper is also where
a future loader-owned field would go without disturbing the sampler's
schema — and no such field is added now.

The load is transactional in the exact sense the rest of this module is:
three lifecycle guards run **before** ``state`` is read at all (closed,
then transaction, then active iteration), the wrapper is validated
completely, the **whole** of the nested sampler validation is delegated
to the sampler's one validation-only seam rather than restated here, and
only then does the commit run — through the same non-failing write seam a
delivery commit and a rollback use. Nothing mutates until every check has
passed and the only remaining step cannot fail, which is what makes a
rejected load leave the world byte-identical without a rollback path of
its own.

``state_dict()`` is allowed at every moment except one: it is **refused**
while a §9.4 transaction is in flight, through the sampler's own guard
rather than a second inconsistent authority, because inside the
commit-before-delivery window there is no honest answer. It is allowed
after ``close()`` — recording where a loader stopped is not a resource
operation.

What this milestone deliberately does not add
---------------------------------------------

No checkpoint integration in either direction: this module imports no
checkpoint code, no checkpoint code imports it, nothing discovers a
loader, and no global registry of loaders exists. Carrying the state
through the existing validated version-3 ``metadata`` channel is the
**caller's** step, and J5 proves that workflow end to end. No checkpoint
root field, no checkpoint version 4, no ``save``/``restore``/``state``/
``load_state`` alias, no public validator, and no public state class.
No ``__len__``, because mid-epoch it would have to mean either "batches
per epoch" or "batches remaining" and a caller reading the wrong one
would silently mis-schedule a resumed run; ``loader.sampler
.batches_per_epoch`` and ``loader.sampler.remaining`` each say which is
meant. No ``__next__`` on the loader, because the loader is not an
iterator. No batch-size, shuffle, seed, or drop-last argument, because
the sampler owns those and one fact needs one owner. No worker, thread,
lock, queue, future, prefetch, collate, transform, or callback surface,
and no public delivery hook.

**Not thread-safe, and no lock exists here.** One thread at a time per
dataset, sampler, and loader; concurrent use requires external locking.
The transaction's claim guards **reentrancy** — a finalizer, a callback,
or a signal handler arriving on the calling thread is refused
deterministically — and that is a correctness mechanism, not a
concurrency one. Two genuinely concurrent threads can both read "no
claim" before either writes one. What survives even a raced transaction
is the exact-match rule: no cleanup can roll back, close, or complete a
record that is not its own.

Adds no kernel, C ABI symbol, ctypes declaration, checkpoint field or
version, optimizer-state version, capability registry value, dtype,
device, or dependency. It has no ``dtype`` and no ``device`` argument:
the dataset owns the one and there is no such thing as the other.
"""

from .native_sampler import (
    NativeBatchSampler,
    # The two schema-shaped rules, **shared rather than restated** — the
    # same stance the sampler itself takes towards ``_validate_uint64``.
    # A duplicated exact-``int`` rule or key-set rule would be a second
    # spelling of one convention, free to drift; there is exactly one of
    # each in the pipeline, and the loader's wrapper is held to it.
    _require_exact_int,
    _require_exact_keys,
)

# The loader state schema's identity (§11.3). The format tag is the whole
# reason the wrapper exists: without it a loader state and a sampler state
# would be the same JSON, and handing one where the other was meant would
# be accepted silently. Private, unexported, and not a public registry —
# there is no version 2, no alias tag, no migration path, and no
# placeholder for either.
_FORMAT = "tensorforge.native_data_loader"
_FORMAT_VERSION = 1
_SUPPORTED_FORMAT_VERSIONS = (1,)

# Exactly three root keys, validated by exact **set** equality in both
# directions, so a missing key is not a default and an unknown key is not
# ignored. The sampler's six configuration and position fields live in the
# nested object and are deliberately **not** duplicated here: the loader
# owns no epoch, cursor, seed, shuffle, batch size, or drop-last of its
# own, and a second copy could disagree with the first.
_STATE_FIELDS = ("format", "format_version", "sampler")


def _deliver_batch(record):
    """The publish → deliver seam of the batch transaction (§9.4, Phase 4).

    It returns the record's ``(features, targets)`` pair and does nothing
    else. It exists for exactly one reason: **so that the
    publish-to-delivery failure position is addressable and can be tested
    deliberately**, by monkeypatching this module attribute to raise
    there. The direct analogue of ``native_generator._deliver_reservation``.

    Private and module-level: never exported, never named by any public
    API, never added to ``experimental.__all__``. It is a **test seam, not
    a hook** — it accepts no user-supplied callable, exposes no public
    callback, and is reachable only by patching a private attribute, so no
    arbitrary code runs inside the transaction.
    """
    return record._features, record._targets


class _NativeBatchIterator:
    """One epoch's traversal, and the **resource half** of one in-flight
    batch handoff.

    Private, never exported, and never constructed by a caller: an
    iterator arrives only from ``iter(loader)``, which is the same stance
    ``NativeGenerator``'s reservation token takes. Its *behavior* is
    public — ``__iter__``, ``__next__``, ``close()``, and the context
    manager — but its name and its constructor are not.

    It captures ``sampler.remaining`` once, at construction, and counts it
    down. The countdown is **captured rather than re-read** because the
    sampler's ``remaining`` resets to a whole epoch the moment the
    canonical boundary is crossed, so an iterator that re-read it would
    never terminate. That is also why a state load is refused while any
    iteration is active: it would leave a captured countdown describing a
    position that no longer exists.

    The countdown decreases **only** on a successful delivery. A failed
    construction, publication, commit, delivery, or rollback leaves it
    exactly where it was, so a retry re-plans the identical batch.
    """

    __slots__ = ("_loader", "_sampler", "_token", "_to_yield",
                 "_closed", "_superseded", "_exhausted",
                 "_txn_serial", "_features", "_targets")

    def __init__(self, loader):
        # Every slot is written before anything that could fail, so a
        # partially built iterator is never observable and ``__del__``
        # never meets an unset attribute.
        self._loader = loader
        self._sampler = loader.sampler
        self._closed = False
        self._superseded = False
        self._exhausted = False
        self._txn_serial = 0
        self._features = None
        self._targets = None
        self._to_yield = 0
        self._token = self._sampler._begin_iteration()
        try:
            # One iterator is one epoch: the batches remaining in the
            # sampler's **current** epoch. From a fresh position that is
            # the whole epoch; from a restored mid-epoch position it is
            # exactly the tail, which is what an exact resume needs.
            self._to_yield = self._sampler.remaining
        except BaseException:
            # §17.3's last row: a failed iterator creation releases the
            # participation it had already taken.
            self._sampler._end_iteration(self._token)
            self._closed = True
            raise

    # -- iteration ------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self):
        """Run the five-phase transaction and return
        ``(NativeTensor, numpy.ndarray)``.

        ``StopIteration`` when the captured countdown is spent.
        ``RuntimeError`` when this iterator is closed, was superseded, its
        loader is closed, its dataset is closed, or a transaction is
        already in flight — four lifecycle faults and one reentrancy
        fault, each distinct from exhaustion, because a ``for`` loop must
        never silently swallow one.
        """
        sampler = self._sampler
        # --- Phase 1, step 1: iterator and loader lifecycle. A closed or
        # superseded traversal is a lifecycle error, never an exhausted
        # one, so none of these is StopIteration.
        if self._closed:
            raise RuntimeError(
                "this batch iterator is closed; call iter(loader) again to "
                "continue from the committed sampler position"
            )
        if self._superseded:
            raise RuntimeError(
                "this batch iterator was superseded by a later iter() over "
                "the same loader; only the most recent iterator may deliver "
                "a batch, so two traversals can never interleave over one "
                "cursor"
            )
        if self._loader.closed:
            raise RuntimeError(
                "this batch iterator's NativeDataLoader is closed; no "
                "further batch can be delivered"
            )
        # --- step 2: the captured one-epoch countdown.
        if self._to_yield == 0:
            self._finish()
            raise StopIteration
        # --- step 3: the dataset must still be able to materialize. A
        # closed dataset is a supported, deterministic situation: planning
        # and state keep working and only materialization refuses, having
        # claimed nothing and advanced nothing.
        dataset = sampler.dataset
        if dataset.closed:
            raise RuntimeError(
                "cannot materialize a batch: this loader's "
                "NativeTensorDataset is closed (its host snapshots were "
                "released by close()). Nothing was claimed, nothing was "
                "allocated, and no batch position was consumed."
            )
        # --- steps 4-9: reject a second transaction, snapshot the
        # committed position, plan the candidate indices and the candidate
        # post-delivery position, mint the serial, publish the claim.
        # Nothing here moves the committed epoch or cursor.
        serial, indices = sampler._claim_batch(self._token)
        self._txn_serial = serial
        delivered = False
        try:
            # --- Phase 2: construct. Both objects become iterator-owned
            # the instant they exist, so the unconditional rollback below
            # releases them however this call fails.
            self._features = dataset.feature_batch(indices)
            self._targets = dataset.target_batch(indices)
            # --- Phase 3: publish the pending-delivery record.
            sampler._publish_pending(serial, self._token)
            # --- Phase 4: commit the candidate position, then deliver.
            sampler._commit_pending(serial, self._token)
            pair = _deliver_batch(self)
            # The seam may have been patched to re-enter and close this
            # iterator or its loader, which rolls the transaction back. So
            # completion is re-verified rather than assumed: the same
            # serial must still be live, this token must still own it, it
            # must not already be resolved, the resource record must still
            # hold the batch, and neither object may have been closed out
            # from under the delivery.
            if (self._closed
                    or self._loader.closed
                    or self._txn_serial != serial
                    or self._features is None
                    or not sampler._complete_pending(serial, self._token)):
                raise RuntimeError(
                    f"batch-delivery transaction {serial} was resolved by a "
                    f"reentrant operation while it was being delivered; the "
                    f"batch was not delivered and no position was consumed"
                )
            # --- Phase 6: ownership transfers to the caller. Both
            # references are dropped **here**, before anything else can
            # raise, so no close, rollback, or finalizer can ever reach a
            # delivered batch.
            self._features = None
            self._targets = None
            self._txn_serial = 0
            # The countdown moves exactly once, and only now.
            self._to_yield -= 1
            delivered = True
            return pair
        finally:
            # Unconditional, so no failure path — including an
            # asynchronous exception — can skip it. Exact-match, so it
            # does nothing at all once the handoff completed or a
            # reentrant close already performed it.
            if not delivered:
                self._rollback(serial)

    # -- transaction cleanup --------------------------------------------

    def _rollback(self, serial):
        """The Phase-5 rollback, in the contracted order: restore the exact
        pre-delivery position and clear the sampler's record (both inside
        the sampler, through its non-failing write seam), then clear this
        iterator's resource record, close the undelivered feature tensor,
        and release the host target reference.

        The sampler half is **exact-match**, so it can never disturb a
        newer transaction, a foreign iterator's transaction, a completed
        one, or an already rolled-back one — and so it is idempotent. The
        resource half needs no match and is released unconditionally: an
        iterator owns at most one transaction's resources at a time, and
        the only path that ever detaches them without releasing them is
        Phase 6, which hands them to the caller. So "release whatever this
        iterator still holds" can never reach a delivered batch, and it
        catches a resource that a *reentrant* close left the interrupted
        call to re-attach afterwards.
        """
        matched = self._sampler._rollback_pending(serial, self._token)
        self._release_resources()
        return matched

    def _release_resources(self):
        """Drop the resource half and close what it owned. Idempotent.

        The references are cleared **before** the tensor is closed, so a
        second arrival finds nothing to close and no path can double-close
        one. It can never reach a delivered batch: Phase 6 clears both
        references before it returns.
        """
        features = self._features
        self._features = None
        self._targets = None
        self._txn_serial = 0
        if features is not None:
            features.close()

    # -- lifecycle -------------------------------------------------------

    def _supersede(self):
        """Mark this iterator replaced by a newer one over the same loader.

        It keeps its active-iteration participation until it closes,
        exhausts, or is finalized, and it cannot be holding a pending
        batch — supersession is refused outright while a transaction is in
        flight.
        """
        self._superseded = True

    def _detach(self):
        """Release the loader's current-iterator slot if this iterator
        still holds it. A superseded iterator no longer does, so this is
        the no-op it should be."""
        loader = self._loader
        if loader is not None and loader._iterator is self:
            loader._iterator = None

    def _finish(self):
        """Ordinary exhaustion: release participation and detach, once."""
        self._exhausted = True
        self._detach()
        self._sampler._end_iteration(self._token)

    def close(self):
        """Roll back any in-flight transaction, release everything this
        iterator owns, and detach. Idempotent; returns ``None``; **never
        refused**, in any state.

        Refusing it during a transaction would strand exactly the
        resources it exists to release, so it is the recovery path rather
        than another guarded operation. It **never touches a delivered
        batch**: after Phase 6 the iterator holds no reference to one.
        """
        if self._closed:
            return None
        # Set first, so a reentrant arrival from a finalizer during the
        # rollback below is a no-op rather than a second pass.
        self._closed = True
        try:
            self._rollback(self._txn_serial)
        finally:
            # Belt and braces: whatever the rollback did or did not match,
            # this iterator must retain nothing releasable afterwards.
            self._release_resources()
            self._detach()
            self._sampler._end_iteration(self._token)
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        """Best-effort cleanup only, exactly as ``NativeTensor.__del__`` is.

        It exists so an abandoned iterator cannot permanently hold an
        active-iteration count or an undelivered batch. **No correctness
        depends on it and no test asserts collection timing**: every
        delivered batch is the caller's and every pending one is reachable
        through an explicit ``close()``. Exceptions are suppressed —
        a finalizer that raised during interpreter shutdown would be
        noise, not information.
        """
        try:
            if not self._closed:
                self.close()
        except BaseException:      # pragma: no cover - finalizer safety net
            pass

    def __repr__(self):
        """Status and countdown only. Never a batch, a value, an index, an
        address, an object id, or a transaction record."""
        if self._closed:
            status = "closed"
        elif self._superseded:
            status = "superseded"
        elif self._exhausted or self._to_yield == 0:
            status = "exhausted"
        else:
            status = "open"
        return f"<_NativeBatchIterator remaining={self._to_yield} ({status})>"


class NativeDataLoader:
    """Iteration over a ``NativeBatchSampler``, yielding native mini-batches.

    ``NativeDataLoader(sampler)`` — and nothing else. The loader takes a
    sampler rather than a dataset plus six keyword arguments, so the
    composition is explicit, the dataset is named once, and each
    configuration value is spelled in exactly one constructor instead of
    being duplicated across two with two sets of validation and two sets
    of error messages.

    ``iter(loader)`` returns a fresh private iterator for **one epoch**
    and supersedes any previous one, so the ordinary
    break-and-restart pattern continues cleanly from the committed
    position::

        for features, targets in loader:
            ...
            features.close()
            break
        for features, targets in loader:     # continues where that stopped
            ...
            features.close()

    Nested or overlapping iteration fails **loudly**, on the superseded
    loop's next step, rather than interleaving two traversals over one
    cursor.

    **The caller owns and closes every delivered feature batch.** Closing
    the loader never closes, mutates, invalidates, or retains one — and it
    never closes the sampler or the dataset either.

    ``state_dict()`` and ``load_state_dict(state)`` carry where a loader
    stopped, in memory, exactly: a three-key tagged wrapper around the
    sampler's own state, transactional on the way back in, and enough on
    its own to reproduce every remaining batch of an interrupted epoch.
    See the module docstring for the schema and the load ordering.
    """

    # No ``__dict__``: the three members are read-only properties, so
    # ``loader.closed = False`` raises rather than shadowing one, and no
    # attribute can be injected onto a loader.
    __slots__ = ("_sampler", "_iterator", "_closed")

    def __init__(self, sampler):
        # ``isinstance`` rather than an exact type check, on J2's recorded
        # precedent for the same question one level down: the loader reads
        # only the sampler's documented surface, which any subclass
        # inherits intact.
        if not isinstance(sampler, NativeBatchSampler):
            raise TypeError(
                f"sampler must be a NativeBatchSampler, got "
                f"{type(sampler).__name__}"
            )
        # Stored by **identity**: not copied, cloned, wrapped, or
        # reconstructed. A sampler whose dataset is already closed is
        # accepted, exactly as the sampler accepts one — planning and
        # state keep working, and only materialization refuses.
        #
        # Construction plans nothing, allocates nothing native,
        # materializes no batch, and moves no position.
        self._sampler = sampler
        self._iterator = None
        self._closed = False

    # -- read-only surface ----------------------------------------------

    @property
    def sampler(self):
        """The sampler this loader was constructed with — **identity**,
        not a copy. Readable before and after close."""
        return self._sampler

    @property
    def dataset(self):
        """``self.sampler.dataset`` — the exact dataset object, by
        identity. A convenience only; the loader holds no second
        reference and is not a second authority on it."""
        return self._sampler.dataset

    @property
    def closed(self):
        return self._closed

    # -- iteration -------------------------------------------------------

    def __iter__(self):
        """A fresh one-epoch iterator, superseding any previous one.

        ``RuntimeError`` if the loader is closed, and ``RuntimeError``
        while a batch transaction is in flight — where supersession is
        **refused rather than performed**, because detaching the iterator
        that owns the undelivered batch would strand both a position and a
        tensor. That is reachable only from a reentrant caller, and it
        changes nothing: no iterator is created, no slot moves, and no
        participation is taken.
        """
        if self._closed:
            raise RuntimeError(
                "cannot iterate a closed NativeDataLoader; its iterator slot "
                "was released by close()"
            )
        self._sampler._require_no_transaction("iterate this loader")
        iterator = _NativeBatchIterator(self)
        previous = self._iterator
        if previous is not None:
            previous._supersede()
        self._iterator = iterator
        return iterator

    # -- state (§11.3, §12.5) -------------------------------------------

    def state_dict(self):
        """A fresh, plain, JSON-compatible snapshot of §11.3's schema.

        Exactly three keys — ``format``, ``format_version``, and
        ``sampler`` — the last being **exactly** the sampler's own §11.2
        state, unchanged and undecorated. The loader duplicates none of
        the sampler's six configuration and position fields at the root,
        because it owns none of them and a second copy could disagree
        with the first.

        Every container is new at every call: the root dict, the nested
        sampler dict, its nested ``dataset`` dict, and the
        ``feature_shape`` list. So the result shares nothing mutable with
        the loader, the sampler, the dataset, the permutation cache, or a
        previous result, and a caller may edit what they are given
        without reaching anything.

        It carries no permutation, no dataset content, no NumPy object,
        no ``NativeTensor``, no generator, no transaction serial, no
        active-iteration token, no iterator, no cache, no object id, no
        address, nothing callable, and nothing whose size grows with the
        number of samples. Every field is JSON-native and passes the
        checkpoint's existing ``_validated_metadata`` unchanged, which is
        what lets a **caller** carry it through the existing version-3
        metadata channel without the archive growing a field or a
        version. Placing it there is the caller's step (J5): this module
        imports no checkpoint code and no checkpoint code knows a loader
        exists.

        **Pure.** It moves no epoch, no cursor, and no configuration;
        creates, supersedes, and closes no iterator; changes no
        active-iteration tracking; allocates nothing native; and closes
        or reopens nothing. Repeated calls return equal values in
        distinct containers.

        Allowed immediately after construction, between batches while an
        iterator is active, after an iterator is exhausted or superseded,
        at an epoch boundary, mid-epoch, with a **closed dataset**, and
        after the loader's own ``close()`` — recording where a loader
        stopped is not a resource operation.

        **Refused with ``RuntimeError`` while a §9.4 batch transaction is
        in flight**, reading nothing and changing nothing. Inside Phase 4
        the candidate position has been applied but the batch has not
        been delivered, so there is no honest single answer: reporting
        the candidate would expose a committed cursor that skipped an
        undelivered batch, and reporting the pre-delivery one would
        contradict the sampler's own fields. The refusal comes from the
        **sampler's** existing guard rather than a second transaction
        authority here, so the two can never disagree about what is in
        flight.
        """
        # The sampler's own snapshot is the guard as well as the payload:
        # it refuses mid-transaction, and it is where the freshness of
        # every nested container is already contracted. Taken first, so
        # nothing at all is built when it refuses.
        sampler_state = self._sampler.state_dict()
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "sampler": sampler_state,
        }

    def load_state_dict(self, state):
        """Restore this loader's sampler configuration and position from a
        §11.3 state. Returns ``None``.

        **Transactional.** Every check runs before anything is written,
        and the one remaining step cannot fail, so a rejected load leaves
        the loader, its sampler, its dataset, the position, the
        configuration, the permutation cache's behavior, the iterator
        slot, the active-iteration tracking, and the native live-storage
        count exactly as it found them. There is no rollback path here
        because there is nothing to roll back from.

        The order is §12.5's, exactly:

        1. **Closed guard** — a closed loader refuses, *before* ``state``
           is inspected at all. Restoring a position into a closed loader
           is meaningless, and it must never silently reopen one. There
           is no reopen method, and ``state_dict()`` stays readable.
        2. **Transaction guard** — a live §9.4 claim or pending-delivery
           record refuses, reading nothing, mutating nothing, and
           disturbing neither the record nor the batch it owns. It comes
           before the iteration guard because it is the more specific and
           the more dangerous condition.
        3. **Active-iteration guard** — any live iterator participation
           refuses, including a superseded iterator that has not yet
           closed, exhausted, or been finalized. An iterator captures its
           epoch's remaining batch count when it is created (§9.3), so
           replacing the position underneath it would leave that
           countdown describing a position that no longer exists.
        4. **The wrapper** — exact ``dict``, exact three-key set, exact
           ``str`` ``format`` with exactly this module's tag, exact
           ``int`` ``format_version`` (``bool`` rejected) equal to 1, and
           an exact ``dict`` under ``sampler``.
        5. **The nested sampler state**, delegated *whole* to the
           sampler's validation-only seam, which preserves every J2/J3
           ordering — key set, format, version, the dataset block's shape
           and its four compatibility fields in order, the configuration
           types, the ranges, the zero-batch joint rule, and the cursor
           last. It is **not** restated here: one rule, one authority.
           The public sampler ``load_state_dict`` is deliberately *not*
           called, because it would mutate before this wrapper's
           transaction is complete.
        6. **Commit**, through the sampler's non-failing write seam —
           six already-validated ``int``/``bool`` assignments plus a
           cache invalidation, the same seam a delivery commit and a
           rollback share.

        **Dataset identity is validated, never adopted**, and the loader
        keeps the exact sampler and the exact dataset objects it already
        had. **Configuration is adopted**: ``seed``, ``shuffle``,
        ``batch_size``, ``drop_last``, ``epoch``, and ``cursor`` all come
        from the state, so a loader deliberately built with a different
        seed, batch size, drop-last setting, and position adopts the
        state's validated values — the constructor's configuration is not
        authoritative after a restoration. Object identity is preserved
        absolutely: ``id(loader)``, ``loader.sampler``, and
        ``loader.dataset`` are unchanged, no iterator is created or
        replaced, and nothing is rebound or recreated.

        Nothing is cast, coerced, truncated, clamped, wrapped, rounded,
        defaulted, or ignored: ``True`` is not ``1``, ``1`` is not
        ``True``, ``1.0`` is not ``1``, ``"1"`` is not ``1``, a NumPy
        scalar is not a Python one, a missing key is not a default, an
        unknown key is not ignored, and a cursor past the end is not
        clamped.
        """
        # 1. The lifecycle guard, ahead of every read of ``state``.
        if self._closed:
            raise RuntimeError(
                "cannot load state into a closed NativeDataLoader; restoring "
                "a position into one would be meaningless, and a load must "
                "never silently reopen it. state_dict() is still readable."
            )
        sampler = self._sampler
        # 2. A live batch handoff, next: the more specific and the more
        #    dangerous of the two remaining conditions. Replacing a
        #    position underneath one would make its pre/post pair describe
        #    a stream that no longer exists.
        sampler._require_no_transaction("load state into this loader")
        # 3. Then any live iterator participation, whose captured
        #    countdown a replaced position would strand.
        sampler._require_no_active_iteration("load state into this loader")
        # 4. The wrapper: container, exact key set, tag, version, and the
        #    nested container's type. Exact-type discipline throughout — a
        #    dict subclass, an OrderedDict, a mapping proxy, a JSON string,
        #    a list, and a tuple are each refused rather than converted.
        if type(state) is not dict:
            raise TypeError(
                f"loader state must be a dict, got {type(state).__name__}"
            )
        _require_exact_keys(state, _STATE_FIELDS, "loader state")
        format_tag = state["format"]
        if type(format_tag) is not str:
            raise TypeError(
                f"loader state 'format' must be a str, got "
                f"{type(format_tag).__name__}"
            )
        if format_tag != _FORMAT:
            raise ValueError(
                f"loader state format mismatch: expected {_FORMAT!r}, got "
                f"{format_tag!r}. A sampler state is not a loader state; the "
                f"tags exist so the two cannot be confused."
            )
        version = _require_exact_int(state["format_version"],
                                     "loader state 'format_version'")
        if version not in _SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"unsupported loader state format version {version}; "
                f"supported: {list(_SUPPORTED_FORMAT_VERSIONS)}"
            )
        nested = state["sampler"]
        if type(nested) is not dict:
            raise TypeError(
                f"loader state 'sampler' must be a dict, got "
                f"{type(nested).__name__}"
            )
        # 5. The whole of the nested validation, delegated. It mutates
        #    nothing and returns the six values; the public sampler loader
        #    is deliberately not used, because it would commit before this
        #    wrapper's transaction finished.
        values = sampler._validate_state(nested)
        # 6. The commit — the shared non-failing write seam. Six
        #    already-validated assignments and a cache invalidation, none
        #    of which can fail, allocate native storage, replace the
        #    sampler, replace the dataset, or create an iterator.
        sampler._assign_state(*values)
        return None

    # -- lifecycle -------------------------------------------------------

    def close(self):
        """Close the current iterator — performing any in-flight rollback
        through it — release the iterator slot, and mark the loader
        closed. Idempotent; returns ``None``; **never refused**.

        It never closes a delivered batch, never closes the sampler, never
        closes the dataset, and never alters a committed position.
        Afterwards ``iter()`` raises ``RuntimeError``; ``closed``,
        ``sampler``, ``dataset``, and ``repr`` still work.
        """
        iterator = self._iterator
        try:
            if iterator is not None:
                iterator.close()
        finally:
            self._iterator = None
            self._closed = True
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __repr__(self):
        """Configuration, position, and lifecycle only — delegated from the
        sampler. Never a feature value, a target, a permutation, a
        fingerprint, a batch, an address, an object id, or a pending
        resource. Valid after both loader close and dataset close."""
        sampler = self._sampler
        return (
            f"NativeDataLoader(samples={sampler.dataset.samples}, "
            f"batch_size={sampler.batch_size}, shuffle={sampler.shuffle}, "
            f"seed={sampler.seed}, drop_last={sampler.drop_last}, "
            f"epoch={sampler.epoch}, cursor={sampler.cursor}, "
            f"batches_per_epoch={sampler.batches_per_epoch}, "
            f"remaining={sampler.remaining}, closed={self._closed})"
        )
