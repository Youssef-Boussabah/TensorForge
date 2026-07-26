"""NativeGenerator — explicit, inspectable, serializable native random
state (Phase G, milestone G1; contract locked in
docs/native_rng_dropout_design.md §3–§5).

``NativeGenerator`` is the Python half of Phase G's central split:
**random state is Python-managed and native kernels are stateless**. A
generator holds the complete key for a stream — an algorithm identifier
and version, a 64-bit seed, and a counter of *committed* stochastic
calls — and hands out one call index per operation. A future native
kernel (milestone G2) receives ``(seed, call_index)`` and computes a mask
from it; it never reads, holds, or advances a generator.

**This milestone generates no random values.** There is no finalizer, no
bit derivation, no mask, and no Dropout here: G1 is state ownership and
the call transaction only.

What it is
----------

A **pure-Python value holder**. It owns no native storage, allocates no
native memory, and has **no** ``close()`` — inventing one would advertise
a lifetime that does not exist. It is not a ``NativeTensor``, not a
``NativeParameter``, not a buffer, and it never enters the tensor
state-dict key space. Constructing, registering, inspecting, and
discarding generators leaves the native live-storage count untouched.

State
-----

Exactly four fields, all read-only properties:

===================  =====  =================================================
``algorithm``        str    ``"tensorforge.splitmix64"``
``algorithm_version``int    ``1``
``seed``             int    unsigned 64-bit, ``0 <= seed <= 2**64 - 1``
``calls``            int    unsigned 64-bit, **committed** stochastic calls
===================  =====  =================================================

``(algorithm, algorithm_version)`` is the whole compatibility key: a
change to the future derivation must mint a new pair rather than
reinterpret a saved seed. Seed and counter are exact Python ints — never
floats, and never truncated, because a ``uint64`` above ``2**53`` is not
representable in a double.

State changes go through ``state()`` / ``load_state()`` / ``reseed()`` /
``reset()``, each of which validates everything before mutating anything,
so a rejected call leaves the generator bit-identical. ``state()``
returns a fresh plain dict that shares nothing with the generator.

Identity, not value
-------------------

No ``__eq__`` and no ``__hash__`` override: two generators with equal
state are two generators, exactly as for ``NativeParameter``,
``NativeTensor``, and ``NativeModule``. Every traversal, deduplication,
and (later) checkpoint rule keys on ``id()``. Sharing is done by sharing
the object — there is no ``clone()``, ``split()``, ``jump()``, or
``advance()``, and ``copy``/``deepcopy``/pickle are refused, because a
copied generator would silently produce identical masks in two places,
which is the exact failure mode explicit state exists to prevent.

The call transaction
--------------------

One successful stochastic forward will consume exactly one call. The
private protocol is ``_reserve_call()`` → (do the work) →
``_commit_call(token)`` or ``_abandon_call(token)``, and the counter
advances **only** at commit, after the operation's result has been
published. Every failure path abandons, so a failed forward consumes
nothing.

Each generator owns one private ``threading.RLock``. It covers
reservation claiming and publication, commit and cancellation, counter
and state inspection, state replacement, reseeding, and resetting — and
nothing else: the numerical work a caller does between reserve and commit
runs with the lock released, so it can never be held across a native call
or an arbitrarily long kernel.

Reservation creation is a **two-phase claim / construct / publish /
deliver** transaction, and the token is built with **no generator lock
held**:

1. *claim* — under the lock, reject an active reservation, a claim
   already in progress, and an exhausted counter; capture the candidate
   index and serial; publish only the internal construction claim.
   ``calls`` and the next serial do not move. Release the lock.
2. *construct* — build the token owning nothing. If it raises, the
   ``finally`` reacquires the lock, verifies the matching claim, clears
   it, publishes no reservation, and re-raises; the generator is
   immediately reusable and no serial was skipped.
3. *publish* — reacquire the lock, verify the claim still matches, write
   the active reservation, advance the never-reused serial exactly once,
   and clear the claim.
4. *deliver* — hand the token back to the caller.

Those last two steps fail differently, and **clearing the claim does not
cover the gap between them**. Once a reservation is published the claim
is gone; an asynchronous exception arriving before the caller receives
the token would leave an active reservation whose only token is being
dropped — unreleasable, and blocking every later reservation. So a failed
delivery runs its own cleanup, ``_release_undelivered``, which cancels
**only** a live reservation matching that token's generator, serial, and
index exactly, leaves ``calls`` untouched, and leaves any newer, foreign,
committed, or already-cancelled reservation strictly alone. A failed
delivery consumes an opaque reservation serial; it never consumes a call
index.

The claim is what makes the window safe. While it is set, another
reservation, ``load_state``, ``reseed``, ``reset``, and the
multi-generator transaction all raise ``RuntimeError`` and mutate
nothing; state inspection is unaffected. And because no generator lock is
held during construction, a finalizer that allocation happens to run
cannot start a multi-generator transaction while this thread owns one
generator's lock — so it cannot invert the global lock order (§9.6) and
cannot deadlock.

The lock stays an ``RLock``. No user code, callback, or generator-owned
allocation runs while it is held, but CPython can still start a
collection at any container allocation, and the remaining critical
sections do allocate small objects: the dict ``state()`` returns, the
tuple ``_snapshot_state()`` returns, and the message and traceback of
every ``RuntimeError``/``TypeError`` raised under the lock. An ``RLock``
turns a finalizer that re-enters through one of those into a
deterministic refusal (for a mutation) or a correct read (for an
inspection) rather than a permanent deadlock, and it weakens nothing:
across threads it behaves exactly like a ``Lock``, and re-acquiring a
lock the current thread already owns never blocks, so it cannot reorder
acquisitions.

The lock is **serialization for correctness, not a performance feature,
and parallel stochastic execution is not claimed**: at most one
reservation exists at a time, and a second caller — another thread, or
the same thread re-entering — fails deterministically *before* an index
is minted. That is what makes a committed call index provably unique.

``_reserve_call()`` returns an **opaque token** carrying the owning
generator, the reserved index, and a per-generator serial that is never
reused. Commit and cancel accept only the currently active matching
token; stale, foreign, duplicated, already-committed, and
already-cancelled tokens are refused with ``RuntimeError`` and change
nothing, and a non-token raises ``TypeError``. The token type is private
implementation, not a public API — callers receive tokens, they never
build them.
"""

import secrets
import threading
from collections import namedtuple
from collections.abc import Mapping
from contextlib import ExitStack

# The compatibility key for the stream. Nothing else identifies it: a
# change to the (future) derivation must mint a new pair rather than
# reinterpret a saved seed under different rules.
ALGORITHM = "tensorforge.splitmix64"
ALGORITHM_VERSION = 1

# Unsigned 64-bit domain for the seed and the call counter. The counter
# never wraps: a reservation is refused at UINT64_MAX, so the largest
# usable call index is UINT64_MAX - 1 and ``calls`` stays in range.
UINT64_MAX = 2**64 - 1

# The exact key set of a generator state mapping — no more, no fewer.
_STATE_FIELDS = ("algorithm", "algorithm_version", "seed", "calls")

# Sentinel for "no reservation is active". Serials start at 1, so 0 can
# never collide with a real reservation.
_NO_RESERVATION = 0

# One target of a multi-generator state transaction: the generator, the
# candidate state mapping, and a label used in error messages (the
# caller's canonical name for it).
GeneratorStateEntry = namedtuple(
    "GeneratorStateEntry", ("label", "generator", "state")
)


def _validate_uint64(value, what):
    """Validate ``value`` as an exact unsigned 64-bit Python ``int``.

    Exact-type discipline, matching the checkpoint metadata validator:
    ``bool`` is not a seed (``True`` is not ``1`` here) and a NumPy
    integer scalar is not a Python int. Python ints are arbitrary
    precision, so an out-of-range value is a ``ValueError`` rather than a
    silent truncation."""
    if type(value) is not int:
        raise TypeError(
            f"{what} must be an int, got {type(value).__name__}"
        )
    if not 0 <= value <= UINT64_MAX:
        raise ValueError(
            f"{what} must be in [0, {UINT64_MAX}], got {value}"
        )
    return value


class _ReservationToken:
    """Private, inert proof of one reservation.

    It exposes no behavior at all: only the generator that minted it can
    interpret it, and only under that generator's lock. ``__slots__``
    keeps it small and closed, and the recorded outcome is written by the
    generator (never by a caller) purely so that a duplicate commit can
    say *which* duplicate it was."""

    __slots__ = ("_generator", "_serial", "_index", "_outcome")

    def __init__(self, generator, serial, index):
        self._generator = generator
        self._serial = serial
        self._index = index
        self._outcome = None  # None | "committed" | "abandoned"

    def __repr__(self):
        state = self._outcome or "outstanding"
        return f"<NativeGenerator reservation {self._serial} ({state})>"


def _deliver_reservation(token):
    """The publish → deliver seam of the reservation transaction.

    A deliberate no-op. It exists so that the window between publishing a
    reservation and the caller receiving its token is *addressable*: an
    asynchronous exception there leaves an active reservation whose only
    token is being dropped, and that path has to be provable rather than
    argued. Tests monkeypatch this module attribute to raise exactly
    there.

    Private and module-level: never exported, never referenced from a
    public API, and it does nothing a caller could depend on."""
    return token


def _require_token(token, what):
    """Reject a non-token before any lock is taken: passing something
    that is not a reservation token is a caller bug, not a state
    conflict, so it is a ``TypeError`` and touches nothing."""
    if not isinstance(token, _ReservationToken):
        raise TypeError(
            f"cannot {what} a call: expected a reservation token from this "
            f"generator, got {type(token).__name__}"
        )


class NativeGenerator:
    """Explicit native random state: an algorithm identity, a 64-bit
    seed, and a counter of committed stochastic calls.

    ``NativeGenerator(seed=None)`` draws one 64-bit seed from operating
    system entropy; an explicit ``seed`` must be an exact ``int`` in
    ``[0, 2**64 - 1]``. See the module docstring for the full state,
    identity, and call-transaction contract.

    This milestone (G1) generates **no random values** — the generator is
    state and a call transaction, and the kernel that will consume
    ``(seed, call_index)`` does not exist yet."""

    # No __dict__: the four state fields are read-only properties, so
    # ``generator.seed = 7`` raises rather than shadowing the property,
    # and no attribute can be injected onto a generator.
    __slots__ = (
        "_lock",
        "_seed",
        "_calls",
        "_active_serial",
        "_active_index",
        "_next_serial",
        "_claim_serial",
        "_claim_index",
    )

    def __init__(self, seed=None):
        # Validate before writing anything: construction never leaves a
        # partially initialized generator observable.
        if seed is None:
            # The ONLY use of non-deterministic entropy in Phase G. It
            # happens exactly once per generator, and the drawn value is
            # immediately readable through ``.seed`` and serializable.
            # Nothing here consults the clock, the process id, an
            # address, NumPy's global RNG, or Python's ``random``.
            seed = secrets.randbits(64)
        else:
            _validate_uint64(seed, "seed")
        # An RLock, not a plain Lock. No user code, callback, or
        # generator-owned allocation runs while it is held — the token is
        # built with it released — but the remaining critical sections
        # still allocate (state()'s dict, _snapshot_state()'s tuple, and
        # every exception raised under the lock), and any container
        # allocation can start a CPython collection and therefore run a
        # finalizer. An RLock makes such a re-entry a deterministic
        # refusal or a correct read instead of a permanent deadlock. It
        # weakens nothing: across threads it is exactly a Lock, and
        # re-acquiring an already-owned lock never blocks, so it cannot
        # reorder acquisitions.
        self._lock = threading.RLock()
        self._seed = seed
        self._calls = 0
        self._active_serial = _NO_RESERVATION
        self._active_index = 0
        self._next_serial = 1
        # The construction claim: the candidate identity of a reservation
        # that has been decided on but whose token does not exist yet.
        # Set between phase 1 and phase 3 of _reserve_call, and the only
        # thing that makes that window visible to another caller — who
        # would otherwise see a generator with no active reservation and a
        # counter that is about to describe a different stream.
        self._claim_serial = _NO_RESERVATION
        self._claim_index = 0

    # -- state (read-only) ---------------------------------------------
    #
    # Every read takes the lock. For ``seed`` and ``calls`` that is what
    # makes a read atomic against a concurrent commit or reseed; for
    # ``algorithm`` and ``algorithm_version`` it is uniformity with the
    # locked contract rather than necessity (both are constants that no
    # operation can change — ``load_state`` validates equality).

    @property
    def algorithm(self):
        """The algorithm identifier, ``"tensorforge.splitmix64"``."""
        with self._lock:
            return ALGORITHM

    @property
    def algorithm_version(self):
        """The algorithm version, ``1``."""
        with self._lock:
            return ALGORITHM_VERSION

    @property
    def seed(self):
        """The unsigned 64-bit seed, an exact Python int."""
        with self._lock:
            return self._seed

    @property
    def calls(self):
        """The number of **committed** stochastic calls, an exact
        unsigned 64-bit Python int. Reservations that were abandoned,
        refused, or are still outstanding are not counted."""
        with self._lock:
            return self._calls

    def state(self):
        """An independent ``{algorithm, algorithm_version, seed, calls}``
        snapshot as a fresh plain dict.

        It shares nothing with the generator (all four values are
        immutable), so mutating the returned dict affects nothing.
        Reading state creates no reservation and advances no counter."""
        with self._lock:
            return {
                "algorithm": ALGORITHM,
                "algorithm_version": ALGORITHM_VERSION,
                "seed": self._seed,
                "calls": self._calls,
            }

    # -- state replacement ---------------------------------------------

    def load_state(self, state):
        """Replace this generator's seed and call counter from ``state``
        **in place**, preserving object identity.

        ``state`` must be a mapping with exactly the four
        ``state()`` keys. The algorithm identifier and version must equal
        this generator's; the seed and counter must be exact ints in the
        unsigned 64-bit range. Everything is validated before anything is
        assigned, so a rejected load leaves the generator bit-identical —
        and the assignment itself cannot fail.

        Raises ``RuntimeError`` if a reservation is outstanding: the
        reserved index is only meaningful relative to the seed it was
        reserved under, so replacing state underneath an in-flight draw
        is refused rather than silently corrupting the stream."""
        seed, calls = self._validate_state(state)
        with self._lock:
            self._require_no_reservation("load state into")
            self._seed = seed
            self._calls = calls

    def reseed(self, seed):
        """Set a new ``seed`` (validated exactly as the constructor does)
        and reset the call counter to ``0`` — "a different stream from
        the start", without rebuilding the model.

        Refused with ``RuntimeError``, changing nothing, while a
        reservation is outstanding."""
        _validate_uint64(seed, "seed")
        with self._lock:
            self._require_no_reservation("reseed")
            self._seed = seed
            self._calls = 0

    def reset(self):
        """Reset the call counter to ``0``, keeping the seed — "the same
        stream again".

        Refused with ``RuntimeError``, changing nothing, while a
        reservation is outstanding."""
        with self._lock:
            self._require_no_reservation("reset")
            self._calls = 0

    @staticmethod
    def _validate_state(state):
        """Validate a candidate state mapping completely and return the
        ``(seed, calls)`` it carries. Reads no live field, so it runs
        outside the lock and the critical section stays tiny."""
        if not isinstance(state, Mapping):
            raise TypeError(
                f"generator state must be a mapping, got "
                f"{type(state).__name__}"
            )
        keys = set(state.keys())
        expected = set(_STATE_FIELDS)
        if keys != expected:
            missing = sorted(expected - keys)
            unexpected = sorted(str(key) for key in keys - expected)
            raise ValueError(
                f"generator state must have exactly the keys "
                f"{list(_STATE_FIELDS)}: missing {missing}, "
                f"unexpected {unexpected}"
            )
        algorithm = state["algorithm"]
        if not isinstance(algorithm, str):
            raise TypeError(
                f"generator state 'algorithm' must be a str, got "
                f"{type(algorithm).__name__}"
            )
        if algorithm != ALGORITHM:
            raise ValueError(
                f"generator state algorithm mismatch: this generator uses "
                f"{ALGORITHM!r}, the state carries {algorithm!r}"
            )
        version = state["algorithm_version"]
        if type(version) is not int:
            raise TypeError(
                f"generator state 'algorithm_version' must be an int, got "
                f"{type(version).__name__}"
            )
        if version != ALGORITHM_VERSION:
            raise ValueError(
                f"generator state algorithm version mismatch: this "
                f"generator uses version {ALGORITHM_VERSION}, the state "
                f"carries {version}"
            )
        seed = _validate_uint64(state["seed"], "generator state 'seed'")
        calls = _validate_uint64(state["calls"], "generator state 'calls'")
        return seed, calls

    def _snapshot_state(self):
        """The ``(seed, calls)`` pair ``_assign_state`` would restore.
        Private; the rollback half of the multi-generator transaction."""
        with self._lock:
            return self._seed, self._calls

    def _assign_state(self, seed, calls):
        """Write an already-validated ``(seed, calls)`` pair.

        Private, and **non-failing by construction**: two integer
        assignments to ``__slots__`` fields, with no validation and no
        reservation check because the caller has already done both. It is
        the single write seam — ``load_state``, ``reseed``, ``reset``, and
        the multi-generator transaction all end here — and the reason the
        transaction's rollback is exact: a rollback that could itself
        raise would be able to leave one generator restored and another
        not.

        The lock is re-entered rather than assumed, which the RLock makes
        safe: a caller that already holds it (the multi-generator
        transaction) pays only a recursion count."""
        with self._lock:
            self._seed = seed
            self._calls = calls

    def _require_no_reservation(self, what):
        """Refuse a state change while a reservation is in flight —
        either published, or claimed but not yet published (§3.6).

        The caller holds the lock (possibly re-entrantly). Both cases
        matter: an *active* reservation holds an index describing the
        current seed, and a *claimed* one has already decided on an index
        against that same seed, so replacing state underneath either would
        make that index describe a stream that no longer exists."""
        if self._active_serial != _NO_RESERVATION:
            raise RuntimeError(
                f"cannot {what} this generator while a call reservation is "
                f"outstanding: the reserved index describes the current "
                f"seed. Complete or abandon the call first."
            )
        if self._claim_serial != _NO_RESERVATION:
            raise RuntimeError(
                f"cannot {what} this generator while a call reservation is "
                f"being constructed: its index has already been decided "
                f"against the current seed."
            )

    # -- the call transaction ------------------------------------------
    #
    # Private: the only intended caller is the (future, G3) differentiable
    # Dropout operation. Deciding the index, claiming it, and advancing
    # the counter all happen under the lock, so no two callers can ever
    # receive the same successful call index.
    #
    # Creation is deliberately split into three steps so that the one
    # allocation involved — minting the token — happens with **no
    # generator lock held**. Constructing an object can run interpreter
    # finalization, and a finalizer may do anything, including starting a
    # multi-generator state transaction (§9.6). If this thread owned one
    # generator's lock at that moment it could acquire a second one out of
    # the global order and deadlock against a thread doing the reverse.
    # Holding nothing makes that impossible: the finalizer's transaction
    # simply takes the global order like any other caller, and is refused
    # by the construction claim if it names this generator.

    def _reserve_call(self):
        """Reserve the next call index and return an opaque token.

        Raises ``RuntimeError`` if a reservation is already outstanding,
        if one is already claimed, or if the counter is exhausted — in
        every case **before** an index is claimed, so the caller receives
        nothing and consumes nothing.

        The transaction is claim → construct → publish → deliver (§3.6),
        and the four failure positions are **different**, so they get
        different cleanup:

        1. **Construction fails** (``MemoryError``, ``KeyboardInterrupt``,
           a reentrant failure from a finalizer). Nothing is published;
           ``_discard_claim`` releases the claim. ``calls`` and the serial
           are unchanged and no serial is skipped.
        2. **Publication fails.** Only possible if the claim no longer
           matches, which would already mean broken internal state. No
           reservation is published, and the claim is released the same
           way.
        3. **Publication succeeded but delivery did not** — an
           asynchronous exception between publishing and the caller
           receiving its token. The claim is *already gone*, so releasing
           it does nothing; what matters is that an **active reservation
           exists whose only token is about to be dropped**. That is what
           ``_release_undelivered`` cancels, matching the token exactly
           and leaving ``calls`` untouched. Without it the generator would
           be permanently stranded: no caller could commit or abandon a
           reservation whose token no longer exists.
        4. **Delivery succeeded.** Both cleanups match nothing and are
           no-ops.

        A failed delivery consumes the internal reservation *serial* —
        that is deliberate, since serials are opaque and never reused —
        but it never consumes a call index and never advances ``calls``.

        While the claim stands, a nested ``_reserve_call`` and every state
        replacement (``load_state``/``reseed``/``reset``, and the
        multi-generator transaction) raise ``RuntimeError`` and mutate
        nothing, whichever thread they arrive from. State *inspection* is
        unaffected and keeps working, as §3.6 says."""
        # -- phase 1: claim, under the lock.
        serial, index = self._claim_reservation()
        token = None
        delivered = False
        try:
            # -- phase 2: construct, holding no generator lock.
            token = _ReservationToken(self, serial, index)
            # -- phase 3: publish, under the lock.
            self._publish_reservation(serial, index)
            # -- phase 4: deliver. The seam exists so this window is
            # addressable at all — see _deliver_reservation.
            _deliver_reservation(token)
            delivered = True
        finally:
            # Two distinct cleanups for two distinct failures. Neither
            # can raise, and neither advances ``calls``.
            self._discard_claim(serial)
            if token is not None and not delivered:
                self._release_undelivered(token)
        return token

    def _claim_reservation(self):
        """Phase 1: decide the next call index and claim it.

        Publishes *only* the construction claim: no active reservation
        exists yet, ``calls`` does not move, and the never-reused serial
        does not advance — so a claim that is later discarded costs
        nothing and skips nothing. Returns the candidate
        ``(serial, index)``."""
        with self._lock:
            if self._active_serial != _NO_RESERVATION:
                raise RuntimeError(
                    "this generator already has an outstanding call "
                    "reservation; a generator serves one stochastic call "
                    "at a time (commit or abandon the first one). "
                    "Concurrent stochastic use of one generator is not "
                    "supported."
                )
            if self._claim_serial != _NO_RESERVATION:
                raise RuntimeError(
                    "this generator is already constructing a call "
                    "reservation: its index has been claimed but its "
                    "token does not exist yet. A generator serves one "
                    "stochastic call at a time, whether the second "
                    "caller is another thread or a reentrant one (a "
                    "finalizer, a callback, or a signal handler)."
                )
            if self._calls >= UINT64_MAX:
                raise RuntimeError(
                    f"this generator's call counter is exhausted "
                    f"({self._calls} calls); it never wraps. Reseed or "
                    f"reset it to start a new stream."
                )
            serial = self._next_serial
            index = self._calls
            self._claim_serial = serial
            self._claim_index = index
            return serial, index

    def _publish_reservation(self, serial, index):
        """Phase 3: turn the claim into the active reservation.

        The claim is rechecked rather than assumed — only its owner can
        clear it, so a mismatch would mean the invariant was already
        broken and is worth failing loudly on. The four assignments
        cannot fail, and the serial advances here, exactly once."""
        with self._lock:
            if (self._claim_serial != serial
                    or self._claim_index != index):
                raise RuntimeError(
                    f"reservation {serial} lost its construction claim "
                    f"before it could be published; this generator's "
                    f"internal state is inconsistent"
                )
            self._next_serial = serial + 1
            self._active_serial = serial
            self._active_index = index
            self._claim_serial = _NO_RESERVATION
            self._claim_index = 0

    def _discard_claim(self, serial):
        """Release the construction claim for ``serial`` if it is still
        standing.

        Verified rather than unconditional, and therefore idempotent: it
        finds nothing to do once phase 3 has published, so it can run in
        an unconditional ``finally``. A claim is only ever clearable by
        the caller that made it — another caller cannot claim ``serial``
        while this one stands — so this can never release someone else's
        window or disturb a published reservation.

        It handles **only** the construction window. A reservation that
        was already published is past this cleanup's reach; that case is
        ``_release_undelivered``."""
        with self._lock:
            if self._claim_serial == serial:
                self._claim_serial = _NO_RESERVATION
                self._claim_index = 0

    def _release_undelivered(self, token):
        """Cancel a reservation that was published but never delivered.

        The failure this exists for: an asynchronous exception lands
        after ``_publish_reservation`` succeeded and before the caller
        receives ``token``. The claim is already cleared, an active
        reservation exists, and the only token for it is about to be
        dropped — so without this the generator would be permanently
        stranded, unable to reserve (one at a time) and unable to release
        (no token).

        Cancellation is **exact-match**: the live reservation must be
        this generator's, must carry this token's serial *and* index, and
        the token must still be unfinished. Anything else is left
        strictly alone, so this can never cancel a newer reservation, a
        foreign generator's reservation, or one that was already
        committed or abandoned. ``calls`` is never advanced, and the
        consumed serial is **not** restored — serials are opaque and
        never reused, so skipping one costs nothing, while restoring one
        could hand a later reservation a serial an existing token already
        carries.

        Non-failing, holds only this generator's lock, and does no
        callback-capable work while holding it — so it cannot participate
        in the global multi-generator lock order (§9.6) at all."""
        with self._lock:
            # The outcome check is *redundant* given the serial check —
            # commit and abandon both clear the active slot, so a
            # finished token can never match a live serial — and is kept
            # anyway so the predicate reads as correct without the reader
            # having to re-derive that invariant. The generator, serial,
            # and index checks are the load-bearing ones.
            if (token._generator is self
                    and token._outcome is None
                    and self._active_serial == token._serial
                    and self._active_index == token._index):
                self._active_serial = _NO_RESERVATION
                token._outcome = "discarded before delivery"

    def _commit_call(self, token):
        """Publish the reserved call: advance ``calls`` by exactly one
        and clear the reservation.

        Only the currently active matching token is accepted. A stale,
        foreign, already-committed, or already-abandoned token raises
        ``RuntimeError`` and changes nothing — in particular a duplicate
        commit never advances the counter a second time, and never
        disturbs a *newer* reservation."""
        _require_token(token, "commit")
        # One acquisition covers the match and the mutation, so the
        # decision and its effect cannot be separated by another caller.
        with self._lock:
            self._match_under_lock(token, "commit")
            self._calls += 1
            self._active_serial = _NO_RESERVATION
            token._outcome = "committed"

    def _abandon_call(self, token):
        """Cancel the reserved call: clear the reservation and leave
        ``calls`` unchanged. Runs on every failure path.

        Only the currently active matching token is accepted; every other
        token raises ``RuntimeError`` and changes nothing, so a duplicate
        cancel can never release a *newer* reservation."""
        _require_token(token, "abandon")
        with self._lock:
            self._match_under_lock(token, "abandon")
            self._active_serial = _NO_RESERVATION
            token._outcome = "abandoned"

    def _match_under_lock(self, token, what):
        """The token/active-slot comparison itself. Caller holds the
        lock. Order matters: a foreign token is reported as foreign even
        when this generator happens to have a reservation outstanding."""
        if token._generator is not self:
            raise RuntimeError(
                f"cannot {what} a call: that reservation token belongs to "
                f"a different NativeGenerator"
            )
        if self._active_serial == _NO_RESERVATION:
            if token._outcome is not None:
                raise RuntimeError(
                    f"cannot {what} a call: that reservation was already "
                    f"{token._outcome}"
                )
            raise RuntimeError(
                f"cannot {what} a call: this generator has no outstanding "
                f"reservation"
            )
        if self._active_serial != token._serial:
            raise RuntimeError(
                f"cannot {what} a call: that reservation is stale "
                f"(already {token._outcome or 'superseded'}); a newer "
                f"reservation is outstanding and is left untouched"
            )

    def _call_committed(self, token):
        """Whether ``token``'s reserved call was **committed on this
        generator** — a read-only outcome query, and nothing more.

        It exists for exactly one caller and one question. The
        differentiable Dropout operation (§5, §8) commits as the last
        action before returning its result, and an exception can still
        arrive in the window between a *successful* commit and that
        return. Its cleanup must then behave differently: the call is
        irreversibly consumed, the reservation slot is already clear, and
        abandoning the token would raise "already committed" and mask the
        failure the caller actually needs to see. A local boolean set
        after ``_commit_call`` cannot answer this, because the commit can
        succeed and the assignment never run; the token's own recorded
        outcome can.

        Changes nothing: no reservation is created, cleared, or matched,
        ``calls`` does not move, no serial is consumed, and no claim is
        touched. A foreign generator's token answers ``False`` here rather
        than raising, because the question is "did *this* generator commit
        it", and a non-token is a caller bug and raises ``TypeError``
        exactly as commit and abandon do. Private, like the rest of the
        reservation protocol — a token's outcome is never public."""
        _require_token(token, "inspect")
        # Read under the lock, like every other state read: the commit
        # that writes this outcome holds the same lock while doing so.
        with self._lock:
            return (token._generator is self
                    and token._outcome == "committed")

    def _has_active_reservation(self):
        """Whether a reservation is **in flight** — published, or claimed
        and still under construction. Private, read-only, and used by
        tests and diagnostics; the multi-generator transaction gets the
        same guarantee from ``_require_no_reservation`` while it holds
        every target's lock."""
        with self._lock:
            return (self._active_serial != _NO_RESERVATION
                    or self._claim_serial != _NO_RESERVATION)

    # -- identity ------------------------------------------------------
    #
    # No __eq__ / __hash__ override: identity is object identity, like
    # every other native object. Copying is refused rather than left to
    # the default machinery, because a copy is exactly the "two places,
    # one stream" bug explicit state exists to prevent — and the default
    # copy/pickle path would produce one silently.

    def __copy__(self):
        raise TypeError(
            "NativeGenerator cannot be copied: two generators with the "
            "same seed and counter would produce the same values in two "
            "places. Share the object, or construct a new generator with "
            "a different seed."
        )

    def __deepcopy__(self, memo):
        raise TypeError(
            "NativeGenerator cannot be deep-copied: two generators with "
            "the same seed and counter would produce the same values in "
            "two places. Share the object, or construct a new generator "
            "with a different seed."
        )

    def __reduce_ex__(self, protocol):
        raise TypeError(
            "NativeGenerator cannot be pickled: unpickling would produce "
            "a second generator on the same stream. Persist its state() "
            "explicitly instead."
        )

    def __repr__(self):
        state = self.state()
        return (
            f"NativeGenerator(algorithm={state['algorithm']!r}, "
            f"algorithm_version={state['algorithm_version']}, "
            f"seed={state['seed']}, calls={state['calls']})"
        )


# ----------------------------------------------------------------------
# The multi-generator state transaction
# ----------------------------------------------------------------------
#
# The generator analogue of ``_native_state.replace_native_state``: it
# replaces several generators' states as one all-or-nothing operation.
# It lives here, beside the lock it has to reason about, rather than in
# native_module.py — a caller should never have to know how a generator
# is synchronized to load state into a model.
#
# The hard requirement is that **no reservation may begin on any target
# between the final reservation check and the end of the commit**. A
# per-generator check-then-release-then-write cannot provide that: a
# second thread can reserve in the gap, and the write would then move the
# seed out from under a live token's index. So the transaction holds
# *every* target's lock across the recheck, the snapshots, and the
# writes.
#
# Holding several locks at once is a deadlock risk, so the acquisition
# order is **global and independent of the caller**: sorted by ``id()``.
# Object identity is a property of the generator, not of the mapping,
# the module tree, or the canonical names — so two concurrent loads over
# overlapping sets, arriving through different modules and in different
# orders, still acquire in the same sequence and cannot form a cycle.
#
# That order holds for *every* entry into this function, including one
# reached from a finalizer, because no generator lock is held anywhere a
# finalizer can run: reservation tokens are constructed outside the lock
# (§3.6). A transaction started from a finalizer therefore begins owning
# nothing and takes the global order like any other caller — it cannot
# start from the middle of the order and reach backwards.


def _ordered_targets(generators):
    """The unique targets in the transaction's global lock order.

    Sorting on ``id()`` is deliberate. A user-visible order (canonical
    names, mapping order, registration order) differs between two modules
    that share generators, which is exactly the case that deadlocks;
    identity does not. The generators are all referenced for the whole
    transaction, so no id can be recycled underneath it."""
    return sorted(generators, key=id)


def replace_generator_states(entries):
    """Replace several ``NativeGenerator`` states as one transaction.

    ``entries`` is an iterable of ``GeneratorStateEntry(label, generator,
    state)``. Ordering follows validate → lock → recheck → snapshot →
    commit:

    1. Every entry's ``state`` is validated against its generator
       (structure, exact key set, algorithm and version equality, seed
       and counter type and range), with ``label`` naming it in errors.
       Targets are deduplicated by **identity**; the same generator
       supplied twice with different states is a conflict and is
       rejected, so an aliased key can never half-apply.
    2. Every unique target's lock is acquired in the global ``id()``
       order of ``_ordered_targets``, and released in reverse.
    3. **While every lock is held**, each target is rechecked for a
       reservation — published *or* under construction. This is the
       check that matters: it cannot be raced, because no target can
       start a reservation without the lock this transaction is holding.
    4. Each target's previous ``(seed, calls)`` is snapshotted.
    5. The writes run. They are ``__slots__`` integer assignments and
       **cannot fail**, so the only way out of the loop early is an
       asynchronous exception; the rollback restores from the snapshots
       using the same non-failing primitive and completes *before* any
       lock is released, so no other thread can ever observe a partial
       commit.

    Any failure leaves every generator's state, identity, and reservation
    exactly as they were, and no new reservation exists. A concurrent
    reservation either wins the lock first — and the load then rejects
    without mutating anything — or waits and observes the finished state.
    It can never overlap the commit."""
    entries = list(entries)

    # -- validate, and fold aliases together by identity.
    staged = {}          # id(generator) -> (generator, (seed, calls), label)
    for entry in entries:
        if not isinstance(entry.generator, NativeGenerator):
            raise TypeError(
                f"generator state for {entry.label!r}: expected a "
                f"NativeGenerator, got {type(entry.generator).__name__}"
            )
        try:
            seed, calls = entry.generator._validate_state(entry.state)
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"generator state for {entry.label!r}: {error}"
            ) from None
        previous = staged.get(id(entry.generator))
        if previous is not None and previous[1] != (seed, calls):
            raise ValueError(
                f"conflicting states supplied for one generator: "
                f"{previous[2]!r} and {entry.label!r} name the same object "
                f"but carry different states"
            )
        staged[id(entry.generator)] = (entry.generator, (seed, calls),
                                       entry.label)

    if not staged:
        return

    # A list, not a generator expression: ``_ordered_targets`` is a
    # documented seam, and handing it a one-shot iterator would make any
    # wrapper that looks at its input silently produce an empty
    # transaction instead of failing.
    targets = _ordered_targets(
        [generator for generator, _, _ in staged.values()]
    )
    labels = {id(generator): label for generator, _, label in staged.values()}

    with ExitStack() as stack:
        # -- lock every target, in the global order, before deciding
        # anything. Released in reverse by the ExitStack.
        for generator in targets:
            stack.enter_context(generator._lock)

        # -- recheck under the locks. Nothing can start a reservation on
        # a target from here to the end of the commit.
        for generator in targets:
            try:
                generator._require_no_reservation("load state into")
            except RuntimeError as error:
                raise RuntimeError(
                    f"cannot load generator state for "
                    f"{labels[id(generator)]!r}: {error}"
                ) from None

        # -- snapshot, then write. The writes cannot fail; the rollback
        # runs while the locks are still held.
        previous_states = [
            (generator, generator._snapshot_state()) for generator in targets
        ]
        committed = []
        try:
            for generator in targets:
                seed, calls = staged[id(generator)][1]
                generator._assign_state(seed, calls)
                committed.append(generator)
        except BaseException:
            restore = dict(previous_states)
            for generator in reversed(committed):
                generator._assign_state(*restore[generator])
            raise
