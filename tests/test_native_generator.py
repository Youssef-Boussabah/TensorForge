"""Tests for NativeGenerator and NativeModule generator registration
(Phase G, milestone G1; contract locked in
docs/native_rng_dropout_design.md §3-§5 and §9).

G1 ships explicit, inspectable, serializable random **state** and its
place in the module hierarchy. It generates **no random values**: there
is no finalizer, no bit derivation, no mask, no random kernel, and no
Dropout, and this file asserts that boundary as carefully as it asserts
the state contract.

What is under test:

- the four-field state (algorithm, algorithm version, seed, committed
  call counter), its exact-int validation, and its read-only properties;
- ``state()``/``load_state()``/``reseed()``/``reset()`` — atomic, so a
  rejected call leaves the generator bit-identical;
- identity semantics: no value equality, no copy, no deepcopy, no
  pickle, no ``close()``, and no native storage;
- the lock-protected, token-validated call transaction — one live
  reservation, commit advancing exactly once, cancellation never
  advancing, and stale/foreign/duplicate/finished tokens inert;
- concurrency: a second reservation fails deterministically without
  receiving an index, and no two callers ever get the same call index
  (proved with barriers and events, never with sleeps);
- generator registration as NativeModule's fourth category: assignment
  and explicit registration, category collisions, deletion, recursive
  identity-deduplicated cycle-safe traversal, and the separate
  ``generator_state_dict()`` surface;
- that ordinary ``state_dict()`` stays tensor-only and unchanged.

Tests that construct native tensors skip when the compiled backend is
not built; the pure-Python generator and hierarchy tests run everywhere.

Selector: python -m pytest -q -k "generator"
"""

import copy
import pickle
import secrets
import threading

import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeGenerator,
    NativeModule,
    NativeParameter,
    NativeTensor,
)
from tensorforge.experimental import native_generator as native_generator_module

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)

ALGORITHM = "tensorforge.splitmix64"
ALGORITHM_VERSION = 1
UINT64_MAX = 2**64 - 1

# A bounded join: every concurrency test must finish or fail loudly,
# never hang the suite.
JOIN_TIMEOUT = 10.0


def state_of(seed, calls):
    """A well-formed state mapping, for load_state tests."""
    return {
        "algorithm": ALGORITHM,
        "algorithm_version": ALGORITHM_VERSION,
        "seed": seed,
        "calls": calls,
    }


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    deterministic native-allocation instrumentation. A generator owns no
    native storage, so this set must never move around one."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


# ======================================================================
# Construction and the state contract
# ======================================================================


def test_explicit_seed_gives_deterministic_initial_state():
    generator = NativeGenerator(1234567890123456789)
    assert generator.seed == 1234567890123456789
    assert generator.calls == 0
    assert generator.algorithm == ALGORITHM
    assert generator.algorithm_version == ALGORITHM_VERSION


def test_two_generators_with_the_same_seed_start_identical():
    a = NativeGenerator(7)
    b = NativeGenerator(7)
    assert a.state() == b.state()


@pytest.mark.parametrize("seed", [0, 1, 2**63, UINT64_MAX])
def test_seed_boundaries_are_accepted_exactly(seed):
    assert NativeGenerator(seed).seed == seed


def test_seed_none_draws_a_valid_uint64_from_os_entropy():
    generator = NativeGenerator()
    assert type(generator.seed) is int
    assert 0 <= generator.seed <= UINT64_MAX
    assert generator.calls == 0


def test_seed_none_goes_through_secrets_and_nothing_else(monkeypatch):
    """The one entropy draw in Phase G is ``secrets.randbits(64)``.
    Patching it proves the path, and patching the *forbidden* sources to
    explode proves nothing else is consulted."""
    monkeypatch.setattr(secrets, "randbits", lambda bits: 0xABCDEF0123456789)

    import random as python_random
    import time

    import numpy as np

    def forbidden(*args, **kwargs):
        raise AssertionError("a forbidden entropy source was consulted")

    monkeypatch.setattr(python_random, "getrandbits", forbidden)
    monkeypatch.setattr(python_random, "random", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "randint", forbidden)
    monkeypatch.setattr(time, "time", forbidden)
    monkeypatch.setattr(time, "time_ns", forbidden)

    generator = NativeGenerator()
    assert generator.seed == 0xABCDEF0123456789
    # And an explicit seed draws nothing at all.
    monkeypatch.setattr(secrets, "randbits", forbidden)
    assert NativeGenerator(5).seed == 5


def test_two_unseeded_generators_differ():
    # Not a randomness test: 64 bits of OS entropy colliding twice would
    # mean the entropy source is broken, which is worth knowing.
    seeds = {NativeGenerator().seed for _ in range(8)}
    assert len(seeds) == 8


@pytest.mark.parametrize("bad", [True, False, 1.0, "7", b"7", object(), []])
def test_non_int_seed_is_a_type_error(bad):
    # ``None`` is deliberately absent: it is the "draw from OS entropy"
    # sentinel, covered by its own tests above.
    with pytest.raises(TypeError):
        NativeGenerator(bad)


def test_bool_is_not_a_seed():
    """``True`` is an int subclass but is not a seed: exact-type
    discipline, matching the checkpoint metadata validator."""
    with pytest.raises(TypeError):
        NativeGenerator(True)


def test_numpy_integer_scalar_is_not_a_seed():
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError):
        NativeGenerator(np.int64(7))


@pytest.mark.parametrize("bad", [-1, -(2**70), 2**64, 2**80])
def test_out_of_range_seed_is_a_value_error(bad):
    with pytest.raises(ValueError):
        NativeGenerator(bad)


def test_int_subclass_seed_is_rejected():
    class Sneaky(int):
        pass

    with pytest.raises(TypeError):
        NativeGenerator(Sneaky(3))


# ======================================================================
# Read-only properties and state inspection
# ======================================================================


@pytest.mark.parametrize(
    "name, value",
    [("seed", 9), ("calls", 3), ("algorithm", "x"), ("algorithm_version", 2)],
)
def test_state_properties_are_read_only(name, value):
    generator = NativeGenerator(11)
    with pytest.raises(AttributeError):
        setattr(generator, name, value)
    assert generator.state() == state_of(11, 0)


def test_no_attribute_can_be_injected_onto_a_generator():
    generator = NativeGenerator(11)
    with pytest.raises(AttributeError):
        generator.extra = 1


def test_state_returns_exactly_the_four_fields():
    generator = NativeGenerator(11)
    state = generator.state()
    assert list(state) == ["algorithm", "algorithm_version", "seed", "calls"]
    assert state == state_of(11, 0)
    assert type(state["seed"]) is int and type(state["calls"]) is int


def test_state_snapshots_are_independent_of_each_other_and_the_generator():
    generator = NativeGenerator(11)
    first = generator.state()
    first["seed"] = 999
    first["calls"] = 999
    assert generator.state() == state_of(11, 0)
    assert generator.seed == 11
    assert generator.calls == 0
    assert generator.state() is not first


def test_state_inspection_creates_no_reservation_and_advances_nothing():
    generator = NativeGenerator(11)
    for _ in range(5):
        generator.state()
        _ = generator.seed, generator.calls, generator.algorithm
    assert generator.calls == 0
    # A reservation is still available, so no phantom one was left behind.
    token = generator._reserve_call()
    generator._commit_call(token)
    assert generator.calls == 1


def test_repr_reports_state_without_leaking_internals():
    text = repr(NativeGenerator(11))
    assert "NativeGenerator(" in text
    assert ALGORITHM in text and "seed=11" in text and "calls=0" in text
    for leaked in ("lock", "token", "serial", "_active"):
        assert leaked not in text


# ======================================================================
# State replacement: load_state / reseed / reset
# ======================================================================


def test_load_state_replaces_seed_and_counter_in_place():
    generator = NativeGenerator(1)
    generator.load_state(state_of(42, 17))
    assert generator.seed == 42
    assert generator.calls == 17


def test_load_state_accepts_the_generators_own_snapshot():
    generator = NativeGenerator(3)
    token = generator._reserve_call()
    generator._commit_call(token)
    snapshot = generator.state()
    generator.reseed(99)
    generator.load_state(snapshot)
    assert generator.state() == snapshot


def test_load_state_preserves_object_identity():
    generator = NativeGenerator(1)
    module = NativeModule()
    module.g = generator
    generator.load_state(state_of(42, 17))
    assert module.g is generator


@pytest.mark.parametrize(
    "bad, error",
    [
        (None, TypeError),
        ([("seed", 1)], TypeError),
        ("state", TypeError),
        ({}, ValueError),
        ({"seed": 1, "calls": 0}, ValueError),
        ({"algorithm": ALGORITHM, "algorithm_version": 1, "seed": 1,
          "calls": 0, "extra": 1}, ValueError),
    ],
)
def test_load_state_rejects_malformed_mappings(bad, error):
    generator = NativeGenerator(5)
    with pytest.raises(error):
        generator.load_state(bad)
    assert generator.state() == state_of(5, 0)


def test_load_state_rejects_an_algorithm_mismatch():
    generator = NativeGenerator(5)
    bad = state_of(9, 9)
    bad["algorithm"] = "tensorforge.philox"
    with pytest.raises(ValueError, match="algorithm mismatch"):
        generator.load_state(bad)
    assert generator.state() == state_of(5, 0)


def test_load_state_rejects_a_non_string_algorithm():
    generator = NativeGenerator(5)
    bad = state_of(9, 9)
    bad["algorithm"] = 1
    with pytest.raises(TypeError):
        generator.load_state(bad)
    assert generator.state() == state_of(5, 0)


def test_load_state_rejects_an_algorithm_version_mismatch():
    generator = NativeGenerator(5)
    bad = state_of(9, 9)
    bad["algorithm_version"] = 2
    with pytest.raises(ValueError, match="algorithm version mismatch"):
        generator.load_state(bad)
    assert generator.state() == state_of(5, 0)


@pytest.mark.parametrize("field", ["seed", "calls"])
@pytest.mark.parametrize(
    "bad, error",
    [(True, TypeError), (1.0, TypeError), ("1", TypeError),
     (-1, ValueError), (2**64, ValueError)],
)
def test_load_state_validates_seed_and_counter_atomically(field, bad, error):
    generator = NativeGenerator(5)
    candidate = state_of(9, 9)
    candidate[field] = bad
    with pytest.raises(error):
        generator.load_state(candidate)
    # The *other* field was valid, so a non-atomic implementation would
    # have written it.
    assert generator.state() == state_of(5, 0)


@pytest.mark.parametrize("calls", [0, 1, UINT64_MAX])
def test_load_state_accepts_the_counter_boundaries(calls):
    generator = NativeGenerator(5)
    generator.load_state(state_of(5, calls))
    assert generator.calls == calls


def test_reseed_sets_the_seed_and_clears_the_counter():
    generator = NativeGenerator(1)
    generator._commit_call(generator._reserve_call())
    generator._commit_call(generator._reserve_call())
    assert generator.calls == 2
    generator.reseed(77)
    assert generator.state() == state_of(77, 0)


@pytest.mark.parametrize(
    "bad, error",
    [(True, TypeError), (1.5, TypeError), (None, TypeError),
     (-1, ValueError), (2**64, ValueError)],
)
def test_reseed_rejects_a_bad_seed_atomically(bad, error):
    generator = NativeGenerator(1)
    generator._commit_call(generator._reserve_call())
    with pytest.raises(error):
        generator.reseed(bad)
    assert generator.state() == state_of(1, 1)


def test_reset_keeps_the_seed_and_clears_the_counter():
    generator = NativeGenerator(1)
    generator._commit_call(generator._reserve_call())
    generator.reset()
    assert generator.state() == state_of(1, 0)


def test_reset_on_a_fresh_generator_is_a_no_op():
    generator = NativeGenerator(1)
    generator.reset()
    assert generator.state() == state_of(1, 0)


# ======================================================================
# Identity, copying, and the absent native lifecycle
# ======================================================================


def test_generators_use_identity_not_value_equality():
    a = NativeGenerator(7)
    b = NativeGenerator(7)
    assert a.state() == b.state()
    assert a != b
    assert a == a
    assert len({a, b}) == 2       # hashable by identity
    assert a is not b


def test_copy_is_refused():
    generator = NativeGenerator(7)
    with pytest.raises(TypeError):
        copy.copy(generator)
    assert generator.state() == state_of(7, 0)


def test_deepcopy_is_refused():
    generator = NativeGenerator(7)
    with pytest.raises(TypeError):
        copy.deepcopy(generator)
    with pytest.raises(TypeError):
        copy.deepcopy({"g": generator})
    assert generator.state() == state_of(7, 0)


def test_pickling_is_refused():
    with pytest.raises(TypeError):
        pickle.dumps(NativeGenerator(7))


@pytest.mark.parametrize(
    "absent",
    ["close", "closed", "clone", "split", "spawn", "jump", "advance",
     "manual_seed", "random", "rand", "randn", "bernoulli", "uniform",
     "dropout", "next", "__next__"],
)
def test_the_generator_has_no_lifecycle_or_numerical_surface(absent):
    assert not hasattr(NativeGenerator, absent), absent


def test_the_generator_module_exposes_only_state_names():
    """Milestone G1 generates no random values: the module carries the
    algorithm *identity* and nothing that derives bits from it."""
    for absent in ("mix64", "GOLDEN", "stream", "next_bits", "uniform"):
        assert not hasattr(native_generator_module, absent), absent
    assert native_generator_module.ALGORITHM == ALGORITHM
    assert native_generator_module.ALGORITHM_VERSION == ALGORITHM_VERSION
    assert native_generator_module.UINT64_MAX == UINT64_MAX


@needs_native
def test_generator_lifecycle_moves_no_native_storage(live_storages):
    baseline = len(live_storages)
    generators = [NativeGenerator(i) for i in range(4)]
    module = NativeModule()
    for index, generator in enumerate(generators):
        module.register_generator(f"g{index}", generator)
    for generator in generators:
        generator.state()
        generator._commit_call(generator._reserve_call())
        generator._abandon_call(generator._reserve_call())
        generator.reseed(99)
        generator.reset()
        generator.load_state(state_of(1, 1))
    module.generator_state_dict()
    list(module.named_generators())
    del module, generators
    assert len(live_storages) == baseline


# ======================================================================
# The call transaction: reserve / commit / abandon
# ======================================================================


def test_reserve_then_commit_advances_the_counter_exactly_once():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    assert generator.calls == 0            # not until commit
    generator._commit_call(token)
    assert generator.calls == 1


def test_reserve_then_abandon_leaves_the_counter_alone():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    generator._abandon_call(token)
    assert generator.calls == 0


def test_successive_calls_get_successive_indices():
    generator = NativeGenerator(1)
    indices = []
    for _ in range(5):
        token = generator._reserve_call()
        indices.append(token._index)
        generator._commit_call(token)
    assert indices == [0, 1, 2, 3, 4]
    assert generator.calls == 5


def test_an_abandoned_call_index_is_reused_by_the_next_reservation():
    """The counter counts *committed* calls, so abandoning leaves the
    index free — a failed forward consumes nothing at all."""
    generator = NativeGenerator(1)
    first = generator._reserve_call()
    generator._abandon_call(first)
    second = generator._reserve_call()
    assert second._index == first._index == 0
    generator._commit_call(second)
    assert generator.calls == 1


def test_a_second_reservation_is_refused_while_one_is_outstanding():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    with pytest.raises(RuntimeError, match="outstanding call reservation"):
        generator._reserve_call()
    # The first reservation is untouched and still usable.
    generator._commit_call(token)
    assert generator.calls == 1


def test_a_refused_reservation_consumes_nothing():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    for _ in range(3):
        with pytest.raises(RuntimeError):
            generator._reserve_call()
    assert generator.calls == 0
    generator._abandon_call(token)
    assert generator.calls == 0


def test_duplicate_commit_is_refused_and_never_advances_twice():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    generator._commit_call(token)
    with pytest.raises(RuntimeError, match="already committed"):
        generator._commit_call(token)
    assert generator.calls == 1


def test_duplicate_cancel_is_refused_and_never_advances():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    generator._abandon_call(token)
    with pytest.raises(RuntimeError, match="already abandoned"):
        generator._abandon_call(token)
    assert generator.calls == 0


def test_commit_after_cancel_is_refused():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    generator._abandon_call(token)
    with pytest.raises(RuntimeError, match="already abandoned"):
        generator._commit_call(token)
    assert generator.calls == 0


def test_cancel_after_commit_is_refused():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    generator._commit_call(token)
    with pytest.raises(RuntimeError, match="already committed"):
        generator._abandon_call(token)
    assert generator.calls == 1


def test_a_token_from_another_generator_is_refused_on_both():
    mine = NativeGenerator(1)
    theirs = NativeGenerator(2)
    foreign = theirs._reserve_call()
    mine_token = mine._reserve_call()
    with pytest.raises(RuntimeError, match="different NativeGenerator"):
        mine._commit_call(foreign)
    with pytest.raises(RuntimeError, match="different NativeGenerator"):
        mine._abandon_call(foreign)
    # Neither generator moved, and both reservations are still live.
    assert mine.calls == 0 and theirs.calls == 0
    mine._commit_call(mine_token)
    theirs._commit_call(foreign)
    assert mine.calls == 1 and theirs.calls == 1


def test_a_foreign_token_is_refused_even_when_nothing_is_outstanding():
    mine = NativeGenerator(1)
    theirs = NativeGenerator(2)
    foreign = theirs._reserve_call()
    with pytest.raises(RuntimeError, match="different NativeGenerator"):
        mine._commit_call(foreign)
    assert mine.calls == 0 and theirs.calls == 0


def test_a_stale_token_never_disturbs_the_newer_reservation():
    generator = NativeGenerator(1)
    stale = generator._reserve_call()
    generator._abandon_call(stale)
    fresh = generator._reserve_call()
    with pytest.raises(RuntimeError):
        generator._commit_call(stale)
    with pytest.raises(RuntimeError):
        generator._abandon_call(stale)
    assert generator.calls == 0
    # The live reservation survived both attempts.
    generator._commit_call(fresh)
    assert generator.calls == 1


def test_commit_or_cancel_without_a_reservation_is_refused():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    generator._commit_call(token)
    other = NativeGenerator(1)
    spare = other._reserve_call()
    other._abandon_call(spare)
    with pytest.raises(RuntimeError):
        generator._commit_call(spare)
    assert generator.calls == 1


@pytest.mark.parametrize("bad", [None, 0, "token", 1.5, object(), (0, 0)])
def test_a_malformed_token_is_a_type_error(bad):
    generator = NativeGenerator(1)
    live = generator._reserve_call()
    with pytest.raises(TypeError):
        generator._commit_call(bad)
    with pytest.raises(TypeError):
        generator._abandon_call(bad)
    assert generator.calls == 0
    generator._commit_call(live)
    assert generator.calls == 1


def test_serials_are_never_reused_within_a_generator():
    generator = NativeGenerator(1)
    serials = []
    for _ in range(6):
        token = generator._reserve_call()
        serials.append(token._serial)
        generator._abandon_call(token)
    assert len(set(serials)) == len(serials)
    assert serials == sorted(serials)


def test_the_reservation_token_type_is_not_public():
    import tensorforge.experimental as experimental

    assert "_ReservationToken" not in experimental.__all__
    assert not hasattr(experimental, "_ReservationToken")
    assert not hasattr(experimental, "ReservationToken")


def test_a_token_exposes_no_mutating_behavior():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    public = [name for name in dir(token) if not name.startswith("_")]
    assert public == []
    with pytest.raises(AttributeError):
        token.anything = 1
    generator._abandon_call(token)


# ======================================================================
# Reservation creation: claim / construct / publish
# ======================================================================
#
# Creating a reservation is a two-phase transaction whose middle step —
# minting the token — runs with **no generator lock held**. Constructing
# an object can run interpreter finalization, and a finalizer may start a
# multi-generator transaction (§9.6); if this thread owned a generator
# lock at that moment it could acquire a second one out of the global
# order and deadlock against a thread doing the reverse. Owning nothing
# during construction removes that possibility entirely, and the
# construction *claim* is what keeps the window safe in the meantime.


def _lock_is_held(generator, timeout=JOIN_TIMEOUT):
    """Whether ``generator``'s lock is currently owned by any thread.

    Two independent observations, because neither alone is sufficient:
    ``_is_owned()`` answers for the calling thread (the case that would
    deadlock a reentrant finalizer on a plain ``Lock``), and a bounded
    acquisition from a *separate* thread answers for every other thread.
    Bounded, so a held lock fails a test rather than hanging it."""
    if generator._lock._is_owned():
        return True
    acquired = []

    def probe():
        if generator._lock.acquire(timeout=1.0):
            acquired.append(True)
            generator._lock.release()

    thread = threading.Thread(target=probe, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "the lock probe never returned"
    return not acquired


def test_the_token_is_constructed_with_no_generator_lock_held(monkeypatch):
    """The invariant the two-phase protocol exists for: no callback-capable
    operation runs while a generator lock is held.

    Token construction is the one allocation in the reservation path, and
    allocation can run a finalizer. Observed from inside the constructor
    itself, both by ownership and by an independent thread acquiring the
    lock outright."""
    generator = NativeGenerator(5)
    observed = {}
    real = native_generator_module._ReservationToken

    def observing_constructor(gen, serial, index):
        observed["owned_by_me"] = gen._lock._is_owned()
        observed["held_at_all"] = _lock_is_held(gen)
        return real(gen, serial, index)

    monkeypatch.setattr(
        native_generator_module, "_ReservationToken", observing_constructor
    )
    token = generator._reserve_call()
    monkeypatch.undo()

    assert observed["owned_by_me"] is False
    assert observed["held_at_all"] is False, (
        "the generator lock was held while the token was constructed"
    )
    generator._commit_call(token)
    assert generator.calls == 1


def test_the_claim_is_published_but_the_reservation_is_not(monkeypatch):
    """Phase 1 publishes *only* the construction claim: no active
    reservation, no counter movement, no serial movement. Observed from
    inside the construction window."""
    generator = NativeGenerator(5)
    seen = {}
    real = native_generator_module._ReservationToken

    def observing_constructor(gen, serial, index):
        seen["claim"] = (gen._claim_serial, gen._claim_index)
        seen["active"] = gen._active_serial
        seen["calls"] = gen._calls
        seen["next_serial"] = gen._next_serial
        seen["in_flight"] = gen._has_active_reservation()
        return real(gen, serial, index)

    monkeypatch.setattr(
        native_generator_module, "_ReservationToken", observing_constructor
    )
    token = generator._reserve_call()
    monkeypatch.undo()

    assert seen["claim"] == (token._serial, token._index)
    assert seen["active"] == 0                 # nothing published yet
    assert seen["calls"] == 0                  # the counter has not moved
    assert seen["next_serial"] == token._serial  # nor has the serial
    assert seen["in_flight"] is True           # but the window is visible

    # Phase 3 published the reservation and cleared the claim.
    assert generator._claim_serial == 0
    assert generator._active_serial == token._serial
    assert generator._next_serial == token._serial + 1
    generator._abandon_call(token)


def test_a_failed_token_construction_leaves_no_active_reservation(monkeypatch):
    """The token is built **before** any active-reservation state is
    published. If construction raises, the generator must not be left
    holding a reservation nobody has a token for — that would be
    permanently uncommittable and uncancellable."""
    generator = NativeGenerator(5)
    before = generator.state()

    boom = RuntimeError("token allocation failed")

    def exploding_token(*args, **kwargs):
        raise boom

    monkeypatch.setattr(
        native_generator_module, "_ReservationToken", exploding_token
    )
    with pytest.raises(RuntimeError) as caught:
        generator._reserve_call()
    assert caught.value is boom                       # propagates unchanged

    # Nothing was published, and the claim was released: no reservation,
    # no counter movement, and no lock left held.
    assert not generator._has_active_reservation()
    assert generator._claim_serial == 0
    assert generator._active_serial == 0
    assert not _lock_is_held(generator)
    assert generator.state() == before

    # State replacement is usable again immediately — proof that no
    # reservation is stranded (a stranded one would refuse all three).
    generator.reset()
    generator.reseed(9)
    generator.load_state(state_of(5, 0))
    assert generator.state() == before

    # And a later reserve/commit succeeds normally.
    monkeypatch.undo()
    token = generator._reserve_call()
    assert token._index == 0
    generator._commit_call(token)
    assert generator.calls == 1


@pytest.mark.parametrize(
    "failure",
    [MemoryError("no room"), KeyboardInterrupt(), RuntimeError("boom")],
    ids=["MemoryError", "KeyboardInterrupt", "RuntimeError"],
)
def test_any_construction_failure_releases_the_claim(failure, monkeypatch):
    """Cleanup is a ``finally``, so it covers ``BaseException`` too — a
    ``KeyboardInterrupt`` landing inside construction must not wedge the
    generator into a permanent claim."""
    generator = NativeGenerator(5)
    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        lambda *a, **k: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(type(failure)):
        generator._reserve_call()
    monkeypatch.undo()

    assert generator._claim_serial == 0
    assert not generator._has_active_reservation()
    assert generator.state() == state_of(5, 0)
    # Both a state replacement and a fresh reservation work immediately.
    generator.reseed(6)
    token = generator._reserve_call()
    generator._commit_call(token)
    assert generator.state() == state_of(6, 1)


def test_a_failed_token_construction_skips_no_reservation_serial(monkeypatch):
    """The never-reused serial advances only in phase 3, after the token
    exists, so a failed construction consumes no serial either."""
    generator = NativeGenerator(5)
    first = generator._reserve_call()
    generator._abandon_call(first)

    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        lambda *a, **k: (_ for _ in ()).throw(MemoryError("no room")),
    )
    with pytest.raises(MemoryError):
        generator._reserve_call()
    monkeypatch.undo()

    second = generator._reserve_call()
    assert second._serial == first._serial + 1
    generator._abandon_call(second)


def _reentrant_constructor(action, record):
    """A token constructor that runs ``action`` — a reentrant call back
    into the generator — before building the real token.

    This is how the reentrancy tests reach the exact window the claim
    guards: constructing a token can run arbitrary interpreter machinery
    (a finalizer), and that machinery may touch the same generator. Doing
    it explicitly is deterministic, where waiting for the garbage
    collector would not be."""
    real = native_generator_module._ReservationToken

    def constructor(generator, serial, index):
        try:
            action(generator)
            record.append(None)
        except BaseException as error:
            record.append(error)
        return real(generator, serial, index)

    return constructor


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda g: g._reserve_call(), id="reserve"),
        pytest.param(lambda g: g.reseed(99), id="reseed"),
        pytest.param(lambda g: g.reset(), id="reset"),
        pytest.param(lambda g: g.load_state(state_of(42, 7)), id="load_state"),
    ],
)
def test_reentering_during_token_construction_fails_instead_of_hanging(
    operation, monkeypatch
):
    """The window the construction claim exists for.

    The constructor runs holding no lock, so a reentrant call reaches the
    generator normally — and the claim then rejects it deterministically,
    mutating nothing. Nothing here may block."""
    generator = NativeGenerator(5)
    record = []
    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        _reentrant_constructor(operation, record),
    )

    token = generator._reserve_call()          # must not hang
    monkeypatch.undo()

    assert len(record) == 1
    assert isinstance(record[0], RuntimeError), (
        f"the reentrant call did not fail: {record[0]!r}"
    )
    # The outer reservation is intact and unaffected by the failed
    # reentrant call, and the state it was reserved against is unchanged.
    assert generator.seed == 5
    assert generator.calls == 0
    assert token._index == 0
    generator._commit_call(token)
    assert generator.state() == state_of(5, 1)


def test_state_inspection_during_token_construction_still_works(monkeypatch):
    """Only *replacement* is refused mid-construction. Reading follows
    the locked §3.6 behavior and keeps working."""
    generator = NativeGenerator(5)
    seen = []
    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        _reentrant_constructor(
            lambda g: seen.append((g.state(), g.seed, g.calls,
                                   g.algorithm, g.algorithm_version)),
            [],
        ),
    )
    token = generator._reserve_call()
    monkeypatch.undo()
    assert seen == [(state_of(5, 0), 5, 0, ALGORITHM, ALGORITHM_VERSION)]
    generator._abandon_call(token)


def test_a_reentrant_module_state_load_during_construction_is_refused(
    monkeypatch
):
    """The multi-generator transaction is a state replacement too, so it
    is refused mid-construction — and refusing writes nothing."""
    module = NativeModule()
    generator = NativeGenerator(5)
    module.gen = generator
    record = []
    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        _reentrant_constructor(
            lambda g: module.load_generator_state_dict({"gen": state_of(9, 9)}),
            record,
        ),
    )
    token = generator._reserve_call()
    monkeypatch.undo()

    assert isinstance(record[0], RuntimeError)
    assert generator.state() == state_of(5, 0)
    generator._abandon_call(token)


def test_a_replacement_naming_the_claimed_generator_is_refused(monkeypatch):
    """``replace_generator_states`` reached from inside construction: any
    transaction that names the claimed generator is refused, and refusing
    writes nothing — not even to the *other* targets it named."""
    claimed, other = NativeGenerator(5), NativeGenerator(6)
    record = []

    def replacement(_):
        native_generator_module.replace_generator_states([
            native_generator_module.GeneratorStateEntry(
                "other", other, state_of(60, 6)),
            native_generator_module.GeneratorStateEntry(
                "claimed", claimed, state_of(50, 5)),
        ])

    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        _reentrant_constructor(replacement, record),
    )
    token = claimed._reserve_call()
    monkeypatch.undo()

    assert isinstance(record[0], RuntimeError)
    assert "being constructed" in str(record[0])
    # Atomic: the co-target was named first and still was not written.
    assert claimed.state() == state_of(5, 0)
    assert other.state() == state_of(6, 0)
    claimed._commit_call(token)
    assert claimed.calls == 1


def test_a_replacement_of_other_generators_during_construction_succeeds(
    monkeypatch
):
    """The claim blocks only its own generator. A transaction over
    unrelated generators, reached from inside construction, completes
    normally — it never needed the claimed generator's lock, and the
    constructor holds none."""
    claimed = NativeGenerator(5)
    first, second = NativeGenerator(1), NativeGenerator(2)
    record = []

    def replacement(_):
        native_generator_module.replace_generator_states([
            native_generator_module.GeneratorStateEntry(
                "first", first, state_of(10, 1)),
            native_generator_module.GeneratorStateEntry(
                "second", second, state_of(20, 2)),
        ])

    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        _reentrant_constructor(replacement, record),
    )
    token = claimed._reserve_call()
    monkeypatch.undo()

    assert record == [None], f"the replacement failed: {record[0]!r}"
    assert first.state() == state_of(10, 1)
    assert second.state() == state_of(20, 2)
    assert claimed.state() == state_of(5, 0)
    claimed._commit_call(token)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_a_replacement_during_construction_ignores_caller_order(
    reverse, monkeypatch
):
    """Reverse the caller's mapping order and nothing changes: the same
    refusal, the same untouched state, and no hang either way — the
    acquisition order is the transaction's, not the caller's."""
    module = NativeModule()
    claimed, other = NativeGenerator(5), NativeGenerator(6)
    module.claimed, module.other = claimed, other
    target = {"claimed": state_of(50, 5), "other": state_of(60, 6)}
    if reverse:
        target = dict(reversed(list(target.items())))
    record = []

    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        _reentrant_constructor(
            lambda _: module.load_generator_state_dict(target), record
        ),
    )
    token = claimed._reserve_call()
    monkeypatch.undo()

    assert isinstance(record[0], RuntimeError)
    assert module.generator_state_dict() == {
        "claimed": state_of(5, 0), "other": state_of(6, 0),
    }
    claimed._abandon_call(token)


def test_a_failed_construction_clears_the_claim(monkeypatch):
    """The claim is released in ``finally``, so a failure does not wedge
    the generator into a permanent "constructing" state."""
    generator = NativeGenerator(5)
    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    for _ in range(3):
        with pytest.raises(RuntimeError, match="boom"):
            generator._reserve_call()
    monkeypatch.undo()

    assert generator._claim_serial == 0
    assert generator._claim_index == 0
    assert not generator._has_active_reservation()
    # Every state operation is available again, and so is reserving.
    generator.reseed(7)
    generator.reset()
    generator.load_state(state_of(5, 0))
    token = generator._reserve_call()
    generator._commit_call(token)
    assert generator.state() == state_of(5, 1)


def test_a_second_thread_meeting_construction_gets_no_duplicate_index():
    """Another thread arriving during construction meets the *claim*, not
    a half-built window: it is refused without receiving an index, and the
    claimed index is still the first caller's. Bounded, and no lock is
    held during construction, so the second caller never blocks on one."""
    generator = NativeGenerator(5)
    constructing = threading.Event()
    refused = threading.Event()
    outcome = {}

    def second_caller():
        assert constructing.wait(JOIN_TIMEOUT)
        try:
            outcome["token"] = generator._reserve_call()
        except RuntimeError as error:
            outcome["error"] = error
        finally:
            refused.set()

    thread = threading.Thread(target=second_caller, daemon=True)
    real = native_generator_module._ReservationToken

    def signalling_constructor(gen, serial, index):
        constructing.set()
        # Hold the window open until the second caller has been answered,
        # so its attempt provably lands inside construction.
        assert refused.wait(JOIN_TIMEOUT)
        return real(gen, serial, index)

    native_generator_module._ReservationToken = signalling_constructor
    try:
        thread.start()
        token = generator._reserve_call()
    finally:
        native_generator_module._ReservationToken = real

    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive(), "the second caller deadlocked"
    assert "token" not in outcome, "two callers received a reservation"
    assert isinstance(outcome["error"], RuntimeError)
    assert "already constructing" in str(outcome["error"])
    assert token._index == 0
    assert generator.calls == 0
    generator._commit_call(token)
    assert generator.calls == 1


def test_the_reservation_lock_is_reentrant_and_nothing_internal_is_public():
    generator = NativeGenerator(5)
    assert isinstance(generator._lock, type(threading.RLock()))
    # Neither the claim nor the lock is reachable as a public name.
    public = [name for name in dir(generator) if not name.startswith("_")]
    assert "lock" not in public and "claim" not in public
    for absent in ("claim_serial", "claim_index", "lock", "active_serial"):
        assert not hasattr(generator, absent), absent


def test_no_generator_lock_is_held_outside_a_generator_call():
    """The steady state: between operations the lock is free, so nothing
    a finalizer does can find it held."""
    generator = NativeGenerator(5)
    assert not _lock_is_held(generator)
    token = generator._reserve_call()
    assert not _lock_is_held(generator)     # a live reservation is not a lock
    generator._commit_call(token)
    assert not _lock_is_held(generator)


@needs_native
def test_a_failed_token_construction_moves_no_native_storage(
    live_storages, monkeypatch
):
    generator = NativeGenerator(5)
    baseline = len(live_storages)
    monkeypatch.setattr(
        native_generator_module, "_ReservationToken",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    for _ in range(3):
        with pytest.raises(RuntimeError):
            generator._reserve_call()
    assert len(live_storages) == baseline
    assert generator.calls == 0


# ======================================================================
# Publication succeeded, delivery did not
# ======================================================================
#
# The fourth failure position, and the one clearing the claim cannot
# reach: once phase 3 publishes, the claim is gone. An asynchronous
# exception arriving before the caller receives its token would leave an
# active reservation whose only token is being dropped — no caller could
# commit or abandon it, and every later reservation would be refused. So
# a failed delivery gets its own exact-match cleanup.
#
# ``_deliver_reservation`` is the private seam that makes this window
# addressable at all; tests raise from it rather than trusting real
# signal timing.


class _TestFailure(BaseException):
    """A BaseException that is neither an Exception nor one of the
    interpreter's own, so nothing can be catching it incidentally."""


def _fail_delivery(failure, after=None):
    """A ``_deliver_reservation`` replacement that optionally runs
    ``after(token)`` — to set up an exact-match scenario — and then
    raises ``failure`` in the publication-to-return window.

    It disarms itself for the duration of ``after`` so a hook that takes
    a *fresh* reservation gets the real seam, and re-arms in ``finally``
    so repeated-failure tests keep failing."""
    real = native_generator_module._deliver_reservation
    armed = [True]

    def deliver(token):
        if not armed[0]:
            return real(token)
        armed[0] = False
        try:
            if after is not None:
                after(token)
            raise failure
        finally:
            armed[0] = True

    return deliver


@pytest.mark.parametrize(
    "failure",
    [KeyboardInterrupt(), MemoryError("no room"), _TestFailure("injected")],
    ids=["KeyboardInterrupt", "MemoryError", "BaseException"],
)
def test_a_failed_delivery_leaves_no_reservation_and_no_claim(
    failure, monkeypatch
):
    """The whole contract for a failed delivery, for each representative
    ``BaseException``: it propagates, nothing is left behind, and the
    generator is immediately reusable."""
    generator = NativeGenerator(5)
    before = generator.state()
    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(failure),
    )
    with pytest.raises(type(failure)) as caught:
        generator._reserve_call()
    assert caught.value is failure               # propagates unchanged
    monkeypatch.undo()

    # Nothing stranded: no published reservation, no claim, no counter
    # movement, and no lock left held.
    assert generator._active_serial == 0
    assert generator._claim_serial == 0
    assert not generator._has_active_reservation()
    assert not _lock_is_held(generator)
    assert generator.state() == before

    # State replacement works — a stranded reservation would refuse all
    # three — and so does a later reserve/commit.
    generator.reset()
    generator.reseed(9)
    generator.load_state(state_of(5, 0))
    token = generator._reserve_call()
    assert token._index == 0
    generator._commit_call(token)
    assert generator.state() == state_of(5, 1)


def test_a_failed_delivery_consumes_no_call_index(monkeypatch):
    """A failed delivery may burn an opaque *serial* — those are never
    reused — but never a call index, and never a committed call."""
    generator = NativeGenerator(5)
    first = generator._reserve_call()
    generator._commit_call(first)

    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(_TestFailure("injected")),
    )
    with pytest.raises(_TestFailure):
        generator._reserve_call()
    monkeypatch.undo()

    assert generator.calls == 1                  # unchanged by the failure
    retry = generator._reserve_call()
    assert retry._index == 1                     # the same unconsumed index
    assert retry._serial > first._serial + 1     # the serial was consumed
    generator._commit_call(retry)
    assert generator.calls == 2


def test_repeated_failed_deliveries_never_strand_the_generator(monkeypatch):
    generator = NativeGenerator(5)
    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(KeyboardInterrupt()),
    )
    for _ in range(5):
        with pytest.raises(KeyboardInterrupt):
            generator._reserve_call()
        assert not generator._has_active_reservation()
        assert generator.calls == 0
    monkeypatch.undo()

    token = generator._reserve_call()
    assert token._index == 0
    generator._commit_call(token)
    assert generator.state() == state_of(5, 1)


@needs_native
def test_failed_deliveries_move_no_native_storage(live_storages, monkeypatch):
    generator = NativeGenerator(5)
    baseline = len(live_storages)
    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(MemoryError("no room")),
    )
    for _ in range(3):
        with pytest.raises(MemoryError):
            generator._reserve_call()
    assert len(live_storages) == baseline
    assert generator.calls == 0
    assert not generator._has_active_reservation()


# -- the cleanup matches exactly, or does nothing --------------------


def test_failed_delivery_cleanup_cannot_cancel_a_newer_reservation(
    monkeypatch
):
    """The scenario the exact match exists for, driven through the real
    seam: the window releases its own token and takes a *new* reservation
    before failing. The cleanup must leave that newer one alone."""
    generator = NativeGenerator(5)
    later = {}

    def take_a_newer_reservation(token):
        generator._abandon_call(token)           # release the published one
        later["token"] = generator._reserve_call()

    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(_TestFailure("injected"), after=take_a_newer_reservation),
    )
    with pytest.raises(_TestFailure):
        generator._reserve_call()
    monkeypatch.undo()

    # The newer reservation survived untouched and still commits.
    newer = later["token"]
    assert generator._active_serial == newer._serial
    assert newer._outcome is None
    generator._commit_call(newer)
    assert generator.calls == 1


def test_failed_delivery_cleanup_cannot_undo_a_commit(monkeypatch):
    """If the window committed the call before failing, the commit
    stands: the token is finished, so the cleanup matches nothing."""
    generator = NativeGenerator(5)
    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(_TestFailure("injected"),
                       after=lambda token: generator._commit_call(token)),
    )
    with pytest.raises(_TestFailure):
        generator._reserve_call()
    monkeypatch.undo()

    assert generator.calls == 1                  # the commit was not undone
    assert not generator._has_active_reservation()


def test_failed_delivery_cleanup_is_inert_on_an_already_abandoned_token(
    monkeypatch
):
    generator = NativeGenerator(5)
    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(_TestFailure("injected"),
                       after=lambda token: generator._abandon_call(token)),
    )
    with pytest.raises(_TestFailure):
        generator._reserve_call()
    monkeypatch.undo()

    assert generator.calls == 0
    assert not generator._has_active_reservation()
    token = generator._reserve_call()
    assert token._index == 0
    generator._commit_call(token)
    assert generator.calls == 1


def test_failed_delivery_cleanup_ignores_a_foreign_generator():
    """Called directly with another generator's token: the owner's live
    reservation is untouched, and so is this generator's."""
    owner, other = NativeGenerator(1), NativeGenerator(2)
    owned = owner._reserve_call()
    strangers = other._reserve_call()

    other._release_undelivered(owned)            # foreign token
    assert owner._active_serial == owned._serial
    assert other._active_serial == strangers._serial
    assert owned._outcome is None and strangers._outcome is None

    owner._commit_call(owned)
    other._commit_call(strangers)
    assert owner.calls == 1 and other.calls == 1


def test_failed_delivery_cleanup_ignores_a_stale_serial():
    """A token whose reservation has been superseded cannot cancel the
    reservation that replaced it."""
    generator = NativeGenerator(5)
    stale = generator._reserve_call()
    generator._abandon_call(stale)
    current = generator._reserve_call()

    generator._release_undelivered(stale)
    assert generator._active_serial == current._serial
    assert current._outcome is None
    generator._commit_call(current)
    assert generator.calls == 1


def test_failed_delivery_cleanup_ignores_a_mismatched_index():
    """Serial *and* index must both match: a token carrying the live
    serial but a different index is not this reservation."""
    generator = NativeGenerator(5)
    token = generator._reserve_call()
    forged = native_generator_module._ReservationToken(
        generator, token._serial, token._index + 1
    )
    generator._release_undelivered(forged)
    assert generator._active_serial == token._serial
    generator._commit_call(token)
    assert generator.calls == 1


def test_failed_delivery_cleanup_does_nothing_when_nothing_is_live():
    generator = NativeGenerator(5)
    token = generator._reserve_call()
    generator._commit_call(token)
    before = generator.state()

    generator._release_undelivered(token)        # already committed
    assert generator.state() == before
    assert token._outcome == "committed"
    assert not generator._has_active_reservation()


def test_a_discarded_token_is_refused_by_commit_and_abandon(monkeypatch):
    """After cleanup the token is finished, so a caller who somehow still
    holds it cannot resurrect the reservation."""
    generator = NativeGenerator(5)
    held = {}
    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(_TestFailure("injected"),
                       after=lambda token: held.update(token=token)),
    )
    with pytest.raises(_TestFailure):
        generator._reserve_call()
    monkeypatch.undo()

    token = held["token"]
    assert token._outcome == "discarded before delivery"
    for release in (generator._commit_call, generator._abandon_call):
        with pytest.raises(RuntimeError, match="already discarded"):
            release(token)
    assert generator.calls == 0


def test_the_delivery_seam_is_private_and_does_nothing_observable():
    import tensorforge.experimental as experimental

    assert not hasattr(experimental, "_deliver_reservation")
    assert "_deliver_reservation" not in experimental.__all__
    generator = NativeGenerator(5)
    token = generator._reserve_call()
    # The real seam is a pass-through; it changes no state at all.
    before = generator.state()
    assert native_generator_module._deliver_reservation(token) is token
    assert generator.state() == before
    assert generator._active_serial == token._serial
    generator._abandon_call(token)


# ======================================================================
# Counter exhaustion — the exact uint64 boundary
# ======================================================================
#
# ``calls`` is a *count of committed calls*, not an index space, so
# UINT64_MAX is a reachable, valid value rather than a sentinel: it is
# what the counter holds after the last representable successful call.
# These tests pin all three boundary states exactly, with no loops.


def test_calls_may_hold_the_full_uint64_range():
    generator = NativeGenerator(1)
    for calls in (0, 1, UINT64_MAX - 2, UINT64_MAX - 1, UINT64_MAX):
        generator.load_state(state_of(1, calls))
        assert generator.calls == calls


def test_a_reservation_uses_the_current_calls_value_as_its_index():
    generator = NativeGenerator(1)
    for calls in (0, 7, UINT64_MAX - 2, UINT64_MAX - 1):
        generator.load_state(state_of(1, calls))
        token = generator._reserve_call()
        assert token._index == calls
        generator._abandon_call(token)


def test_the_last_representable_call_succeeds_and_reaches_uint64_max():
    """At ``calls == UINT64_MAX - 1``: reservation succeeds, the index is
    ``UINT64_MAX - 1``, and committing advances to ``UINT64_MAX``."""
    generator = NativeGenerator(1)
    generator.load_state(state_of(1, UINT64_MAX - 1))
    token = generator._reserve_call()
    assert token._index == UINT64_MAX - 1
    generator._commit_call(token)
    assert generator.calls == UINT64_MAX


def test_a_failed_delivery_at_the_last_index_leaves_it_unconsumed(
    monkeypatch
):
    """The boundary crossed with the delivery failure: at
    ``calls == UINT64_MAX - 1`` a failed delivery must leave the counter
    exactly there, so the very last representable call is still
    available and a later commit still reaches ``UINT64_MAX``."""
    generator = NativeGenerator(1)
    generator.load_state(state_of(1, UINT64_MAX - 1))
    monkeypatch.setattr(
        native_generator_module, "_deliver_reservation",
        _fail_delivery(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        generator._reserve_call()
    monkeypatch.undo()

    assert generator.calls == UINT64_MAX - 1     # the index is unconsumed
    assert not generator._has_active_reservation()

    # The same last index is retryable, and committing it reaches the top.
    token = generator._reserve_call()
    assert token._index == UINT64_MAX - 1
    generator._commit_call(token)
    assert generator.calls == UINT64_MAX
    # And the counter is now genuinely exhausted, not wrapped.
    with pytest.raises(RuntimeError, match="exhausted"):
        generator._reserve_call()
    assert generator.calls == UINT64_MAX


def test_an_exhausted_counter_refuses_to_reserve_and_never_wraps():
    """At ``calls == UINT64_MAX``: reservation fails before producing a
    token, and the counter stays put rather than wrapping to zero."""
    generator = NativeGenerator(1)
    generator.load_state(state_of(1, UINT64_MAX))
    with pytest.raises(RuntimeError, match="exhausted"):
        generator._reserve_call()
    assert generator.calls == UINT64_MAX      # not 0, not UINT64_MAX + 1
    assert not generator._has_active_reservation()
    # Repeated attempts stay refused and still move nothing.
    for _ in range(3):
        with pytest.raises(RuntimeError):
            generator._reserve_call()
    assert generator.calls == UINT64_MAX


def test_cancelling_the_last_reservation_leaves_the_index_unconsumed():
    """Abandoning at ``UINT64_MAX - 1`` leaves the counter there, and the
    next reservation legitimately takes that same index again."""
    generator = NativeGenerator(1)
    generator.load_state(state_of(1, UINT64_MAX - 1))
    first = generator._reserve_call()
    generator._abandon_call(first)
    assert generator.calls == UINT64_MAX - 1

    second = generator._reserve_call()
    assert second._index == UINT64_MAX - 1
    assert second._serial != first._serial
    generator._commit_call(second)
    assert generator.calls == UINT64_MAX


def test_failed_operations_never_move_the_boundary_state():
    """Every rejected operation — stale, foreign, duplicate, malformed,
    conflicting — leaves the counter exactly at the boundary."""
    generator = NativeGenerator(1)
    other = NativeGenerator(2)
    generator.load_state(state_of(1, UINT64_MAX - 1))

    token = generator._reserve_call()
    foreign = other._reserve_call()
    for bad in (foreign, None, 0, "token"):
        with pytest.raises((RuntimeError, TypeError)):
            generator._commit_call(bad)
        with pytest.raises((RuntimeError, TypeError)):
            generator._abandon_call(bad)
    with pytest.raises(RuntimeError):
        generator._reserve_call()             # conflicting reservation
    with pytest.raises(RuntimeError):
        generator.reseed(3)                   # refused mid-reservation
    assert generator.calls == UINT64_MAX - 1

    generator._commit_call(token)
    assert generator.calls == UINT64_MAX
    # A duplicate commit at the boundary must not push it past the max.
    with pytest.raises(RuntimeError):
        generator._commit_call(token)
    assert generator.calls == UINT64_MAX
    other._abandon_call(foreign)


def test_an_exhausted_generator_recovers_through_reset_or_reseed():
    generator = NativeGenerator(1)
    generator.load_state(state_of(1, UINT64_MAX))
    with pytest.raises(RuntimeError):
        generator._reserve_call()
    generator.reset()
    generator._commit_call(generator._reserve_call())
    assert generator.calls == 1

    generator.load_state(state_of(1, UINT64_MAX))
    generator.reseed(42)
    assert generator.state() == state_of(42, 0)
    generator._commit_call(generator._reserve_call())
    assert generator.calls == 1


# ======================================================================
# State changes are refused during a live reservation
# ======================================================================


def test_load_state_is_refused_during_a_reservation():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    with pytest.raises(RuntimeError, match="reservation is outstanding"):
        generator.load_state(state_of(42, 7))
    assert generator.state() == state_of(1, 0)
    generator._commit_call(token)
    assert generator.calls == 1


def test_reseed_is_refused_during_a_reservation():
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    with pytest.raises(RuntimeError, match="reservation is outstanding"):
        generator.reseed(42)
    assert generator.state() == state_of(1, 0)
    generator._abandon_call(token)


def test_reset_is_refused_during_a_reservation():
    generator = NativeGenerator(1)
    generator._commit_call(generator._reserve_call())
    token = generator._reserve_call()
    with pytest.raises(RuntimeError, match="reservation is outstanding"):
        generator.reset()
    assert generator.calls == 1
    generator._abandon_call(token)
    # Once the reservation is gone, the same call succeeds.
    generator.reset()
    assert generator.calls == 0


def test_state_inspection_is_allowed_during_a_reservation():
    """Reading is safe mid-draw; only *replacement* is refused."""
    generator = NativeGenerator(1)
    token = generator._reserve_call()
    assert generator.state() == state_of(1, 0)
    assert generator.seed == 1 and generator.calls == 0
    generator._commit_call(token)


def test_module_generator_state_load_is_refused_during_a_reservation():
    module = NativeModule()
    module.g = NativeGenerator(1)
    token = module.g._reserve_call()
    with pytest.raises(RuntimeError, match="reservation is outstanding"):
        module.load_generator_state_dict({"g": state_of(42, 7)})
    assert module.generator_state_dict() == {"g": state_of(1, 0)}
    # Refusing only reads: the reservation is untouched and still usable.
    module.g._commit_call(token)
    assert module.g.calls == 1


# ======================================================================
# Concurrency — deterministic, with barriers and events, never sleeps
# ======================================================================


def test_a_concurrent_reservation_is_refused_without_receiving_an_index():
    """Thread A holds a reservation; thread B must fail deterministically
    and receive nothing. Sequenced by events, so there is no timing race
    and no sleep."""
    generator = NativeGenerator(1)
    reserved = threading.Event()
    b_finished = threading.Event()
    outcome = {}

    def second_caller():
        reserved.wait(JOIN_TIMEOUT)
        try:
            outcome["token"] = generator._reserve_call()
        except RuntimeError as error:
            outcome["error"] = error
        finally:
            b_finished.set()

    thread = threading.Thread(target=second_caller)
    thread.start()
    token = generator._reserve_call()
    reserved.set()
    assert b_finished.wait(JOIN_TIMEOUT), "the second caller never returned"
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()

    assert "token" not in outcome, "a second reservation received an index"
    assert isinstance(outcome["error"], RuntimeError)
    assert generator.calls == 0
    generator._commit_call(token)
    assert generator.calls == 1


def test_a_reentrant_reservation_on_the_same_thread_is_refused():
    """The same overlap, without threads: a callback running between
    reserve and commit cannot start a second draw."""
    generator = NativeGenerator(1)
    token = generator._reserve_call()

    def callback():
        with pytest.raises(RuntimeError):
            generator._reserve_call()

    callback()
    assert generator.calls == 0
    generator._commit_call(token)
    assert generator.calls == 1


def test_no_two_threads_ever_receive_the_same_call_index():
    """Several threads compete for one generator. Every *successful*
    reservation must carry a unique index, the committed indices must be
    exactly 0..n-1, and the counter must equal the number of commits."""
    generator = NativeGenerator(1)
    threads_count = 4
    per_thread = 25
    total = threads_count * per_thread
    start = threading.Barrier(threads_count)
    seen_lock = threading.Lock()
    reserved_indices = []
    committed_indices = []
    failures = []

    def worker():
        start.wait(JOIN_TIMEOUT)
        done = 0
        # Bounded: contention over a plain Lock with an integer-only
        # critical section resolves immediately, so this is generous.
        for _ in range(total * 100):
            if done == per_thread:
                return
            try:
                token = generator._reserve_call()
            except RuntimeError:
                continue          # another thread holds the reservation
            with seen_lock:
                reserved_indices.append(token._index)
            try:
                generator._commit_call(token)
            except BaseException as error:     # pragma: no cover
                failures.append(error)
                return
            with seen_lock:
                committed_indices.append(token._index)
            done += 1
        failures.append(RuntimeError("a worker ran out of attempts"))

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT)
        assert not thread.is_alive(), "a worker did not finish"

    assert failures == []
    assert len(reserved_indices) == total
    assert len(set(reserved_indices)) == total, "a call index was handed out twice"
    assert sorted(committed_indices) == list(range(total))
    assert generator.calls == total


def test_concurrent_state_reads_never_observe_a_torn_state():
    """A reader thread sampling state() while the main thread commits
    must only ever see well-formed, in-range values."""
    generator = NativeGenerator(1)
    stop = threading.Event()
    samples = []
    problems = []

    def reader():
        while not stop.is_set():
            state = generator.state()
            if set(state) != {"algorithm", "algorithm_version", "seed", "calls"}:
                problems.append(state)
            elif not 0 <= state["calls"] <= UINT64_MAX:
                problems.append(state)
            else:
                samples.append(state["calls"])

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(200):
            generator._commit_call(generator._reserve_call())
    finally:
        stop.set()
        thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()
    assert problems == []
    assert samples, "the reader never sampled"
    assert samples == sorted(samples), "the counter went backwards"
    assert generator.calls == 200


# ======================================================================
# NativeModule registration
# ======================================================================


def test_assignment_registers_a_generator():
    module = NativeModule()
    generator = NativeGenerator(1)
    module.gen = generator
    assert module.gen is generator
    assert list(module.named_generators()) == [("gen", generator)]
    assert module.generators() == [generator]


def test_register_generator_is_the_explicit_form():
    module = NativeModule()
    generator = NativeGenerator(1)
    module.register_generator("gen", generator)
    assert module.gen is generator
    assert module.generators() == [generator]


def test_registration_stores_the_exact_object_never_a_copy():
    module = NativeModule()
    generator = NativeGenerator(1)
    module.gen = generator
    assert module.gen is generator
    generator._commit_call(generator._reserve_call())
    assert module.gen.calls == 1


@pytest.mark.parametrize("bad", ["", 1, "a.b", b"gen", ("gen",)])
def test_register_generator_rejects_invalid_names(bad):
    # ``None`` is absent by design: as a *name* it is invalid, but the
    # unregister sentinel is the second argument, tested separately.
    module = NativeModule()
    with pytest.raises((TypeError, ValueError)):
        module.register_generator(bad, NativeGenerator(1))
    assert module.generators() == []


@pytest.mark.parametrize(
    "reserved", ["_parameters", "_modules", "_buffers", "_generators",
                 "training"]
)
def test_reserved_names_cannot_be_generator_names(reserved):
    module = NativeModule()
    with pytest.raises(ValueError, match="reserved"):
        module.register_generator(reserved, NativeGenerator(1))
    assert module.generators() == []
    assert module.training is True
    assert isinstance(module._generators, dict) and module._generators == {}


@pytest.mark.parametrize(
    "bad", [1, "x", 1.0, object(), NativeModule()],
)
def test_register_generator_rejects_non_generators(bad):
    module = NativeModule()
    with pytest.raises(TypeError):
        module.register_generator("gen", bad)
    assert module.generators() == []


def test_register_generator_none_unregisters():
    module = NativeModule()
    generator = NativeGenerator(1)
    module.gen = generator
    module.register_generator("gen", None)
    assert module.generators() == []
    assert module.gen is None
    # The object itself is untouched.
    assert generator.state() == state_of(1, 0)


def test_register_generator_none_raises_when_nothing_is_registered():
    module = NativeModule()
    with pytest.raises(KeyError):
        module.register_generator("gen", None)


def test_assigning_none_unregisters_a_generator():
    module = NativeModule()
    generator = NativeGenerator(1)
    module.gen = generator
    module.gen = None
    assert module.generators() == []
    assert module.gen is None
    assert generator.state() == state_of(1, 0)


def test_deleting_a_generator_attribute_unregisters_it():
    module = NativeModule()
    generator = NativeGenerator(1)
    module.gen = generator
    del module.gen
    assert module.generators() == []
    with pytest.raises(AttributeError):
        module.gen
    assert generator.state() == state_of(1, 0)


def test_assigning_an_ordinary_value_unregisters_a_generator():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    module.gen = "not a generator"
    assert module.generators() == []
    assert module.gen == "not a generator"


def test_replacing_a_generator_keeps_the_slot_position():
    module = NativeModule()
    first, second, third = (NativeGenerator(i) for i in range(3))
    module.a = first
    module.b = second
    module.a = third
    assert [name for name, _ in module.named_generators()] == ["a", "b"]
    assert module.a is third


def test_registering_a_generator_evicts_the_name_from_other_categories():
    module = NativeModule()
    module.slot = NativeModule()
    assert list(module._modules) == ["slot"]
    module.slot = NativeGenerator(1)
    assert list(module._modules) == []
    assert list(module._generators) == ["slot"]
    assert module.modules() == [module]


def test_registering_a_child_module_evicts_a_generator_of_the_same_name():
    module = NativeModule()
    module.slot = NativeGenerator(1)
    child = NativeModule()
    module.slot = child
    assert module.generators() == []
    assert module.slot is child


@needs_native
def test_registering_a_parameter_evicts_a_generator_of_the_same_name():
    module = NativeModule()
    module.slot = NativeGenerator(1)
    parameter = NativeParameter([[1.0, 2.0]])
    try:
        module.slot = parameter
        assert module.generators() == []
        assert module.slot is parameter
        # ...and back the other way.
        generator = NativeGenerator(2)
        module.slot = generator
        assert module.parameters() == []
        assert module.slot is generator
    finally:
        parameter.close()


@needs_native
def test_registering_a_buffer_evicts_a_generator_of_the_same_name():
    module = NativeModule()
    module.slot = NativeGenerator(1)
    tensor = NativeTensor.zeros((2,))
    try:
        module.register_buffer("slot", tensor)
        assert module.generators() == []
        assert module.slot is tensor
        # ...and back the other way.
        generator = NativeGenerator(2)
        module.slot = generator
        assert module.buffers() == []
        assert module.slot is generator
    finally:
        tensor.close()


def test_registering_before_super_init_raises_clearly():
    class Broken(NativeModule):
        def __init__(self):
            self.gen = NativeGenerator(1)

    with pytest.raises(RuntimeError, match=r"super\(\)\.__init__\(\)"):
        Broken()


def test_a_failed_registration_leaves_every_category_unchanged():
    module = NativeModule()
    child = NativeModule()
    module.kept = NativeGenerator(1)
    module.child = child
    before_generators = list(module._generators)
    before_modules = list(module._modules)
    with pytest.raises(TypeError):
        module.register_generator("new", "not a generator")
    with pytest.raises(ValueError):
        module.register_generator("training", NativeGenerator(2))
    assert list(module._generators) == before_generators
    assert list(module._modules) == before_modules
    assert module.child is child


# ======================================================================
# Traversal
# ======================================================================


def build_tree():
    """root(gen=a) -> child(gen=b) -> grandchild(gen=c)."""
    root, child, grandchild = NativeModule(), NativeModule(), NativeModule()
    a, b, c = NativeGenerator(1), NativeGenerator(2), NativeGenerator(3)
    root.gen = a
    root.child = child
    child.gen = b
    child.grandchild = grandchild
    grandchild.gen = c
    return root, (a, b, c)


def test_traversal_is_recursive_and_pre_order_depth_first():
    root, (a, b, c) = build_tree()
    assert list(root.named_generators()) == [
        ("gen", a), ("child.gen", b), ("child.grandchild.gen", c),
    ]
    assert root.generators() == [a, b, c]


def test_traversal_puts_a_modules_own_generators_before_its_descendants():
    root = NativeModule()
    child = NativeModule()
    child.deep = NativeGenerator(9)
    root.child = child
    root.own = NativeGenerator(1)
    assert [name for name, _ in root.named_generators()] == [
        "own", "child.deep",
    ]


def test_recurse_false_restricts_to_direct_generators():
    root, (a, _, _) = build_tree()
    assert list(root.named_generators(recurse=False)) == [("gen", a)]
    assert root.generators(recurse=False) == [a]


def test_a_prefix_is_applied_to_every_name():
    root, (a, b, c) = build_tree()
    assert [name for name, _ in root.named_generators(prefix="model")] == [
        "model.gen", "model.child.gen", "model.child.grandchild.gen",
    ]


def test_a_shared_generator_is_yielded_once_under_its_first_name():
    root = NativeModule()
    child = NativeModule()
    shared = NativeGenerator(1)
    root.first = shared
    root.child = child
    child.also = shared
    assert list(root.named_generators()) == [("first", shared)]
    assert root.generators() == [shared]


def test_a_generator_registered_twice_on_one_module_is_yielded_once():
    module = NativeModule()
    shared = NativeGenerator(1)
    module.a = shared
    module.b = shared
    assert list(module.named_generators()) == [("a", shared)]


def test_independent_generators_with_identical_state_are_both_yielded():
    """Deduplication is by identity, never by value: two generators with
    the same seed and counter are two streams."""
    module = NativeModule()
    first, second = NativeGenerator(7), NativeGenerator(7)
    module.a = first
    module.b = second
    assert first.state() == second.state()
    assert list(module.named_generators()) == [("a", first), ("b", second)]


def test_a_shared_child_module_contributes_its_generators_once():
    root = NativeModule()
    shared_child = NativeModule()
    shared_child.gen = NativeGenerator(1)
    root.first = shared_child
    root.second = shared_child
    assert [name for name, _ in root.named_generators()] == ["first.gen"]


def test_traversal_terminates_on_a_module_cycle():
    root = NativeModule()
    child = NativeModule()
    root.child = child
    child.back = root          # a cycle
    root.gen = NativeGenerator(1)
    child.gen = NativeGenerator(2)
    names = [name for name, _ in root.named_generators()]
    assert names == ["gen", "child.gen"]


def test_traversal_ignores_ordinary_attributes():
    module = NativeModule()
    module.not_registered = "a string"
    module.number = 7
    assert module.generators() == []


def test_generators_are_not_parameters_or_buffers():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    assert module.parameters() == []
    assert module.buffers() == []
    assert list(module.named_parameters()) == []
    assert list(module.named_buffers()) == []


# ======================================================================
# The generator state surface, and state_dict() staying tensor-only
# ======================================================================


def test_generator_state_dict_reports_canonical_names_and_exact_state():
    root, (a, b, c) = build_tree()
    a.load_state(state_of(11, 5))
    assert root.generator_state_dict() == {
        "gen": state_of(11, 5),
        "child.gen": state_of(2, 0),
        "child.grandchild.gen": state_of(3, 0),
    }


def test_generator_state_dict_is_empty_without_generators():
    assert NativeModule().generator_state_dict() == {}


def test_generator_state_dict_values_are_detached_copies():
    module = NativeModule()
    generator = NativeGenerator(1)
    module.gen = generator
    report = module.generator_state_dict()
    report["gen"]["seed"] = 999
    assert generator.seed == 1
    assert module.generator_state_dict()["gen"]["seed"] == 1


def test_generator_state_dict_does_not_advance_or_reserve():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    for _ in range(3):
        module.generator_state_dict()
    assert module.gen.calls == 0
    module.gen._commit_call(module.gen._reserve_call())
    assert module.gen.calls == 1


def test_a_shared_generator_reports_one_entry():
    root = NativeModule()
    child = NativeModule()
    shared = NativeGenerator(1)
    root.first = shared
    root.child = child
    child.also = shared
    assert list(root.generator_state_dict()) == ["first"]


def test_state_dict_stays_tensor_only_and_ignores_generators():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    assert module.state_dict() == {}


@needs_native
def test_state_dict_is_unchanged_by_adding_a_generator():
    module = NativeModule()
    module.weight = NativeParameter([[1.0, 2.0]])
    try:
        before = module.state_dict()
        try:
            keys_before = list(before)
        finally:
            for value in before.values():
                value.close()
        module.gen = NativeGenerator(1)
        after = module.state_dict()
        try:
            assert list(after) == keys_before == ["weight"]
            assert all(isinstance(value, NativeTensor)
                       for value in after.values())
        finally:
            for value in after.values():
                value.close()
    finally:
        module.weight.close()


def test_load_generator_state_dict_loads_in_place():
    root, (a, b, c) = build_tree()
    result = root.load_generator_state_dict({
        "gen": state_of(11, 1),
        "child.gen": state_of(22, 2),
        "child.grandchild.gen": state_of(33, 3),
    })
    assert result.missing_keys == () and result.unexpected_keys == ()
    assert a.state() == state_of(11, 1)
    assert root.generators() == [a, b, c]      # identities preserved


def test_load_generator_state_dict_round_trips_its_own_report():
    root, _ = build_tree()
    saved = root.generator_state_dict()
    root.load_generator_state_dict({
        name: state_of(0, 0) for name in saved
    })
    root.load_generator_state_dict(saved)
    assert root.generator_state_dict() == saved


def test_load_generator_state_dict_is_strict_about_keys():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    with pytest.raises(ValueError, match="do not match"):
        module.load_generator_state_dict({})
    with pytest.raises(ValueError, match="do not match"):
        module.load_generator_state_dict({
            "gen": state_of(2, 0), "other": state_of(3, 0),
        })
    assert module.gen.state() == state_of(1, 0)


def test_load_generator_state_dict_non_strict_reports_both_lists():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    module.spare = NativeGenerator(2)
    result = module.load_generator_state_dict(
        {"gen": state_of(9, 1), "ghost": state_of(0, 0)}, strict=False
    )
    assert result.missing_keys == ("spare",)
    assert result.unexpected_keys == ("ghost",)
    assert module.gen.state() == state_of(9, 1)
    assert module.spare.state() == state_of(2, 0)


def test_load_generator_state_dict_rejects_an_alias_key():
    root = NativeModule()
    child = NativeModule()
    shared = NativeGenerator(1)
    root.first = shared
    root.child = child
    child.also = shared
    with pytest.raises(ValueError, match="unexpected"):
        root.load_generator_state_dict({
            "first": state_of(5, 0), "child.also": state_of(5, 0),
        })
    assert shared.state() == state_of(1, 0)


def test_load_generator_state_dict_is_atomic_across_generators():
    """A bad value for the *second* generator must leave the first
    unchanged: validation precedes every assignment."""
    module = NativeModule()
    first, second = NativeGenerator(1), NativeGenerator(2)
    module.a = first
    module.b = second
    bad = state_of(9, 9)
    bad["algorithm_version"] = 99
    with pytest.raises(ValueError):
        module.load_generator_state_dict({"a": state_of(7, 7), "b": bad})
    assert first.state() == state_of(1, 0)
    assert second.state() == state_of(2, 0)


@pytest.mark.parametrize("bad", [None, "x", 1])
def test_load_generator_state_dict_rejects_a_non_mapping(bad):
    module = NativeModule()
    module.gen = NativeGenerator(1)
    with pytest.raises(TypeError):
        module.load_generator_state_dict(bad)
    assert module.gen.state() == state_of(1, 0)


def test_load_generator_state_dict_validates_strict_and_keys():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    with pytest.raises(TypeError):
        module.load_generator_state_dict({}, strict="yes")
    with pytest.raises(TypeError):
        module.load_generator_state_dict({1: state_of(2, 0)})
    assert module.gen.state() == state_of(1, 0)


def test_a_shared_generator_loads_once_and_every_alias_observes_it():
    root = NativeModule()
    child = NativeModule()
    shared = NativeGenerator(1)
    root.first = shared
    root.child = child
    child.also = shared
    root.load_generator_state_dict({"first": state_of(42, 4)})
    assert shared.state() == state_of(42, 4)
    assert root.first is child.also is shared


def test_a_shared_generator_is_assigned_exactly_once(monkeypatch):
    """A generator registered under three paths has one canonical key, is
    staged once, and is written once — never twice with the same or
    different values."""
    root = NativeModule()
    child = NativeModule()
    shared = NativeGenerator(1)
    root.first = shared
    root.second = shared
    root.child = child
    child.also = shared
    assert list(root.generator_state_dict()) == ["first"]

    writes = []
    original = NativeGenerator._assign_state

    def counting_assign(self, seed, calls):
        writes.append((id(self), seed, calls))
        original(self, seed, calls)

    monkeypatch.setattr(NativeGenerator, "_assign_state", counting_assign)
    root.load_generator_state_dict({"first": state_of(42, 4)})
    assert writes == [(id(shared), 42, 4)]


@pytest.mark.parametrize("strict", [True, False])
def test_conflicting_states_through_an_alias_are_never_both_applied(strict):
    """Two different states supplied for one shared generator, under its
    canonical name and an alias. The alias is not a canonical key, so it
    is rejected (strict) or reported as unexpected (non-strict) — never
    silently applied on top of the canonical one."""
    root = NativeModule()
    child = NativeModule()
    shared = NativeGenerator(1)
    root.first = shared
    root.child = child
    child.also = shared
    supplied = {"first": state_of(42, 4), "child.also": state_of(99, 9)}

    if strict:
        with pytest.raises(ValueError, match="unexpected"):
            root.load_generator_state_dict(supplied, strict=True)
        assert shared.state() == state_of(1, 0)      # nothing applied
    else:
        result = root.load_generator_state_dict(supplied, strict=False)
        assert result.unexpected_keys == ("child.also",)
        # Exactly the canonical state, never the alias's conflicting one.
        assert shared.state() == state_of(42, 4)


def test_a_failed_commit_rolls_every_generator_back(monkeypatch):
    """Injected failure part-way through the commit. The rollback is
    built from the same non-failing primitive as the commit, so it can
    never leave one generator loaded and another not."""
    module = NativeModule()
    first, second, third = (NativeGenerator(i) for i in (1, 2, 3))
    module.a, module.b, module.c = first, second, third
    before = module.generator_state_dict()

    original = NativeGenerator._assign_state
    calls = {"n": 0}
    boom = KeyboardInterrupt("interrupted mid-commit")

    def failing_assign(self, seed, calls_value):
        calls["n"] += 1
        if calls["n"] == 3:          # the third generator's write
            raise boom
        original(self, seed, calls_value)

    monkeypatch.setattr(NativeGenerator, "_assign_state", failing_assign)
    with pytest.raises(KeyboardInterrupt):
        module.load_generator_state_dict({
            "a": state_of(11, 1), "b": state_of(22, 2), "c": state_of(33, 3),
        })
    monkeypatch.undo()

    # All three are back where they started — not one, not two.
    assert module.generator_state_dict() == before
    # Identities survived, and the module is immediately usable again.
    assert (module.a, module.b, module.c) == (first, second, third)
    module.load_generator_state_dict({
        "a": state_of(11, 1), "b": state_of(22, 2), "c": state_of(33, 3),
    })
    assert module.generator_state_dict() == {
        "a": state_of(11, 1), "b": state_of(22, 2), "c": state_of(33, 3),
    }


def test_a_validation_failure_writes_nothing_at_all(monkeypatch):
    """The complement: everything that can fail happens during staging,
    so a bad value means the commit loop never runs."""
    module = NativeModule()
    module.a = NativeGenerator(1)
    module.b = NativeGenerator(2)
    before = module.generator_state_dict()

    writes = []
    original = NativeGenerator._assign_state
    monkeypatch.setattr(
        NativeGenerator, "_assign_state",
        lambda self, seed, calls: writes.append(id(self)),
    )
    bad = state_of(9, 9)
    bad["algorithm"] = "tensorforge.philox"
    with pytest.raises(ValueError, match="algorithm mismatch"):
        module.load_generator_state_dict({"a": state_of(7, 7), "b": bad})
    monkeypatch.undo()

    assert writes == [], "a value was written despite a staging failure"
    assert module.generator_state_dict() == before


def test_a_load_refused_by_a_reservation_writes_nothing_and_keeps_it():
    """The reservation check is part of staging, so a mid-draw generator
    blocks the whole load — and refusing only reads, so the reservation
    itself is untouched and still committable."""
    module = NativeModule()
    first, second = NativeGenerator(1), NativeGenerator(2)
    module.a, module.b = first, second
    before = module.generator_state_dict()

    token = second._reserve_call()
    with pytest.raises(RuntimeError, match="reservation is outstanding"):
        module.load_generator_state_dict({
            "a": state_of(11, 1), "b": state_of(22, 2),
        })
    # ``a`` was staged first but never written.
    assert module.generator_state_dict() == before
    # The reservation survived the refusal intact.
    second._commit_call(token)
    assert second.calls == 1


# ======================================================================
# The multi-generator load is one locked transaction
# ======================================================================
#
# The rule these pin: **no reservation may begin on any target between
# the final reservation check and the end of the commit.** A
# check-release-write shape cannot provide that, so the transaction holds
# every target's lock across the recheck, the snapshots, and the writes,
# acquiring them in a global id()-based order so overlapping loads cannot
# deadlock. Every test is event- or barrier-sequenced with bounded joins;
# none sleeps.


def test_a_reservation_taken_before_the_locks_rejects_the_whole_load():
    """A reservation that begins after validation but before the
    transaction acquires its locks must make the load fail — and fail
    having written nothing."""
    module = NativeModule()
    first, second = NativeGenerator(1), NativeGenerator(2)
    module.a, module.b = first, second
    before = module.generator_state_dict()
    holder = {}

    original_order = native_generator_module._ordered_targets

    def reserve_then_order(generators):
        # The exact seam: validation is done, no lock is held yet.
        if "token" not in holder:
            holder["token"] = second._reserve_call()
        return original_order(generators)

    native_generator_module._ordered_targets = reserve_then_order
    try:
        with pytest.raises(RuntimeError, match="reservation is outstanding"):
            module.load_generator_state_dict({
                "a": state_of(11, 1), "b": state_of(22, 2),
            })
    finally:
        native_generator_module._ordered_targets = original_order

    assert module.generator_state_dict() == before
    # The pre-existing reservation survived the rejected load intact.
    second._commit_call(holder["token"])
    assert second.calls == 1
    assert module.a is first and module.b is second


def test_a_reservation_racing_the_commit_never_sees_partial_state():
    """A second thread attempting a reservation while the commit holds
    the locks blocks on the lock and, when it proceeds, observes the
    **complete** new state — never a half-applied one."""
    module = NativeModule()
    first, second = NativeGenerator(1), NativeGenerator(2)
    module.a, module.b = first, second
    target = {"a": state_of(11, 1), "b": state_of(22, 2)}

    mid_commit = threading.Event()
    observed = {}

    def racer():
        assert mid_commit.wait(JOIN_TIMEOUT)
        # Blocks until the transaction releases every lock.
        try:
            token = second._reserve_call()
            observed["state"] = second.state()
            second._abandon_call(token)
        except RuntimeError as error:      # pragma: no cover - defensive
            observed["error"] = error
            observed["state"] = second.state()

    thread = threading.Thread(target=racer)
    thread.start()

    original_assign = NativeGenerator._assign_state
    writes = []

    def signalling_assign(self, seed, calls):
        original_assign(self, seed, calls)
        writes.append(id(self))
        if len(writes) == 1:
            mid_commit.set()       # one target written, one not yet

    NativeGenerator._assign_state = signalling_assign
    try:
        module.load_generator_state_dict(target)
    finally:
        NativeGenerator._assign_state = original_assign

    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive(), "the racing reservation never returned"
    assert len(writes) == 2
    # Whatever it saw, it saw the finished transaction.
    assert observed["state"] == state_of(22, 2)
    assert module.generator_state_dict() == target


class _RendezvousLock:
    """A generator lock that makes two threads provably interleave.

    The first thread to enter it is held at a barrier while still owning
    the lock. Installing one on each of two generators forces both
    threads to hold one lock apiece before either reaches for the second
    — the exact state that deadlocks under a caller-derived acquisition
    order and is harmless under a global one. Without this the race is
    real but improbable, and a passing stress loop would prove nothing."""

    def __init__(self, inner, barrier):
        self._inner = inner
        self._barrier = barrier
        self._armed = True

    def __enter__(self):
        result = self._inner.__enter__()
        if self._armed:
            self._armed = False
            try:
                self._barrier.wait(timeout=1.5)
            except threading.BrokenBarrierError:
                pass          # the good path: the other thread is blocked
        return result

    def __exit__(self, *exception):
        return self._inner.__exit__(*exception)


def test_overlapping_loads_cannot_deadlock_even_when_forced_to_interleave():
    """The ordering property, proved rather than sampled.

    Two modules hold the same two generators in **opposite** canonical
    order. Each thread is pinned at a barrier holding its first lock, so
    a caller-derived order would have each thread waiting on the lock the
    other owns. The global ``id()`` order makes both reach for the *same*
    lock first, so one simply waits and then proceeds."""
    first, second = NativeGenerator(1), NativeGenerator(2)
    forward, backward = NativeModule(), NativeModule()
    forward.a, forward.b = first, second
    backward.a, backward.b = second, first
    assert forward.generators() == [first, second]
    assert backward.generators() == [second, first]

    barrier = threading.Barrier(2)
    original_locks = {}
    for generator in (first, second):
        original_locks[id(generator)] = generator._lock
        generator._lock = _RendezvousLock(generator._lock, barrier)

    start = threading.Barrier(2)
    failures = []

    def load(module, states):
        try:
            start.wait(JOIN_TIMEOUT)
            module.load_generator_state_dict(states)
        except BaseException as error:      # pragma: no cover - defensive
            failures.append(error)

    # Daemon threads: if the acquisition order ever regresses, these
    # genuinely deadlock, and a non-daemon thread would then hang the
    # whole session at interpreter exit instead of failing this test.
    threads = [
        threading.Thread(target=load, daemon=True, args=(
            forward, {"a": state_of(11, 1), "b": state_of(22, 2)})),
        threading.Thread(target=load, daemon=True, args=(
            backward, {"a": state_of(22, 2), "b": state_of(11, 1)})),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(JOIN_TIMEOUT)
            assert not thread.is_alive(), (
                "a concurrent generator load deadlocked: the acquisition "
                "order is not global"
            )
    finally:
        barrier.abort()
        for generator in (first, second):
            generator._lock = original_locks[id(generator)]

    assert failures == []
    assert first.state() == state_of(11, 1)
    assert second.state() == state_of(22, 2)


class _GateLock:
    """A generator lock that announces its first acquisition and then
    waits to be let onward.

    Where ``_RendezvousLock`` makes two threads meet symmetrically, this
    makes the meeting *directional*: the test can guarantee which thread
    owns this lock when the other one reaches for the next one. That is
    what turns a lock-order inversion from an improbable race into a
    certainty."""

    def __init__(self, inner, acquired, proceed, timeout=JOIN_TIMEOUT):
        self._inner = inner
        self._acquired = acquired
        self._proceed = proceed
        self._timeout = timeout
        self._armed = True

    def __enter__(self):
        result = self._inner.__enter__()
        if self._armed:
            self._armed = False
            self._acquired.set()
            self._proceed.wait(self._timeout)
        return result

    def __exit__(self, *exception):
        return self._inner.__exit__(*exception)


def test_construction_reentry_cannot_invert_the_multi_generator_lock_order():
    """The inversion the two-phase protocol removes, reproduced exactly.

    ``late`` is the generator that sorts **second** in the global order.
    One thread reserves on it; its token constructor starts a replacement
    naming both generators in *reverse* caller order. A second thread
    starts a replacement naming them forwards, and is gated so that it
    provably owns ``early`` before the first thread's constructor runs.

    If the token were still built under the generator's lock, the
    reserving thread would own ``late`` and reach for ``early`` while the
    second thread owned ``early`` and reached for ``late`` — a cycle, and
    a permanent hang. Constructing outside the lock means the reserving
    thread owns nothing, so both transactions take the same global order,
    and both are simply refused by the live claim. Every thread is a
    daemon: a regression fails this test instead of hanging the session."""
    generators = [NativeGenerator(1), NativeGenerator(2)]
    early, late = native_generator_module._ordered_targets(generators)
    before = {id(early): early.state(), id(late): late.state()}
    target = {id(early): state_of(111, 1), id(late): state_of(222, 2)}

    def entries(order):
        return [
            native_generator_module.GeneratorStateEntry(
                label=f"g{position}", generator=generator,
                state=target[id(generator)],
            )
            for position, generator in enumerate(order)
        ]

    constructing = threading.Event()
    holds_early = threading.Event()
    release_early = threading.Event()
    outcome = {}
    saved_lock = early._lock
    early._lock = _GateLock(saved_lock, holds_early, release_early)
    real = native_generator_module._ReservationToken

    def constructor(generator, serial, index):
        constructing.set()
        # The other thread now owns ``early`` and is about to reach for
        # ``late``. Anything this constructor holds is a cycle.
        assert holds_early.wait(JOIN_TIMEOUT)
        release_early.set()
        try:
            # Reverse caller order — the transaction must still use the
            # global one, or these two threads form a cycle.
            native_generator_module.replace_generator_states(
                entries([late, early])
            )
            outcome["inner"] = None
        except RuntimeError as error:
            outcome["inner"] = error
        return real(generator, serial, index)

    def reserve():
        try:
            outcome["token"] = late._reserve_call()
        except BaseException as error:      # pragma: no cover - defensive
            outcome["reserve_error"] = error

    def replace():
        assert constructing.wait(JOIN_TIMEOUT)
        try:
            native_generator_module.replace_generator_states(
                entries([early, late])
            )
            outcome["outer"] = None
        except RuntimeError as error:
            outcome["outer"] = error

    threads = [
        threading.Thread(target=reserve, daemon=True),
        threading.Thread(target=replace, daemon=True),
    ]
    native_generator_module._ReservationToken = constructor
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(JOIN_TIMEOUT)
        alive = [thread.is_alive() for thread in threads]
    finally:
        native_generator_module._ReservationToken = real
        release_early.set()
        early._lock = saved_lock

    assert alive == [False, False], (
        "a generator lock order inverted: token construction is holding a "
        "lock that a multi-generator transaction then reaches across"
    )
    assert "reserve_error" not in outcome, outcome.get("reserve_error")

    # Both replacements met the in-flight reservation on ``late`` and were
    # refused, so neither generator was written — not even ``early``,
    # which each transaction named first.
    assert isinstance(outcome["inner"], RuntimeError)
    assert isinstance(outcome["outer"], RuntimeError)
    assert early.state() == before[id(early)]
    assert late.state() == before[id(late)]

    # One index, issued once, and the reservation still commits cleanly.
    token = outcome["token"]
    assert token._index == before[id(late)]["calls"]
    late._commit_call(token)
    assert late.calls == before[id(late)]["calls"] + 1


def test_overlapping_loads_from_different_modules_do_not_deadlock():
    """The same property under sustained contention — a cheap stress
    complement to the forced-interleave proof above."""
    first, second = NativeGenerator(1), NativeGenerator(2)
    forward, backward = NativeModule(), NativeModule()
    forward.a, forward.b = first, second
    backward.a, backward.b = second, first
    assert forward.generators() == [first, second]
    assert backward.generators() == [second, first]

    start = threading.Barrier(2)
    failures = []

    def hammer(module, states):
        try:
            start.wait(JOIN_TIMEOUT)
            for _ in range(200):
                module.load_generator_state_dict(states)
        except BaseException as error:     # pragma: no cover - defensive
            failures.append(error)

    threads = [
        threading.Thread(target=hammer, args=(
            forward, {"a": state_of(11, 1), "b": state_of(22, 2)})),
        threading.Thread(target=hammer, args=(
            backward, {"a": state_of(22, 2), "b": state_of(11, 1)})),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT)
        assert not thread.is_alive(), "a concurrent load deadlocked"
    assert failures == []
    # Both loads write the same values, so the end state is determinate.
    assert first.state() == state_of(11, 1)
    assert second.state() == state_of(22, 2)


def test_mapping_order_does_not_change_the_lock_order():
    """The acquisition order is a property of the generators, not of the
    supplied mapping — so the same load in either key order acquires the
    same sequence and behaves identically."""
    module = NativeModule()
    first, second, third = (NativeGenerator(i) for i in (1, 2, 3))
    module.a, module.b, module.c = first, second, third

    seen = []
    original_order = native_generator_module._ordered_targets

    def recording_order(generators):
        ordered = original_order(generators)
        seen.append([id(g) for g in ordered])
        return ordered

    native_generator_module._ordered_targets = recording_order
    try:
        module.load_generator_state_dict({
            "a": state_of(11, 1), "b": state_of(22, 2), "c": state_of(33, 3),
        })
        module.load_generator_state_dict({
            "c": state_of(33, 3), "a": state_of(11, 1), "b": state_of(22, 2),
        })
    finally:
        native_generator_module._ordered_targets = original_order

    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0] == sorted(seen[0])


def test_a_shared_generator_is_locked_and_written_exactly_once():
    """Three registered paths, one object: one lock acquisition, one
    write. Locking it twice would deadlock a plain Lock and merely waste
    work on an RLock — either way it must not happen."""
    root, child = NativeModule(), NativeModule()
    shared = NativeGenerator(1)
    root.first = shared
    root.second = shared
    root.child = child
    child.also = shared

    ordered = []
    writes = []
    original_order = native_generator_module._ordered_targets
    original_assign = NativeGenerator._assign_state

    def recording_order(generators):
        result = original_order(generators)
        ordered.append(list(result))
        return result

    def recording_assign(self, seed, calls):
        writes.append((id(self), seed, calls))
        original_assign(self, seed, calls)

    native_generator_module._ordered_targets = recording_order
    NativeGenerator._assign_state = recording_assign
    try:
        root.load_generator_state_dict({"first": state_of(42, 4)})
    finally:
        native_generator_module._ordered_targets = original_order
        NativeGenerator._assign_state = original_assign

    assert ordered == [[shared]]
    assert writes == [(id(shared), 42, 4)]
    assert shared.state() == state_of(42, 4)


def test_conflicting_states_for_one_object_are_rejected_by_the_transaction():
    """The transaction's own defence, below the module's key handling: the
    same generator handed in twice with different states is a conflict,
    not a last-write-wins."""
    generator = NativeGenerator(1)
    entries = [
        native_generator_module.GeneratorStateEntry(
            "first", generator, state_of(11, 1)),
        native_generator_module.GeneratorStateEntry(
            "alias", generator, state_of(22, 2)),
    ]
    with pytest.raises(ValueError, match="conflicting states"):
        native_generator_module.replace_generator_states(entries)
    assert generator.state() == state_of(1, 0)

    # The same state twice is not a conflict; it folds to one write.
    same = [
        native_generator_module.GeneratorStateEntry(
            "first", generator, state_of(11, 1)),
        native_generator_module.GeneratorStateEntry(
            "alias", generator, state_of(11, 1)),
    ]
    native_generator_module.replace_generator_states(same)
    assert generator.state() == state_of(11, 1)


def test_a_failed_multi_generator_load_leaves_every_generator_unchanged():
    """Four targets, the third one rejected. Nothing is written — and the
    rejection happens under the locks, so no other thread could have seen
    an intermediate state either."""
    module = NativeModule()
    generators = [NativeGenerator(i) for i in range(4)]
    for index, generator in enumerate(generators):
        module.register_generator(f"g{index}", generator)
    before = module.generator_state_dict()

    bad = state_of(9, 9)
    bad["algorithm_version"] = 77
    with pytest.raises(ValueError, match="algorithm version mismatch"):
        module.load_generator_state_dict({
            "g0": state_of(10, 1), "g1": state_of(20, 2),
            "g2": bad, "g3": state_of(40, 4),
        })
    assert module.generator_state_dict() == before
    assert module.generators() == generators        # identities preserved

    # A reservation-caused rejection is the same story.
    token = generators[3]._reserve_call()
    with pytest.raises(RuntimeError, match="reservation is outstanding"):
        module.load_generator_state_dict({
            "g0": state_of(10, 1), "g1": state_of(20, 2),
            "g2": state_of(30, 3), "g3": state_of(40, 4),
        })
    assert module.generator_state_dict() == before
    generators[3]._abandon_call(token)

    # ...and the module is immediately usable afterwards.
    module.load_generator_state_dict({
        "g0": state_of(10, 1), "g1": state_of(20, 2),
        "g2": state_of(30, 3), "g3": state_of(40, 4),
    })
    assert module.generator_state_dict()["g3"] == state_of(40, 4)


def test_an_empty_generator_transaction_is_a_no_op():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    result = module.load_generator_state_dict({}, strict=False)
    assert result.missing_keys == ("gen",)
    assert module.gen.state() == state_of(1, 0)


def test_loading_preserves_every_generator_identity():
    root, (a, b, c) = build_tree()
    identities = [id(g) for g in (a, b, c)]
    root.load_generator_state_dict({
        "gen": state_of(11, 1),
        "child.gen": state_of(22, 2),
        "child.grandchild.gen": state_of(33, 3),
    })
    assert [id(g) for g in root.generators()] == identities
    assert root.gen is a and root.child.gen is b


# ======================================================================
# Interaction with the rest of the module contract
# ======================================================================


def test_train_and_eval_do_not_touch_generator_state():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    module.gen._commit_call(module.gen._reserve_call())
    before = module.gen.state()
    module.eval()
    module.train()
    assert module.gen.state() == before
    assert module.gen is not None


def test_zero_grad_ignores_generators():
    module = NativeModule()
    module.gen = NativeGenerator(1)
    module.zero_grad()
    assert module.gen.state() == state_of(1, 0)


def test_dropping_a_module_does_not_touch_its_generator():
    generator = NativeGenerator(1)
    module = NativeModule()
    module.gen = generator
    del module
    # No close(), no reset, no lifecycle: the object is exactly as it was.
    assert generator.state() == state_of(1, 0)
    generator._commit_call(generator._reserve_call())
    assert generator.calls == 1


def test_unregistering_one_alias_leaves_a_shared_generator_alive():
    shared = NativeGenerator(1)
    first, second = NativeModule(), NativeModule()
    first.gen = shared
    second.gen = shared
    del first.gen
    assert first.generators() == []
    assert second.generators() == [shared]
    assert shared.state() == state_of(1, 0)


def test_named_modules_is_unaffected_by_generators():
    root, _ = build_tree()
    assert [name for name, _ in root.named_modules()] == [
        "", "child", "child.grandchild",
    ]


# ======================================================================
# Capability boundary and stable/native separation
# ======================================================================


def test_native_generator_is_experimental_only():
    import tensorforge
    import tensorforge.experimental as experimental

    assert "NativeGenerator" in experimental.__all__
    assert experimental.NativeGenerator is NativeGenerator
    assert not hasattr(tensorforge, "NativeGenerator")
    for name in ("NativeDropout", "NativeRNG", "manual_seed", "seed"):
        assert not hasattr(tensorforge, name), name


def test_the_generator_layer_ships_no_dropout_or_random_operation():
    """The generator is *state*: nothing in G1 differentiates, draws, or
    exposes a random operation, and nothing above the Core has shipped
    since.

    Two names have left the absence list as their milestones landed, each
    into a different layer. ``"dropout_forward"`` left at **G2**, which
    shipped it as a layer-qualified **Core** wrapper — a stateless kernel
    entry that takes an explicit seed and call index and touches no
    generator. ``"dropout"`` left at **G3**, which shipped the
    differentiable ``NativeTensor.dropout`` operation over that Core; it
    is the one caller of the reservation protocol, and it lives one layer
    above the generator, never on it. Both are the Core/operation split
    conv2d, maxpool2d, and cross_entropy already follow, and both are
    covered elsewhere (tests/test_native_dropout_core.py and
    tests/test_native_dropout_autograd.py). ``NativeDropout`` left at
    **G4**, which shipped the module over that operation — it is a
    *consumer* of registered generator state, covered by
    tests/test_native_dropout_module.py. What stays absent *here* is any
    numerical surface on the **generator itself**."""
    import tensorforge.experimental as experimental

    # The G4 module registers a generator; it does not extend one.
    assert hasattr(experimental, "NativeDropout")
    assert "NativeDropout" in experimental.__all__
    assert not hasattr(NativeGenerator, "NativeDropout")
    for name in ("rand", "randn", "bernoulli", "uniform",
                 "random", "dropout_backward"):
        assert not hasattr(NativeTensor, name), name
        assert not hasattr(cpp.NativeTensorCore, name), name
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
    # The G3 operation exists on NativeTensor, and only there: not on the
    # Core, not on the generator, not as a kernel.
    assert hasattr(NativeTensor, "dropout")
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout" not in cpp.TENSOR_CORE_OPS
    assert "dropout" not in cpp.RAW_KERNELS
    assert not hasattr(cpp.NativeTensorCore, "dropout")
    assert not hasattr(NativeGenerator, "dropout")
    assert not hasattr(NativeGenerator(1), "dropout")
    # The G2 Core forward exists, is layer-qualified, and stops there.
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    assert hasattr(cpp.NativeTensorCore, "dropout_forward")
    assert not hasattr(NativeTensor, "dropout_forward")
    assert "dropout_forward" not in cpp.AUTOGRAD_OPS
    assert "dropout_forward" not in cpp.RAW_KERNELS


def test_g1_moved_no_capability_registry_value():
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "NativeGenerator" not in cpp.NATIVE_MODULES
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS):
        assert "NativeGenerator" not in inventory
        assert "generator" not in inventory


def test_the_checkpoint_format_name_never_moves_and_the_version_did():
    """G1 shipped generator state without touching the checkpoint; G5
    persisted it and moved the version to 2. The **name** is the part
    that is locked forever — a new schema is a new version of one format,
    never a second format."""
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 2
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2)
    # G5 serializes generator state through the generator's own locked
    # snapshot/replacement transactions — never by reaching into the
    # generator's private lock, reservation slot, or token machinery.
    source = native_checkpoint.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for absent in ("_reserve_call", "_commit_call", "_abandon_call",
                   "_ReservationToken", "_claim_serial", "_active_serial",
                   "._lock", "_assign_state"):
        assert absent not in text, (
            f"native_checkpoint.py reaches into {absent!r}; the generator "
            f"transaction owns the locking, not the checkpoint"
        )


def test_the_generator_owns_no_c_abi_symbol():
    """``NativeGenerator`` is pure Python: it has no kernel of its own,
    and the only Phase-G C ABI symbol that exists is G2's stateless
    Dropout forward — which is registered as a checked kernel and is
    reachable only through ``NativeTensorCore``, never as a module-level
    attribute of the backend wrapper."""
    for symbol in ("tf_core_dropout", "tf_core_dropout_backward",
                   "tf_core_random", "tf_core_random_mask",
                   "tf_core_bernoulli", "tf_core_splitmix64",
                   "tf_core_generator", "tf_core_seed"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
        assert not hasattr(cpp, symbol), symbol
    assert "tf_core_dropout_forward" in cpp._CHECKED_KERNELS
    assert not hasattr(cpp, "tf_core_dropout_forward")


def test_the_stable_framework_is_untouched_by_the_native_generator():
    import tensorforge
    import tensorforge.nn as nn

    # The stable Dropout is the stable line's own, unrelated layer.
    assert hasattr(nn, "Dropout")
    assert not hasattr(nn.Dropout, "_reserve_call")
    assert not hasattr(tensorforge, "NativeGenerator")
    # A native generator is not accepted anywhere in the stable line.
    generator = NativeGenerator(1)
    assert not isinstance(generator, tensorforge.Tensor)
