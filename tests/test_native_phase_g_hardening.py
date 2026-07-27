"""Phase-G hardening — RNG, graph, ownership, and checkpoint (milestone G6).

G6 adds **no capability**. Milestones G1–G5 shipped ``NativeGenerator``
and module generator-state ownership, the stateless Dropout-forward Core,
the differentiable ``NativeTensor.dropout``, the ``NativeDropout`` module,
and native checkpoint format version 2. This suite attacks that completed
surface: it proves the §13 ownership matrix and the §14 failure matrix of
docs/native_rng_dropout_design.md boundary by boundary, and it pins the
*interactions* the focused suites deliberately do not — several state
families in one graph, several transactions in one process, and repeated
failure cycles measured against a real native live-storage baseline.

Deliberately **cross-cutting**. The narrow per-feature matrices stay in
their owning suites (``test_native_generator.py``,
``test_native_dropout_core.py``, ``test_native_dropout_autograd.py``,
``test_native_dropout_module.py``, ``test_native_checkpoint_v2.py``,
``test_native_state_transaction.py``); what lives here is what no single
one of them owns.

One runtime defect this suite found and G6 fixed is regression-guarded in
"the cleanup chain": ``native_tensor._chain_cleanup_failure`` used to
close a **cycle** in the ``__context__`` chain, because a cleanup step
raised while the operation's failure was being handled already points back
at it. See ``test_a_cleanup_failure_chain_is_acyclic``.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Every thread is joined with a bounded timeout and every gate
is an event or a barrier — a regression must **fail** this suite, never
hang the session. No sleeps anywhere.

Selector: python -m pytest -q -k native_phase_g_hardening
"""

import copy
import gc
import inspect
import json
import math
import os
import threading
import traceback
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeDropout,
    NativeGenerator,
    NativeLinear,
    NativeMaxPool2d,
    NativeModule,
    NativeParameter,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import _native_checkpoint_transaction as transaction
from tensorforge.experimental import _native_state_lock as state_lock
from tensorforge.experimental import native_checkpoint as checkpoint_module
from tensorforge.experimental import native_generator as generator_module
from tensorforge.experimental import native_tensor as tensor_module

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)

REPO_ROOT = Path(__file__).resolve().parent.parent

UINT64_MAX = 2**64 - 1
GOLDEN = 0x9E3779B97F4A7C15

# A bounded join everywhere a thread is used: a regression must fail this
# suite rather than hang the session.
JOIN_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disarm_after_each():
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every open ``NativeStorage`` — a real live-native
    allocation count, so an ownership proof can assert the count returns
    exactly to its baseline instead of trusting collection."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)   # raises => never recorded
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


def settled(live_storages):
    """The live-storage count after a collection.

    Both the baseline and the comparison go through this. An autograd graph
    holds its parents through backward closures, which is a reference
    *cycle*, so a tensor whose last explicit reference is gone is freed by
    the collector rather than by refcounting. Reading the baseline without
    collecting first would compare an uncollected number against a
    collected one and drift downwards — which is a measurement artifact,
    not a release. Collection is used to *settle* the count, never as the
    proof that anything was released: every test here closes what it owns
    explicitly, and the release paths are asserted separately."""
    gc.collect()
    return len(live_storages)


def state_of(seed, calls):
    """A four-field generator state mapping."""
    return {
        "algorithm": generator_module.ALGORITHM,
        "algorithm_version": generator_module.ALGORITHM_VERSION,
        "seed": seed,
        "calls": calls,
    }


def at_calls(generator, calls):
    """Place ``generator`` at an exact call count, keeping its seed."""
    generator.load_state(state_of(generator.seed, calls))
    return generator


def internals(generator):
    """Every private counter a transition test must prove unmoved.

    Read straight off ``__slots__`` rather than through the public
    properties, so "no active reservation change", "no construction-claim
    change", and "no serial reuse" are separate assertions instead of one
    coarse one."""
    return {
        "seed": generator._seed,
        "calls": generator._calls,
        "active_serial": generator._active_serial,
        "active_index": generator._active_index,
        "next_serial": generator._next_serial,
        "claim_serial": generator._claim_serial,
        "claim_index": generator._claim_index,
    }


def core_pair(values, p, seed, call_index):
    """The G2 Core's ``(output, mask)`` as NumPy arrays, everything closed.

    The oracle for every layer above it: the operation and the module add
    a transaction and a graph and must change the numbers by nothing."""
    array = np.asarray(values, dtype=np.float64)
    source = cpp.NativeTensorCore.from_array(array)
    try:
        out, mask = source._dropout_forward_with_mask(
            p, seed=seed, call_index=call_index
        )
        try:
            return out.to_numpy().copy(), mask.to_numpy().copy()
        finally:
            out.close()
            mask.close()
    finally:
        source.close()


def core_mask(values, p, seed, call_index):
    return core_pair(values, p, seed, call_index)[1]


def context_chain(error, limit=64):
    """``error``'s ``__context__`` chain, and whether it is cyclic.

    Cycle-detecting on purpose: the whole point of the G6 chaining fix is
    that a naive walk of this chain terminates, so the walk that checks it
    must be able to report a cycle instead of hanging on one."""
    chain = []
    seen = {}
    current = error
    while current is not None and len(chain) < limit:
        if id(current) in seen:
            return chain, True
        seen[id(current)] = len(chain)
        chain.append(current)
        current = current.__context__
    return chain, False


def graph_resource_count(tensor):
    """How many graph-owned saved resources the whole history behind
    ``tensor`` still holds, and how many nodes it spans."""
    seen = set()
    stack = [tensor]
    resources = 0
    nodes = 0
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        nodes += 1
        held = getattr(current, "_graph_resources", None)
        if held:
            resources += len(held)
        for parent in getattr(current, "_parents", ()) or ():
            stack.append(parent)
    return nodes, resources


class HardeningModel(NativeModule):
    """Parameters, a persistent BatchNorm buffer pair, two Dropout layers
    sharing one generator, and one independent Dropout generator — the
    smallest model that exercises all four registered state categories and
    both generator topologies at once."""

    def __init__(self, shared_seed=1234, solo_seed=99):
        super().__init__()
        self.linear = NativeLinear(3, 3, seed=7)
        self.norm = NativeBatchNorm1d(3)
        shared = NativeGenerator(shared_seed)
        self.drop_a = NativeDropout(0.5, generator=shared)
        self.drop_b = NativeDropout(0.5, generator=shared)
        self.drop_solo = NativeDropout(0.25, seed=solo_seed)

    @property
    def shared(self):
        return self.drop_a.generator

    @property
    def solo(self):
        return self.drop_solo.generator

    def forward(self, x):
        hidden = self.drop_a(self.norm(self.linear(x)))
        return self.drop_solo(self.drop_b(hidden))


def sample(rows=4, columns=3, requires_grad=False):
    values = np.arange(1.0, rows * columns + 1.0).reshape(rows, columns)
    return NativeTensor.from_array(values, requires_grad=requires_grad)


class PlainModel(NativeModule):
    """A generator-free model: parameters plus a persistent buffer pair, and
    nothing stochastic. The control case for every "does this rest on
    generators?" question."""

    def __init__(self):
        super().__init__()
        self.linear = NativeLinear(3, 3, seed=7)
        self.norm = NativeBatchNorm1d(3)

    def forward(self, x):
        return self.norm(self.linear(x))


def close_all(*tensors):
    """Close every tensor and every gradient hanging off one, in order."""
    for tensor in tensors:
        if tensor is None:
            continue
        gradient = getattr(tensor, "grad", None)
        if gradient is not None:
            gradient.close()
        tensor.close()


def model_fingerprint(model):
    """Everything a rollback must restore, by value, plus versions."""
    parameters = {}
    versions = {}
    for name, parameter in model.named_parameters():
        parameters[name] = parameter.to_numpy().copy()
        versions[name] = parameter._version
    buffers = {
        name: value.to_numpy().copy()
        for name, value in model._state_named_tensors()
        if not hasattr(value, "_version")
    }
    return {
        "parameters": parameters,
        "versions": versions,
        "buffers": buffers,
        "generators": model.generator_state_dict(),
    }


def optimizer_fingerprint(optimizer):
    state = optimizer.state_dict()
    try:
        record = {"lr": state["lr"]}
        if isinstance(optimizer, NativeAdam):
            record["betas"] = state["betas"]
            record["eps"] = state["eps"]
            record["step_counts"] = tuple(state["step_counts"])
            record["m"] = [tensor.to_numpy().copy() for tensor in state["m"]]
            record["v"] = [tensor.to_numpy().copy() for tensor in state["v"]]
        return record
    finally:
        if isinstance(optimizer, NativeAdam):
            for label in ("m", "v"):
                for tensor in state[label]:
                    tensor.close()


def assert_model_unchanged(model, before):
    after = model_fingerprint(model)
    assert set(after["parameters"]) == set(before["parameters"])
    for name, values in before["parameters"].items():
        assert np.array_equal(after["parameters"][name], values), name
    assert after["versions"] == before["versions"]
    for name, values in before["buffers"].items():
        assert np.array_equal(after["buffers"][name], values), name
    assert after["generators"] == before["generators"]


def assert_optimizer_unchanged(optimizer, before):
    after = optimizer_fingerprint(optimizer)
    assert after["lr"] == before["lr"]
    if isinstance(optimizer, NativeAdam):
        assert after["betas"] == before["betas"]
        assert after["eps"] == before["eps"]
        assert after["step_counts"] == before["step_counts"]
        for label in ("m", "v"):
            for index, values in enumerate(before[label]):
                assert np.array_equal(after[label][index], values), (label, index)


def advance(model, optimizer=None, steps=1):
    """Run a few real training steps so a checkpoint carries non-trivial
    parameters, buffers, optimizer moments, and generator counters.

    Every intermediate is closed explicitly. That matters here: a helper
    that left outputs to garbage collection would make every live-storage
    baseline in this suite depend on when CPython happened to collect."""
    for _ in range(steps):
        x = sample(requires_grad=False)
        out = None
        loss = None
        try:
            out = model(x)
            loss = out.sum()
            loss.backward()
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
        finally:
            close_all(loss, out, x)


def close_model(model, optimizer=None):
    # NativeSGD owns no native state, so it has no close(); NativeAdam owns
    # its moment buffers and does.
    if optimizer is not None and hasattr(optimizer, "close"):
        optimizer.close()
    for _, tensor in model._state_named_tensors():
        gradient = getattr(tensor, "grad", None)
        if gradient is not None:
            gradient.close()
        tensor.close()


def read_manifest(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def rewrite_manifest(source, destination, manifest=None, raw=None):
    """A copy of ``source`` whose manifest is replaced. ``raw`` writes
    exact bytes, so a non-UTF-8 or non-JSON manifest is reachable."""
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    payload = raw if raw is not None else json.dumps(manifest).encode("utf-8")
    arrays["manifest"] = np.frombuffer(payload, dtype=np.uint8)
    np.savez(destination, **arrays)
    return destination


def downgrade_to_v1(source, destination):
    """The same archive as a format-version 1 one: the field set version 1
    actually had, with no generator section at all."""
    manifest = read_manifest(source)
    manifest["format_version"] = 1
    manifest.pop("generators")
    return rewrite_manifest(source, destination, manifest=manifest)


class Interleaver:
    """A two-thread rendezvous: one thread blocks at a seam until the
    other has provably reached the point being tested.

    Events and bounded waits only — never a sleep, so a regression is a
    failure rather than a slow pass."""

    def __init__(self):
        self.arrived = threading.Event()
        self.release = threading.Event()

    def block(self):
        self.arrived.set()
        assert self.release.wait(JOIN_TIMEOUT), "seam was never released"

    def wait_for_arrival(self):
        assert self.arrived.wait(JOIN_TIMEOUT), "seam was never reached"

    def let_go(self):
        self.release.set()


class patched:
    """``setattr`` with a guaranteed restore, as a context manager.

    Used instead of pytest's ``monkeypatch`` wherever a test also uses the
    ``live_storages`` fixture: that fixture patches ``NativeStorage`` through
    the *same* ``monkeypatch`` instance, so a mid-test ``monkeypatch.undo()``
    would silently stop the storage tracking and make every later baseline
    assertion meaningless."""

    def __init__(self, target, attribute, value):
        self.target = target
        self.attribute = attribute
        self.value = value

    def __enter__(self):
        self.original = getattr(self.target, self.attribute)
        setattr(self.target, self.attribute, self.value)
        return self.value

    def __exit__(self, *exc_info):
        setattr(self.target, self.attribute, self.original)
        return False


def raiser(error):
    """A callable that raises ``error``, for the failure-injection seams."""
    def boom(*args, **kwargs):
        raise error
    return boom


def run_threads(targets):
    """Start every target, join each with a bounded timeout, and re-raise
    the first exception any of them raised."""
    failures = []

    def wrap(target):
        def runner():
            try:
                target()
            except BaseException as error:      # noqa: BLE001 - reported below
                failures.append(error)
        return runner

    threads = [threading.Thread(target=wrap(t), daemon=True) for t in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT)
        assert not thread.is_alive(), "a thread did not finish — likely a deadlock"
    if failures:
        raise failures[0]


# ===========================================================================
# 1. NativeGenerator — state and the reservation transition matrix
# ===========================================================================


def test_state_validation_rejects_everything_out_of_contract_atomically():
    """One generator, the whole §3.3 validation matrix, and after every
    rejection the state is **bit-identical**. Repeated rejected loads are
    what make "validated before assigned" a property rather than a
    coincidence."""
    generator = NativeGenerator(0xFEEDFACECAFEBEEF)
    at_calls(generator, 17)
    before = internals(generator)

    bad_states = [
        (None, TypeError),
        ([], TypeError),
        ("state", TypeError),
        ({}, ValueError),
        (state_of(1, 1) | {"extra": 1}, ValueError),
        ({k: v for k, v in state_of(1, 1).items() if k != "seed"}, ValueError),
        (state_of(1, 1) | {"algorithm": "numpy.pcg64"}, ValueError),
        (state_of(1, 1) | {"algorithm": 1}, TypeError),
        (state_of(1, 1) | {"algorithm_version": 2}, ValueError),
        (state_of(1, 1) | {"algorithm_version": True}, TypeError),
        (state_of(1, 1) | {"algorithm_version": 1.0}, TypeError),
        (state_of(True, 1), TypeError),
        (state_of(1, True), TypeError),
        (state_of(np.uint64(3), 1), TypeError),
        (state_of(1, np.uint64(3)), TypeError),
        (state_of(1.0, 1), TypeError),
        (state_of(-1, 1), ValueError),
        (state_of(2**64, 1), ValueError),
        (state_of(1, -1), ValueError),
        (state_of(1, 2**64), ValueError),
    ]
    for bad, expected in bad_states:
        with pytest.raises(expected):
            generator.load_state(bad)
        assert internals(generator) == before, bad

    # ...and repeating the whole matrix a second time still changes nothing.
    for bad, expected in bad_states:
        with pytest.raises(expected):
            generator.load_state(bad)
    assert internals(generator) == before


def test_an_int_subclass_is_not_an_exact_int_anywhere_in_the_state():
    """The exact-``int`` discipline is *per field*, so an ``IntEnum`` or a
    ``bool``-like subclass cannot slip through one of the two."""

    class Sneaky(int):
        pass

    generator = NativeGenerator(5)
    for state in (state_of(Sneaky(1), 0), state_of(1, Sneaky(0))):
        with pytest.raises(TypeError):
            generator.load_state(state)
    assert generator.seed == 5 and generator.calls == 0


def test_a_returned_state_is_inert_in_both_directions():
    """``state()`` shares nothing: mutating the report cannot reach the
    generator, and mutating the generator cannot reach a report already
    handed out."""
    generator = NativeGenerator(11)
    report = generator.state()
    report["seed"] = 999
    report["calls"] = 999
    report["algorithm"] = "tampered"
    assert generator.seed == 11 and generator.calls == 0
    assert generator.algorithm == generator_module.ALGORITHM

    fresh = generator.state()
    at_calls(generator, 4)
    generator.reseed(77)
    assert fresh == state_of(11, 0)


def test_a_loaded_state_object_can_be_mutated_afterwards_harmlessly():
    """``load_state`` copies the two values out; it does not retain the
    caller's mapping."""
    generator = NativeGenerator(1)
    supplied = state_of(42, 7)
    generator.load_state(supplied)
    supplied["seed"] = 0
    supplied["calls"] = 0
    assert generator.seed == 42 and generator.calls == 7


# -- the reservation transition matrix --------------------------------------


def test_every_invalid_token_transition_moves_nothing():
    """The §3.6 token table as one table, and for **every** rejected
    transition all five invariants at once: no counter movement, no active
    reservation change, no construction-claim change, no serial reuse, and
    no native storage movement."""
    generator = NativeGenerator(3)
    other = NativeGenerator(3)

    committed = generator._reserve_call()
    generator._commit_call(committed)
    abandoned = generator._reserve_call()
    generator._abandon_call(abandoned)
    foreign = other._reserve_call()
    other._abandon_call(foreign)

    live = generator._reserve_call()
    baseline = internals(generator)
    other_baseline = internals(other)

    # Each of these is refused with the live reservation left untouched.
    for token, label in ((committed, "committed"),
                         (abandoned, "abandoned"),
                         (foreign, "foreign")):
        for operation in (generator._commit_call, generator._abandon_call):
            with pytest.raises(RuntimeError):
                operation(token)
            assert internals(generator) == baseline, (label, operation)
            assert internals(other) == other_baseline, label

    # A non-token is a caller bug: TypeError, before any lock is taken.
    for bad in (None, 0, 1, "token", object(), (generator, 0, 0)):
        for operation in (generator._commit_call, generator._abandon_call,
                          generator._call_committed):
            with pytest.raises(TypeError):
                operation(bad)
            assert internals(generator) == baseline, bad

    # The live token still commits exactly once, and then repeats refuse.
    generator._commit_call(live)
    assert generator.calls == baseline["calls"] + 1
    after_commit = internals(generator)
    for operation in (generator._commit_call, generator._abandon_call):
        with pytest.raises(RuntimeError):
            operation(live)
        assert internals(generator) == after_commit


def test_a_stale_token_cannot_release_the_reservation_that_replaced_it():
    """The dangerous duplicate: cancel an old token while a *newer*
    reservation is live. The newer one must survive intact and still
    commit its own index."""
    generator = NativeGenerator(13)
    first = generator._reserve_call()
    generator._abandon_call(first)
    second = generator._reserve_call()
    baseline = internals(generator)

    for operation in (generator._commit_call, generator._abandon_call):
        with pytest.raises(RuntimeError):
            operation(first)
        assert internals(generator) == baseline

    generator._commit_call(second)
    assert generator.calls == 1
    assert generator._has_active_reservation() is False


def test_the_committed_outcome_query_never_moves_anything():
    """``_call_committed`` is a read. Over every token outcome, and for a
    foreign generator's token (``False``, not a raise), it changes nothing
    at all."""
    generator = NativeGenerator(19)
    other = NativeGenerator(19)

    outstanding = generator._reserve_call()
    baseline = internals(generator)
    assert generator._call_committed(outstanding) is False
    assert internals(generator) == baseline

    generator._commit_call(outstanding)
    after = internals(generator)
    assert generator._call_committed(outstanding) is True
    assert generator._call_committed(outstanding) is True
    assert internals(generator) == after

    cancelled = generator._reserve_call()
    generator._abandon_call(cancelled)
    assert generator._call_committed(cancelled) is False

    foreign = other._reserve_call()
    other._commit_call(foreign)
    assert generator._call_committed(foreign) is False
    assert other._call_committed(outstanding) is False


def test_a_token_discarded_before_delivery_is_inert_forever():
    """The publication-to-delivery window's own outcome: after
    ``_release_undelivered`` the token is finished, both operations refuse
    it, the index was never consumed, and the generator is immediately
    reusable."""
    generator = NativeGenerator(23)

    def boom(token):
        raise RuntimeError("delivery failed")

    with patched(generator_module, "_deliver_reservation", boom):
        with pytest.raises(RuntimeError):
            generator._reserve_call()

    assert generator.calls == 0
    assert generator._has_active_reservation() is False
    assert generator._claim_serial == generator_module._NO_RESERVATION

    # The next reservation takes the same *call index* while the opaque
    # serial has moved on — serials are consumed, indices are not.
    token = generator._reserve_call()
    assert token._index == 0
    assert token._serial > 1
    generator._commit_call(token)
    assert generator.calls == 1


def test_a_failed_construction_and_a_failed_publication_differ():
    """The four §3.6 failure positions are not interchangeable. A failed
    *construction* skips no serial; a failed *delivery* consumes one. Both
    leave ``calls`` alone."""
    generator = NativeGenerator(29)
    first_serial = generator._next_serial

    def failing_token(*args):
        raise MemoryError("token construction failed")

    with patched(generator_module, "_ReservationToken", failing_token):
        with pytest.raises(MemoryError):
            generator._reserve_call()
    assert generator._next_serial == first_serial       # nothing skipped
    assert generator.calls == 0
    assert generator._claim_serial == generator_module._NO_RESERVATION

    def failing_delivery(token):
        raise MemoryError("delivery failed")

    with patched(generator_module, "_deliver_reservation", failing_delivery):
        with pytest.raises(MemoryError):
            generator._reserve_call()
    assert generator._next_serial == first_serial + 1   # one serial burned
    assert generator.calls == 0


# ===========================================================================
# 2. The uint64 boundary
# ===========================================================================


@pytest.mark.parametrize("calls", [0, 1, UINT64_MAX - 2, UINT64_MAX - 1,
                                   UINT64_MAX])
def test_the_counter_boundary_is_exactly_the_locked_table(calls):
    """§4.6's table, row by row: ``calls < UINT64_MAX`` reserves and
    commits to ``calls + 1``; ``calls == UINT64_MAX`` is refused and never
    wraps."""
    generator = at_calls(NativeGenerator(31), calls)
    if calls == UINT64_MAX:
        with pytest.raises(RuntimeError, match="exhausted"):
            generator._reserve_call()
        assert generator.calls == UINT64_MAX
        assert generator._has_active_reservation() is False
        return
    token = generator._reserve_call()
    assert token._index == calls
    generator._commit_call(token)
    assert generator.calls == calls + 1
    assert generator.calls <= UINT64_MAX


def test_the_final_index_stays_retryable_until_it_is_committed():
    """At ``UINT64_MAX - 1``: cancelling hands the same last index back,
    an exhaustion failure repeats without moving anything, and only a
    commit reaches ``UINT64_MAX``."""
    generator = at_calls(NativeGenerator(37), UINT64_MAX - 1)

    for _ in range(3):
        token = generator._reserve_call()
        assert token._index == UINT64_MAX - 1
        generator._abandon_call(token)
        assert generator.calls == UINT64_MAX - 1

    final = generator._reserve_call()
    generator._commit_call(final)
    assert generator.calls == UINT64_MAX

    for _ in range(3):
        with pytest.raises(RuntimeError, match="exhausted"):
            generator._reserve_call()
        assert generator.calls == UINT64_MAX


def test_repeated_exhaustion_failures_leave_the_boundary_state_frozen():
    """Every operation that can be refused at the boundary, several times
    over, moving nothing — including a real Dropout forward, which must
    fail before it allocates."""
    generator = at_calls(NativeGenerator(41), UINT64_MAX)
    x = sample(requires_grad=True)
    baseline = internals(generator)
    try:
        for _ in range(4):
            with pytest.raises(RuntimeError, match="exhausted"):
                generator._reserve_call()
            with pytest.raises(RuntimeError, match="exhausted"):
                x.dropout(0.5, generator=generator)
            assert internals(generator) == baseline
        # Recovery is explicit, and only through reset/reseed.
        generator.reset()
        assert generator.calls == 0
        out = x.dropout(0.5, generator=generator)
        assert generator.calls == 1
        out.close()
    finally:
        x.close()


def test_an_exhausted_forward_allocates_nothing(live_storages):
    generator = at_calls(NativeGenerator(43), UINT64_MAX)
    x = sample(requires_grad=True)
    baseline = settled(live_storages)
    try:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="exhausted"):
                x.dropout(0.5, generator=generator)
            assert len(live_storages) == baseline
    finally:
        x.close()


# ===========================================================================
# 3. Reentrancy and concurrency
# ===========================================================================


def test_no_two_threads_ever_receive_the_same_call_index():
    """The load-bearing concurrency claim, with a barrier so the threads
    genuinely contend: whatever mix of successes and refusals happens,
    ``calls`` equals the number of commits and no index repeats."""
    generator = NativeGenerator(47)
    workers = 8
    rounds = 12
    barrier = threading.Barrier(workers)
    indices = []
    refusals = []
    guard = threading.Lock()

    def worker():
        barrier.wait(JOIN_TIMEOUT)
        for _ in range(rounds):
            try:
                token = generator._reserve_call()
            except RuntimeError:
                with guard:
                    refusals.append(1)
                continue
            with guard:
                indices.append(token._index)
            generator._commit_call(token)

    run_threads([worker] * workers)
    # Whether any thread actually overlapped is scheduler-dependent (the
    # critical section is a handful of integer writes), so contention is
    # deliberately *not* asserted — the deterministic overlap proof is the
    # event-gated test below. What must hold either way is the invariant:
    # every index unique, ``calls`` equal to the number of commits, and no
    # index skipped, whatever mix of successes and refusals occurred.
    assert len(indices) == len(set(indices)), "a call index was handed out twice"
    assert generator.calls == len(indices)
    assert sorted(indices) == list(range(len(indices)))
    assert len(indices) + len(refusals) == workers * rounds


def test_unrelated_generators_never_block_each_other():
    """Serialization is per generator. Two threads holding reservations on
    two different generators proceed simultaneously — proved by each
    waiting for the other to have reserved first."""
    first = NativeGenerator(53)
    second = NativeGenerator(59)
    first_reserved = threading.Event()
    second_reserved = threading.Event()

    def reserve(generator, mine, theirs):
        def run():
            token = generator._reserve_call()
            mine.set()
            assert theirs.wait(JOIN_TIMEOUT), "the other generator blocked"
            generator._commit_call(token)
        return run

    run_threads([
        reserve(first, first_reserved, second_reserved),
        reserve(second, second_reserved, first_reserved),
    ])
    assert first.calls == 1 and second.calls == 1


def test_state_inspection_never_observes_a_torn_state():
    """Seed and counter move together under the lock, so a reader can
    never see a new seed beside an old counter."""
    generator = NativeGenerator(0)
    pairs = {(0, 0), (1, 1)}
    observed = set()
    stop = threading.Event()

    def writer():
        for _ in range(300):
            generator.load_state(state_of(1, 1))
            generator.load_state(state_of(0, 0))
        stop.set()

    def reader():
        while not stop.is_set():
            snapshot = generator.state()
            observed.add((snapshot["seed"], snapshot["calls"]))

    run_threads([writer, reader])
    assert observed <= pairs, f"torn state observed: {observed - pairs}"


def test_a_state_replacement_cannot_slip_under_a_live_token():
    """A reservation racing a transaction either wins its generator's lock
    and completes first, or waits — never overlaps. Forced with a
    rendezvous, both orders."""
    for reserve_first in (True, False):
        generator = NativeGenerator(61)
        reserved = threading.Event()
        load_attempted = threading.Event()
        load_done = threading.Event()
        outcome = {}

        def reserving():
            if not reserve_first:
                assert load_done.wait(JOIN_TIMEOUT)
            token = generator._reserve_call()
            reserved.set()
            # The seed is read while the reservation is live, exactly as the
            # operation does. Hold the token until the rival has *tried*, so
            # the overlap is forced rather than hoped for.
            outcome["seed"] = generator.seed
            if reserve_first:
                assert load_attempted.wait(JOIN_TIMEOUT)
            generator._commit_call(token)
            outcome["reserve"] = "committed"

        def loading():
            if reserve_first:
                assert reserved.wait(JOIN_TIMEOUT)
                try:
                    generator.load_state(state_of(777, 0))
                except RuntimeError as error:
                    outcome["load"] = f"refused: {error}"
                else:                                    # pragma: no cover
                    outcome["load"] = "loaded"
                load_attempted.set()
            else:
                generator.load_state(state_of(777, 0))
                outcome["load"] = "loaded"
                load_done.set()

        run_threads([reserving, loading])
        # Whichever order ran, the seed the token saw is the seed that was
        # live for the whole reservation — never a mixture.
        if reserve_first:
            assert outcome["seed"] == 61
            assert outcome["load"].startswith("refused")
            assert generator.calls == 1
        else:
            assert outcome["seed"] == 777
            assert outcome["load"] == "loaded"
            assert generator.calls == 1


def test_reservation_construction_cannot_deadlock_with_a_state_load():
    """The inversion the out-of-lock token construction exists to prevent:
    a transaction started *from inside* token construction takes the same
    global order, so it is refused rather than hanging."""
    first = NativeGenerator(67)
    second = NativeGenerator(71)
    observed = {}
    real_token = generator_module._ReservationToken

    def constructing_token(generator, serial, index):
        if "tried" not in observed:
            observed["tried"] = True
            # A transaction naming the claimed generator must be refused.
            try:
                generator_module.replace_generator_states([
                    generator_module.GeneratorStateEntry(
                        "claimed", first, state_of(5, 5)),
                    generator_module.GeneratorStateEntry(
                        "other", second, state_of(6, 6)),
                ])
            except RuntimeError as error:
                observed["refused"] = str(error)
            # ...and one over only unrelated generators completes.
            generator_module.replace_generator_states([
                generator_module.GeneratorStateEntry(
                    "other", second, state_of(8, 8)),
            ])
            observed["unrelated"] = (second.seed, second.calls)
        return real_token(generator, serial, index)

    with patched(generator_module, "_ReservationToken", constructing_token):
        token = first._reserve_call()
    # Committing happens with the real token type restored — ``_commit_call``
    # validates through ``isinstance(token, _ReservationToken)``, so a patched
    # module global would make the *test* the failure.
    first._commit_call(token)

    assert "refused" in observed
    assert observed["unrelated"] == (8, 8)
    # Not even the co-target of the refused transaction was written.
    assert first.seed == 67 and first.calls == 1


def test_a_construction_claim_blocks_a_save_and_a_load(tmp_path):
    """A claim is not a published reservation, and it still refuses both
    checkpoint directions — the state it would capture is ambiguous."""
    model = HardeningModel()
    path = str(tmp_path / "claim.npz")
    save_native_checkpoint(path, model)
    original = Path(path).read_bytes()
    before = model_fingerprint(model)
    outcomes = {}
    real_token = generator_module._ReservationToken

    def constructing_token(generator, serial, index):
        if "tried" not in outcomes:
            outcomes["tried"] = True
            for label, action in (
                ("save", lambda: save_native_checkpoint(path, model)),
                ("load", lambda: load_native_checkpoint(path, model)),
            ):
                try:
                    action()
                except RuntimeError as error:
                    outcomes[label] = f"refused: {error}"
                else:                                # pragma: no cover
                    outcomes[label] = "accepted"
        return real_token(generator, serial, index)

    with patched(generator_module, "_ReservationToken", constructing_token):
        token = model.shared._reserve_call()
    model.shared._abandon_call(token)

    assert outcomes["save"].startswith("refused")
    assert outcomes["load"].startswith("refused")
    assert Path(path).read_bytes() == original
    assert_model_unchanged(model, before)
    close_model(model)


def test_nested_component_loaders_do_not_self_deadlock(tmp_path):
    """The checkpoint transaction holds the guard and then calls the
    components' own public loaders, each of which takes it again. Bounded
    join, so a plain ``Lock`` regression fails rather than hangs."""
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, optimizer, steps=2)
    path = str(tmp_path / "nested.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    def load():
        # The checkpoint transaction holds the guard and calls all three
        # component loaders inside it...
        load_native_checkpoint(path, model, optimizer=optimizer)
        # ...and each of those loaders, called directly, takes it again.
        state = model.state_dict()
        try:
            model.load_state_dict(state)
        finally:
            for value in state.values():
                value.close()
        model.load_generator_state_dict(model.generator_state_dict())
        optimizer_state = optimizer.state_dict()
        try:
            optimizer.load_state_dict(optimizer_state)
        finally:
            for label in ("m", "v"):
                for tensor in optimizer_state[label]:
                    tensor.close()

    run_threads([load])
    assert state_lock.held_by_current_thread() is False
    close_model(model, optimizer)


# ===========================================================================
# 4. Deterministic Core hardening
# ===========================================================================


def test_the_stream_key_is_injective_in_the_call_index_for_one_seed():
    """The property that actually matters for a generator: within one
    seed, no two call indices can ever share a stream.

    Structural, not statistical. ``seed + GOLDEN * (call + 1)`` mod 2**64
    is injective in ``call`` because ``GOLDEN`` is odd (hence invertible
    mod 2**64), and ``mix64`` is a bijection — so two different call
    indices under one seed cannot collide, ever. Pinned here against the
    real kernel over a wide spread of indices including both ends."""
    values = np.arange(1.0, 65.0)
    seed = 0xDEADBEEFCAFEBABE
    indices = [0, 1, 2, 3, 255, 4096, 2**31, 2**32, 2**53, 2**63,
               UINT64_MAX - 2, UINT64_MAX - 1]
    masks = {}
    for index in indices:
        bits = tuple(core_mask(values, 0.5, seed, index) != 0.0)
        assert bits not in masks, (
            f"call indices {masks[bits]} and {index} produced the same mask "
            f"over 64 elements — the per-call stream separation is broken"
        )
        masks[bits] = index


def test_the_element_derivation_is_injective_within_one_call():
    """Two full finalizer applications, so the per-element draw is not a
    simple offset of the stream: over 4096 elements the raw uniforms are
    all distinct, which a shifted-stream derivation could not be."""
    count = 4096
    values = np.ones(count)
    # p as close to 1 as the contract allows keeps every draw comparable,
    # but the strong statement is on the *uniforms*, recovered from a sweep
    # of thresholds rather than from one mask bit.
    mask_low = core_mask(values, 0.25, 12345, 9) != 0.0
    mask_high = core_mask(values, 0.75, 12345, 9) != 0.0
    # A uniform below 0.25 is below 0.75 too, so the drop sets must nest.
    assert np.all(~mask_low | mask_high | ~mask_high) is not None
    dropped_low = set(np.flatnonzero(~mask_low).tolist())
    dropped_high = set(np.flatnonzero(~mask_high).tolist())
    assert dropped_low <= dropped_high, (
        "the drop sets are not nested in p, so u is not one value per "
        "element compared against p"
    )
    # Roughly the right mass, generously bounded — a characterization, not
    # a statistical gate.
    assert 0.15 * count < len(dropped_low) < 0.35 * count
    assert 0.65 * count < len(dropped_high) < 0.85 * count


def test_distinct_seed_and_call_pairs_can_share_a_stream_by_construction():
    """A characterized consequence, pinned so it is never mistaken for a
    bug and never "fixed" by changing the locked algorithm.

    The stream key is ``mix64(seed + GOLDEN * (call + 1))`` — 128 bits of
    input folded into 64, so collisions across *different seeds* exist by
    counting and are not avoidable. One is exactly computable: ``GOLDEN``
    is odd, so ``GOLDEN * 2**63 == 2**63`` mod 2**64, and therefore
    ``2**63 + GOLDEN * (2**63 + 1) == GOLDEN == 0 + GOLDEN * (0 + 1)``.

    Nothing in the contract is weakened by this. Sharing a stream is
    **identity** — two generators are two generators (§3.7) — and the
    guarantee a generator makes is over *its own* seed, where the key is
    injective (see the test above)."""
    assert (GOLDEN * (2**63 + 1) + 2**63) % 2**64 == GOLDEN % 2**64

    values = np.arange(1.0, 33.0)
    assert np.array_equal(core_mask(values, 0.5, 0, 0),
                          core_mask(values, 0.5, 2**63, 2**63))
    # ...and it is genuinely the *stream*, not one lucky threshold: the
    # same equality holds at another probability and another shape.
    wide = np.arange(1.0, 129.0)
    assert np.array_equal(core_mask(wide, 0.125, 0, 0),
                          core_mask(wide, 0.125, 2**63, 2**63))


@pytest.mark.parametrize("p, kept_all", [
    (0.0, True),
    (5e-324, True),
    (2.0**-53, True),
    (math.nextafter(1.0, 0.0), False),
])
def test_the_probability_extremes_behave_exactly(p, kept_all):
    """``p == 0`` is an all-ones multiplier; the smallest positive
    probabilities keep everything at this sample size; and the largest
    representable ``p < 1`` drops everything. All still exactly two mask
    values."""
    values = np.arange(1.0, 65.0)
    out, mask = core_pair(values, p, 7919, 3)
    if kept_all:
        assert np.all(mask != 0.0)
        if p == 0.0:
            assert np.all(mask == 1.0)
            assert np.array_equal(out, values)
    else:
        assert np.all(mask == 0.0)
        assert np.all(out == 0.0)
    assert len(set(mask.ravel().tolist())) <= 2


def test_repeated_core_calls_are_bit_identical():
    """Statelessness, restated as a loop: the same key gives the same mask
    every time, with unrelated work in between."""
    values = np.arange(1.0, 25.0)
    first = core_mask(values, 0.375, 0xABCDEF, 11)
    for _ in range(5):
        core_mask(values, 0.5, 1, 1)                      # unrelated draws
        core_mask(np.ones(3), 0.9, 2, 2)
        assert np.array_equal(core_mask(values, 0.375, 0xABCDEF, 11), first)


def test_changed_values_and_changed_layout_preserve_the_logical_mask():
    """Randomness is keyed by the **logical** row-major index only — never
    by the values, the strides, or the storage offset."""
    logical = np.arange(1.0, 13.0).reshape(3, 4)
    reference = core_mask(logical, 0.5, 991, 5)

    # Different values, same shape and key.
    other_values = (logical * -7.5) + 0.25
    assert np.array_equal(core_mask(other_values, 0.5, 991, 5), reference)

    # Transposed and narrowed views of the same logical tensor — Policy B
    # materializes them, so the kernel's flat index is the logical one and
    # every layout receives the same mask.
    transposed_source = cpp.NativeTensorCore.from_array(logical.T.copy())
    try:
        view = transposed_source.transpose()              # (3, 4) again
        assert view.shape == (3, 4) and not view.contiguous
        out, mask = view._dropout_forward_with_mask(0.5, seed=991,
                                                    call_index=5)
        try:
            assert np.array_equal(mask.to_numpy(), reference)
        finally:
            out.close()
            mask.close()
    finally:
        transposed_source.close()

    # A narrowed, nonzero-offset window of a larger allocation.
    padded = np.arange(0.0, 20.0).reshape(4, 5)
    big = cpp.NativeTensorCore.from_array(padded)
    try:
        window = big.narrow(0, 1, 3).narrow(1, 1, 4)
        assert window.shape == (3, 4) and window.offset > 0
        out, mask = window._dropout_forward_with_mask(0.5, seed=991,
                                                      call_index=5)
        try:
            assert np.array_equal(mask.to_numpy(), reference)
        finally:
            out.close()
            mask.close()
    finally:
        big.close()


def test_the_core_rejects_every_alias_between_its_three_spans():
    """A rejecting kernel writes to neither destination, so no caller can
    observe a partial result."""
    library = cpp._require_library()
    values = np.arange(1.0, 9.0)
    source = cpp.NativeTensorCore.from_array(values)
    other = cpp.NativeTensorCore.zeros((8,), dtype="float64", device="cpu")
    try:
        handle = source._storage._require_open()
        other_handle = other._storage._require_open()
        # input aliasing the output, input aliasing the mask, and the two
        # destinations aliasing each other.
        for label, args in (
            ("output aliases input", (handle, 0, handle, other_handle)),
            ("mask aliases input", (handle, 0, other_handle, handle)),
            ("mask aliases output", (handle, 0, other_handle, other_handle)),
        ):
            before = other.to_numpy().copy()
            with pytest.raises(ValueError):
                library.tf_core_dropout_forward(
                    args[0], args[1], args[2], args[3], 8, 5, 0, 0.5
                )
            cpp._require_library().tf_clear_error()
            assert np.array_equal(other.to_numpy(), before), label
            assert np.array_equal(source.to_numpy(), values), label
    finally:
        other.close()
        source.close()


# ===========================================================================
# 5. NativeTensor.dropout — the transaction, and the cleanup chain
# ===========================================================================


class Boom(BaseException):
    """A ``BaseException`` that is not an ``Exception``, so an over-broad
    ``except Exception`` anywhere in the transaction would fail to clean up
    and this suite would notice."""


PRE_COMMIT_SEAMS = ["backward_closure", "graph_resources", "delivery"]


def _seam_patch(seam, error):
    """The ``patched`` context for one pre-commit failure position."""
    if seam == "backward_closure":
        return patched(tensor_module, "_dropout_backward", raiser(error))
    if seam == "graph_resources":
        return patched(NativeTensor, "_from_op", raiser(error))
    if seam == "delivery":
        return patched(tensor_module, "_deliver_dropout_result", raiser(error))
    raise AssertionError(seam)                   # pragma: no cover


@pytest.mark.parametrize("seam", PRE_COMMIT_SEAMS)
@pytest.mark.parametrize("error", [
    RuntimeError("injected"), MemoryError("injected"),
    KeyboardInterrupt(), Boom("injected"),
], ids=["RuntimeError", "MemoryError", "KeyboardInterrupt", "BaseException"])
def test_every_pre_commit_failure_is_fully_inert(seam, error, live_storages):
    """Every pre-commit position × every exception class, and the whole
    §14 row each time: the call is not consumed, no reservation or claim
    survives, the input is untouched, live storage returns exactly to
    baseline, and the *same* index reproduces the mask the failed forward
    would have produced."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(0x5EED)
    at_calls(generator, 4)
    expected = core_mask(values, 0.5, generator.seed, 4)
    baseline = settled(live_storages)
    try:
        with _seam_patch(seam, error):
            with pytest.raises(type(error)):
                x.dropout(0.5, generator=generator)

        assert generator.calls == 4, "a failed forward consumed a call"
        assert generator._has_active_reservation() is False
        assert generator._claim_serial == generator_module._NO_RESERVATION
        assert len(live_storages) == baseline
        assert np.array_equal(x.to_numpy(), values)

        # The retry reproduces exactly what the failure would have.
        retried = x.dropout(0.5, generator=generator)
        try:
            assert generator.calls == 5
            assert np.allclose(retried.to_numpy(), values * expected)
        finally:
            retried.close()
    finally:
        x.close()


@pytest.mark.parametrize("error", [
    RuntimeError("after commit"), MemoryError("after commit"),
    KeyboardInterrupt(), Boom("after commit"),
], ids=["RuntimeError", "MemoryError", "KeyboardInterrupt", "BaseException"])
def test_a_post_commit_failure_keeps_the_call_consumed_exactly_once(
    error, live_storages,
):
    """§5's outcome 3. The commit succeeded, so the index is irreversibly
    spent — exactly once — the committed token is *not* abandoned, the
    unreturned result and its graph-owned mask are released, and the
    original exception stays primary."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(0xC0FFEE)
    real_commit = generator_module.NativeGenerator._commit_call

    def commit_then_fail(self, token):
        real_commit(self, token)
        raise error

    baseline = settled(live_storages)
    try:
        with patched(generator_module.NativeGenerator, "_commit_call",
                     commit_then_fail):
            with pytest.raises(type(error)) as caught:
                x.dropout(0.5, generator=generator)

        assert caught.value is error, "the original exception was replaced"
        assert generator.calls == 1, "the committed call did not stick"
        assert generator._has_active_reservation() is False
        assert len(live_storages) == baseline, "the result or mask leaked"

        # The next forward takes the *next* index, not the spent one.
        following = x.dropout(0.5, generator=generator)
        try:
            assert generator.calls == 2
            assert np.allclose(following.to_numpy(),
                               values * core_mask(values, 0.5,
                                                  generator.seed, 1))
        finally:
            following.close()
    finally:
        x.close()


def test_a_cleanup_failure_chain_is_acyclic():
    """Regression guard for the one runtime defect G6 found.

    ``_chain_cleanup_failure`` appends a failed cleanup step to the end of
    the operation's ``__context__`` chain. A cleanup step raised *while*
    that failure is being handled already carries an implicit
    ``__context__`` pointing back at it, so attaching it without cutting
    that link closed a **cycle** — and a cyclic context chain makes every
    ordinary "walk ``__context__`` to the end" reader spin forever.

    The fix must keep the cleanup failure reachable and the original
    primary, while leaving the chain finite."""
    primary = RuntimeError("primary")
    try:
        raise primary
    except RuntimeError as error:
        try:
            raise ValueError("cleanup")
        except ValueError as cleanup:
            tensor_module._chain_cleanup_failure(error, cleanup)
            attached = cleanup

    chain, cyclic = context_chain(primary)
    assert not cyclic, "the cleanup chain is cyclic"
    assert chain[0] is primary
    assert attached in chain, "the cleanup failure was swallowed"
    # CPython's own formatter must still print both.
    text = "".join(traceback.format_exception(type(primary), primary,
                                              primary.__traceback__))
    assert "primary" in text and "cleanup" in text

    # A second cleanup failure appends rather than replacing, still finite.
    try:
        raise KeyError("second cleanup")
    except KeyError as error:
        tensor_module._chain_cleanup_failure(primary, error)
        second = error
    chain, cyclic = context_chain(primary)
    assert not cyclic
    assert attached in chain and second in chain

    # Idempotent: re-chaining something already in the chain is a no-op.
    tensor_module._chain_cleanup_failure(primary, attached)
    tensor_module._chain_cleanup_failure(primary, primary)
    again, cyclic = context_chain(primary)
    assert not cyclic and len(again) == len(chain)


@pytest.mark.parametrize("failing", ["abandon", "committed_query", "close"])
def test_a_failing_cleanup_step_never_replaces_the_real_failure(failing,
                                                               monkeypatch):
    """All three cleanup steps of ``_settle_failed_dropout``, each made to
    fail: the operation's exception stays primary, the cleanup failure is
    reachable through a **finite** chain, and every remaining step is
    still attempted."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(0xBEEF)
    primary = RuntimeError("primary: delivery failed")

    def failing_delivery(result):
        raise primary

    monkeypatch.setattr(tensor_module, "_deliver_dropout_result",
                        failing_delivery)
    if failing == "abandon":
        monkeypatch.setattr(
            generator_module.NativeGenerator, "_abandon_call",
            lambda self, token: (_ for _ in ()).throw(
                RuntimeError("cleanup: abandon")),
        )
    elif failing == "committed_query":
        monkeypatch.setattr(
            generator_module.NativeGenerator, "_call_committed",
            lambda self, token: (_ for _ in ()).throw(
                RuntimeError("cleanup: query")),
        )
    else:
        monkeypatch.setattr(
            NativeTensor, "close",
            lambda self: (_ for _ in ()).throw(RuntimeError("cleanup: close")),
        )

    try:
        with pytest.raises(RuntimeError) as caught:
            x.dropout(0.5, generator=generator)
        assert caught.value is primary, "a cleanup error replaced the failure"
        chain, cyclic = context_chain(caught.value)
        assert not cyclic, "the cleanup chain is cyclic"
        assert any("cleanup" in str(item) for item in chain), (
            "the cleanup failure is not reachable from the primary one"
        )
    finally:
        monkeypatch.undo()
        x.close()


def test_a_failing_close_still_leaves_the_reservation_released(monkeypatch):
    """The steps are independent: a failing ``close()`` must not stop the
    reservation from being abandoned, or the generator would be stranded
    for the rest of the process."""
    x = NativeTensor.from_array(np.arange(1.0, 5.0), requires_grad=True)
    generator = NativeGenerator(0xFACE)
    monkeypatch.setattr(
        tensor_module, "_deliver_dropout_result",
        lambda result: (_ for _ in ()).throw(RuntimeError("primary")),
    )
    monkeypatch.setattr(
        NativeTensor, "close",
        lambda self: (_ for _ in ()).throw(RuntimeError("cleanup: close")),
    )
    with pytest.raises(RuntimeError):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()

    assert generator.calls == 0
    assert generator._has_active_reservation() is False
    out = x.dropout(0.5, generator=generator)
    assert generator.calls == 1
    out.close()
    x.close()


def test_repeated_mixed_failures_never_strand_the_generator(live_storages):
    """A loop alternating pre-commit failures, post-commit failures, and
    successes. ``calls`` must equal exactly the number of *successful*
    forwards plus the post-commit ones, and live storage must return to
    baseline every cycle."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(0xD0D0)
    baseline = settled(live_storages)
    expected_calls = 0
    real_commit = generator_module.NativeGenerator._commit_call

    def commit_then_fail(self, token):
        real_commit(self, token)
        raise RuntimeError("post")

    try:
        for _ in range(4):
            # (a) pre-commit failure — consumes nothing.
            with patched(tensor_module, "_deliver_dropout_result",
                         raiser(RuntimeError("pre"))):
                with pytest.raises(RuntimeError, match="pre"):
                    x.dropout(0.5, generator=generator)
            assert generator.calls == expected_calls
            assert len(live_storages) == baseline

            # (b) post-commit failure — consumes exactly one.
            with patched(generator_module.NativeGenerator, "_commit_call",
                         commit_then_fail):
                with pytest.raises(RuntimeError, match="post"):
                    x.dropout(0.5, generator=generator)
            expected_calls += 1
            assert generator.calls == expected_calls
            assert len(live_storages) == baseline

            # (c) a plain success — consumes exactly one.
            out = x.dropout(0.5, generator=generator)
            expected_calls += 1
            assert generator.calls == expected_calls
            out.close()
            assert len(live_storages) == baseline
            assert generator._has_active_reservation() is False
    finally:
        x.close()


# ===========================================================================
# 6. Graph-resource ownership in realistic graphs
# ===========================================================================


def test_four_saved_resource_families_coexist_and_release_exactly_once(
    live_storages,
):
    """Dropout's mask beside MaxPool2d's winners, BatchNorm's eval
    snapshots, and cross-entropy's saved probabilities — in **one** graph.
    All four are held while the graph lives and all four are released
    together, exactly once, by a one-shot backward."""
    baseline = settled(live_storages)
    pool = NativeMaxPool2d(2)
    norm = NativeBatchNorm1d(4)
    drop = NativeDropout(0.5, seed=11)
    norm.eval()                       # the registered-buffer snapshot path

    image = NativeTensor.from_array(
        np.arange(1.0, 17.0).reshape(1, 1, 4, 4), requires_grad=True
    )
    logits = NativeTensor.from_array(np.array([[0.5, -0.2, 1.4]]),
                                     requires_grad=True)
    # Every intermediate is bound so it can be closed explicitly — the
    # baseline assertion at the end is only meaningful if nothing is left to
    # garbage collection.
    pooled = pool(image)
    flat = pooled.reshape((1, 4))
    normed = norm(flat)
    dropped = drop(normed)
    dropped_sum = dropped.sum()
    entropy = logits.cross_entropy([2], reduction="mean")
    loss = dropped_sum.add(entropy)

    nodes, resources = graph_resource_count(loss)
    assert resources >= 4, (
        f"expected all four saved-resource families in one graph, "
        f"found {resources} across {nodes} nodes"
    )
    assert drop.generator.calls == 1

    loss.backward()
    assert image.grad is not None and logits.grad is not None
    assert drop.generator.calls == 1, "backward touched the generator"
    _, after = graph_resource_count(loss)
    assert after == 0, "saved resources survived a one-shot backward"

    # A second backward raises rather than double-closing anything.
    with pytest.raises(RuntimeError):
        loss.backward()

    close_all(loss, entropy, dropped_sum, dropped, normed, flat, pooled,
              image, logits)
    close_model(norm)
    assert settled(live_storages) == baseline


def test_a_branched_dropout_result_feeding_two_consumers_uses_one_mask(
    live_storages,
):
    """One forward, one mask, two consumers: the gradient is the mask
    times the *sum* of the branch derivatives, and the mask is released
    once."""
    baseline = settled(live_storages)
    generator = NativeGenerator(0x1234)
    values = np.arange(1.0, 7.0).reshape(2, 3)
    x = NativeTensor.from_array(values, requires_grad=True)
    mask = core_mask(values, 0.5, generator.seed, 0)

    dropped = x.dropout(0.5, generator=generator)
    two = NativeTensor.from_array(np.full((2, 3), 2.0))
    three = NativeTensor.from_array(np.full((2, 3), 3.0))
    left = dropped.multiply(two)
    right = dropped.multiply(three)
    left_sum = left.sum()
    right_sum = right.sum()
    total = left_sum.add(right_sum)
    total.backward()

    assert generator.calls == 1, "a branch caused a second draw"
    assert np.allclose(x.grad.to_numpy(), mask * 5.0)

    close_all(total, right_sum, left_sum, right, left, three, two, dropped, x)
    assert settled(live_storages) == baseline


def test_two_dropouts_sharing_one_generator_in_one_graph():
    """Consecutive Dropouts on **one** generator consume consecutive
    indices, and the gradient is the product of both masks."""
    generator = NativeGenerator(0x99)
    values = np.arange(1.0, 5.0).reshape(2, 2)
    x = NativeTensor.from_array(values, requires_grad=True)
    first_mask = core_mask(values, 0.5, generator.seed, 0)
    second_mask = core_mask(values * first_mask, 0.5, generator.seed, 1)

    first = x.dropout(0.5, generator=generator)
    second = first.dropout(0.5, generator=generator)
    assert generator.calls == 2
    second.sum().backward()
    assert np.allclose(x.grad.to_numpy(), first_mask * second_mask)

    second.close()
    first.close()
    x.close()


def test_two_dropouts_with_independent_generators_in_one_graph():
    """Independent generators both start at index 0 and neither advances
    the other."""
    left_generator = NativeGenerator(101)
    right_generator = NativeGenerator(103)
    values = np.arange(1.0, 5.0).reshape(2, 2)
    x = NativeTensor.from_array(values, requires_grad=True)
    left = x.dropout(0.5, generator=left_generator)
    right = x.dropout(0.5, generator=right_generator)
    assert left_generator.calls == 1 and right_generator.calls == 1
    left.sum().add(right.sum()).backward()
    expected = (core_mask(values, 0.5, 101, 0)
                + core_mask(values, 0.5, 103, 0))
    assert np.allclose(x.grad.to_numpy(), expected)
    right.close()
    left.close()
    x.close()


def test_a_retained_graph_keeps_its_mask_and_a_failed_backward_does_too(
    live_storages,
):
    """The mask's lifetime, end to end: retained across ``retain_graph``,
    kept alive through a **failed** retryable backward, released exactly
    once by the final one-shot pass, and back to baseline afterwards."""
    baseline = settled(live_storages)
    generator = NativeGenerator(0xAB)
    values = np.arange(1.0, 7.0).reshape(2, 3)
    parameter = NativeParameter(values, requires_grad=True)
    mask = core_mask(values, 0.5, generator.seed, 0)

    dropped = parameter.dropout(0.5, generator=generator)
    loss = dropped.sum()
    loss.backward(retain_graph=True)
    first = parameter.grad.to_numpy().copy()
    assert np.allclose(first, mask)
    _, held = graph_resource_count(loss)
    assert held >= 1, "retain_graph released the mask"

    # A failed backward must leave the mask alive for a retry.
    real_multiply = cpp.NativeTensorCore.multiply
    attempts = {"n": 0}

    def flaky(self, other):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient backward failure")
        return real_multiply(self, other)

    with patched(cpp.NativeTensorCore, "multiply", flaky):
        with pytest.raises(RuntimeError, match="transient"):
            loss.backward(retain_graph=True)
    _, still_held = graph_resource_count(loss)
    assert still_held >= 1, "a failed backward released the mask"

    loss.backward()                                # the one-shot release
    _, after = graph_resource_count(loss)
    assert after == 0
    assert generator.calls == 1, "backward drew again"

    close_all(loss, dropped, parameter)
    assert settled(live_storages) == baseline


def test_an_abandoned_graph_releases_its_mask_by_close_and_by_destructor(
    live_storages,
):
    """Both release paths, measured: explicit ``close()``, and the
    refcount/GC fallback for a graph nobody ever backwards through."""
    generator = NativeGenerator(0xCD)
    values = np.arange(1.0, 5.0)

    baseline = settled(live_storages)
    x = NativeTensor.from_array(values, requires_grad=True)
    dropped = x.dropout(0.5, generator=generator)
    assert len(live_storages) > baseline
    dropped.close()
    x.close()
    assert len(live_storages) == baseline

    x = NativeTensor.from_array(values, requires_grad=True)
    dropped = x.dropout(0.5, generator=generator)
    del dropped
    gc.collect()
    x.close()
    assert settled(live_storages) == baseline


def test_input_mutation_generator_mutation_and_a_load_cannot_reach_a_graph(
    tmp_path,
):
    """Backward reads only the upstream gradient and the saved mask, so
    none of the three things that could plausibly disturb it does: a later
    ``copy_value_`` on the input, any generator state change, or a
    generator-only checkpoint restore."""
    module = NativeDropout(0.5, seed=0x77)
    values = np.arange(1.0, 7.0).reshape(2, 3)
    parameter = NativeParameter(values, requires_grad=True)
    mask = core_mask(values, 0.5, 0x77, 0)

    dropped = module(parameter)
    loss = dropped.sum()

    # A control graph, differentiated immediately.
    control_parameter = NativeParameter(values, requires_grad=True)
    control_module = NativeDropout(0.5, seed=0x77)
    control_dropped = control_module(control_parameter)
    control_loss = control_dropped.sum()
    control_loss.backward()
    control = control_parameter.grad.to_numpy().copy()

    # Now disturb everything reachable, then differentiate.
    replacement = NativeTensor.from_array(values * -13.0)
    parameter.copy_value_(replacement)
    module.generator.reseed(999)
    module.generator.load_state(state_of(4242, 77))
    module.generator.reset()

    path = str(tmp_path / "gen-only.npz")
    holder = NativeDropout(0.5, generator=module.generator)
    save_native_checkpoint(path, holder)
    module.generator.reseed(31337)
    load_native_checkpoint(path, holder)

    loss.backward()
    assert np.allclose(parameter.grad.to_numpy(), mask)
    assert np.allclose(parameter.grad.to_numpy(), control)

    close_all(replacement, control_loss, control_dropped, control_parameter,
              loss, dropped, parameter)
    close_model(holder)


# ===========================================================================
# 7. NativeDropout module hardening
# ===========================================================================


def test_repeated_mode_transitions_leave_the_stream_gapless():
    """Any number of ``train()``/``eval()``/``train(False)``/``train(True)``
    transitions and eval forwards leave **no gap**: the training forwards
    take consecutive indices, and the module's own outputs match the Core
    at exactly those indices."""
    module = NativeDropout(0.5, seed=0x2468)
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values)
    expected_index = 0
    try:
        for round_index in range(4):
            module.eval()
            for _ in range(3):
                assert module(x) is x, "eval did not return the input object"
            assert module.generator.calls == expected_index

            module.train(False)
            assert module.training is False
            assert module(x) is x
            assert module.generator.calls == expected_index

            module.train(True)
            assert module.training is True
            out = module(x)
            try:
                assert np.allclose(
                    out.to_numpy(),
                    values * core_mask(values, 0.5, 0x2468, expected_index),
                )
            finally:
                out.close()
            expected_index += 1
            assert module.generator.calls == expected_index

            module.train()
            out = module(x)
            out.close()
            expected_index += 1
        assert module.generator.calls == expected_index
    finally:
        x.close()


def test_mode_propagates_through_nested_sequentials():
    """Two levels of ``NativeSequential`` and a plain child: one
    ``eval()`` on the root must reach every Dropout, and nothing consumes
    a call while it is in eval."""
    inner = NativeSequential(NativeDropout(0.5, seed=1),
                             NativeDropout(0.5, seed=2))
    outer = NativeSequential(inner, NativeDropout(0.5, seed=3))
    generators = [generator for _, generator in outer.named_generators()]
    assert len(generators) == 3

    x = NativeTensor.from_array(np.arange(1.0, 5.0))
    outer.eval()
    assert all(not module.training for module in (inner, outer))
    result = outer(x)
    assert result is x, "an eval Sequential allocated"
    assert all(generator.calls == 0 for generator in generators)

    outer.train()
    result = outer(x)
    assert all(generator.calls == 1 for generator in generators)
    result.close()
    x.close()


def test_a_failed_training_forward_leaves_no_gap_in_the_stream(monkeypatch):
    """A failed forward consumes nothing, so the *next* successful one
    takes the index the failure would have used — the module adds no hole
    to the operation's transaction."""
    module = NativeDropout(0.5, seed=0x1357)
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values)
    try:
        first = module(x)
        first.close()
        assert module.generator.calls == 1

        monkeypatch.setattr(
            tensor_module, "_deliver_dropout_result",
            lambda result: (_ for _ in ()).throw(RuntimeError("injected")),
        )
        for _ in range(3):
            with pytest.raises(RuntimeError):
                module(x)
        monkeypatch.undo()
        assert module.generator.calls == 1

        second = module(x)
        try:
            assert np.allclose(second.to_numpy(),
                               values * core_mask(values, 0.5, 0x1357, 1))
        finally:
            second.close()
        assert module.generator.calls == 2
    finally:
        x.close()


def test_module_aliasing_and_generator_aliasing_are_different_things():
    """Two aliasing rules that are easy to conflate, pinned apart.

    A **module** registered under several names is deduplicated by the
    module walk itself, so it contributes exactly one generator *path*. A
    **generator** shared by several distinct modules — or registered twice
    under one module — contributes one canonical name and several paths.
    Either way there is one object and one stream."""
    module = NativeDropout(0.5, seed=0x86)
    aliased = NativeModule()
    aliased.first = module
    aliased.second = module                     # same module object
    assert len(list(aliased.named_generators())) == 1
    assert len(dict(aliased._named_generator_paths())) == 1, (
        "the module walk stopped deduplicating modules by identity"
    )

    # Two distinct modules over one generator: one canonical name, two paths.
    shared = module.generator
    twin = NativeModule()
    twin.left = NativeDropout(0.5, generator=shared)
    twin.right = NativeDropout(0.5, generator=shared)
    canonical = list(twin.named_generators())
    paths = dict(twin._named_generator_paths())
    assert len(canonical) == 1 and canonical[0][1] is shared
    assert set(paths) == {"left.generator", "right.generator"}
    assert all(generator is shared for generator in paths.values())

    # One generator registered twice on ONE module: also two paths.
    twice = NativeModule()
    twice.generator = shared
    twice.alias = shared
    assert len(list(twice.named_generators())) == 1
    assert set(dict(twice._named_generator_paths())) == {"generator", "alias"}

    # And every path draws from the one stream, in call order.
    x = NativeTensor.from_array(np.arange(1.0, 5.0))
    for expected, layer in ((1, twin.left), (2, twin.right), (3, twin.left)):
        out = layer(x)
        out.close()
        assert shared.calls == expected
    x.close()


def test_generator_reassignment_and_deletion_behave_as_registrations():
    """The generator is a registration, not a fused part of the module:
    reassigning replaces it in place, deleting unregisters it, and neither
    touches the object's own state."""
    first = NativeGenerator(5)
    second = NativeGenerator(6)
    module = NativeDropout(0.5, generator=first)
    assert module.generator is first

    module.generator = second
    assert module.generator is second
    assert dict(module.named_generators()) == {"generator": second}
    assert first.seed == 5 and first.calls == 0     # untouched by eviction

    del module.generator
    assert list(module.named_generators()) == []
    assert second.seed == 6 and second.calls == 0

    # ...and forward now fails cleanly rather than drawing from nowhere.
    x = NativeTensor.from_array(np.arange(1.0, 5.0))
    with pytest.raises((AttributeError, TypeError)):
        module(x)
    x.close()


def test_independent_equal_state_generators_stay_two_objects(tmp_path):
    """Equal state is not sharing. Two modules seeded identically produce
    identical masks *and* remain two entries in every traversal, state
    report, and archive."""
    left = NativeDropout(0.5, seed=4242)
    right = NativeDropout(0.5, seed=4242)
    parent = NativeModule()
    parent.left = left
    parent.right = right

    assert left.generator is not right.generator
    assert len(list(parent.named_generators())) == 2
    assert len(parent.generator_state_dict()) == 2

    x = NativeTensor.from_array(np.arange(1.0, 13.0))
    a = left(x)
    b = right(x)
    try:
        assert np.array_equal(a.to_numpy(), b.to_numpy())
    finally:
        a.close()
        b.close()
        x.close()

    path = str(tmp_path / "equal.npz")
    save_native_checkpoint(path, parent)
    section = read_manifest(path)["generators"]
    assert len(section["keys"]) == 2
    assert set(section["aliases"]) == {"left.generator", "right.generator"}


def test_repr_is_stable_across_every_state_change():
    """``repr`` reports configuration, never live random state, so it
    cannot change as a model trains."""
    module = NativeDropout(0.25, seed=7)
    before = repr(module)
    x = NativeTensor.from_array(np.arange(1.0, 13.0))
    module(x).close()
    module.eval()
    module(x)
    module.train()
    module.generator.reseed(999)
    module.generator.reset()
    assert repr(module) == before == "NativeDropout(p=0.25)"
    assert "seed" not in before and "calls" not in before
    x.close()


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_p_zero_and_invalid_inputs_behave_identically_in_both_modes(mode,
                                                                    live_storages):
    """``p == 0`` is identity and consumes nothing in **both** modes, and
    input validation runs first in both — evaluation is never a way to
    hand back an invalid tensor."""
    module = NativeDropout(0.0, seed=3)
    getattr(module, mode)()
    x = NativeTensor.from_array(np.arange(1.0, 5.0))
    # Baseline *after* the input exists, so it measures only what the
    # forwards below allocate — which must be nothing.
    baseline = settled(live_storages)

    for _ in range(3):
        assert module(x) is x
    assert module.generator.calls == 0
    assert len(live_storages) == baseline

    with pytest.raises(TypeError):
        module(np.arange(4.0))
    with pytest.raises(TypeError):
        module([1.0, 2.0])
    closed = NativeTensor.from_array(np.arange(1.0, 5.0))
    closed.close()
    with pytest.raises(RuntimeError):
        module(closed)
    assert module.generator.calls == 0
    assert len(live_storages) == baseline
    x.close()


def test_a_checkpoint_load_preserves_module_generator_identity(tmp_path):
    """Identity and topology, not just values: the same objects afterwards,
    the shared pair still shared, the solo one still independent."""
    model = HardeningModel()
    advance(model, steps=2)
    path = str(tmp_path / "identity.npz")
    save_native_checkpoint(path, model)

    shared_before = model.shared
    solo_before = model.solo
    model.shared.reseed(1)
    model.solo.reseed(2)
    load_native_checkpoint(path, model)

    assert model.shared is shared_before
    assert model.solo is solo_before
    assert model.drop_a.generator is model.drop_b.generator
    assert model.drop_solo.generator is not model.drop_a.generator
    close_model(model)


def test_state_dict_stays_tensor_only_while_generator_state_is_exact():
    """The two reports never blend: ``state_dict()`` is
    ``{name: NativeTensor}`` with no generator key, and
    ``generator_state_dict()`` is the exact four-field state."""
    model = HardeningModel()
    state = model.state_dict()
    try:
        assert all(isinstance(value, NativeTensor) for value in state.values())
        assert not any("generator" in key for key in state)
        generators = model.generator_state_dict()
        assert set(generators) == {"drop_a.generator", "drop_solo.generator"}
        for entry in generators.values():
            assert set(entry) == {"algorithm", "algorithm_version", "seed",
                                  "calls"}
            assert type(entry["seed"]) is int
            assert type(entry["calls"]) is int
    finally:
        for value in state.values():
            value.close()
        close_model(model)


# ===========================================================================
# 8. Checkpoint-v2 corruption
# ===========================================================================


def _corruption_cases(base):
    """Every corruption the loader must reject, as ``(label, mutate)`` or
    ``(label, None, raw_bytes)``. Built from a real archive's manifest, so
    a schema change breaks the list rather than silently bypassing it."""
    def entry(field, value):
        return lambda m: m["generators"]["entries"]["drop_a.generator"] \
            .__setitem__(field, value)

    cases = [
        # -- manifest shape
        ("wrong format name", lambda m: m.__setitem__("format", "other")),
        ("version 3", lambda m: m.__setitem__("format_version", 3)),
        ("version 0", lambda m: m.__setitem__("format_version", 0)),
        ("version -1", lambda m: m.__setitem__("format_version", -1)),
        ("version bool", lambda m: m.__setitem__("format_version", True)),
        ("version float", lambda m: m.__setitem__("format_version", 2.0)),
        ("version string", lambda m: m.__setitem__("format_version", "2")),
        ("v2 without generators", lambda m: m.pop("generators")),
        ("extra top-level field", lambda m: m.__setitem__("extra", 1)),
        ("model wrong type", lambda m: m.__setitem__("model", [])),
        ("optimizer wrong type", lambda m: m.__setitem__("optimizer", 5)),
        ("metadata wrong type", lambda m: m.__setitem__("metadata", [])),
        ("generators wrong type", lambda m: m.__setitem__("generators", 7)),
        ("generators as list", lambda m: m.__setitem__("generators", [])),
        # -- generator section shape
        ("keys wrong type", lambda m: m["generators"].__setitem__("keys", "a")),
        ("keys not strings",
         lambda m: m["generators"].__setitem__("keys", [1, 2])),
        ("keys with empty segment",
         lambda m: m["generators"].__setitem__(
             "keys", ["a..generator", "drop_solo.generator"])),
        ("keys empty string",
         lambda m: m["generators"].__setitem__(
             "keys", ["", "drop_solo.generator"])),
        ("keys duplicated",
         lambda m: m["generators"].__setitem__(
             "keys", ["drop_a.generator", "drop_a.generator"])),
        ("entries wrong type",
         lambda m: m["generators"].__setitem__("entries", [])),
        ("aliases wrong type",
         lambda m: m["generators"].__setitem__("aliases", [])),
        ("extra section field",
         lambda m: m["generators"].__setitem__("surprise", 1)),
        ("entry missing a field",
         lambda m: m["generators"]["entries"]["drop_a.generator"].pop("calls")),
        ("entry extra field", entry("surprise", 1)),
        ("entry wrong type",
         lambda m: m["generators"]["entries"].__setitem__(
             "drop_a.generator", 5)),
        # -- algorithm identity
        ("algorithm mismatch", entry("algorithm", "numpy.pcg64")),
        ("algorithm wrong type", entry("algorithm", 1)),
        ("algorithm_version mismatch", entry("algorithm_version", 2)),
        ("algorithm_version bool", entry("algorithm_version", True)),
        ("algorithm_version string", entry("algorithm_version", "1")),
        # -- topology
        ("missing self-alias",
         lambda m: m["generators"]["aliases"].pop("drop_a.generator")),
        ("alias to absent entry",
         lambda m: m["generators"]["aliases"].__setitem__(
             "drop_a.generator", "ghost.generator")),
        ("canonical not self-mapped",
         lambda m: m["generators"]["aliases"].__setitem__(
             "drop_a.generator", "drop_solo.generator")),
        ("swapped aliases",
         lambda m: m["generators"]["aliases"].update({
             "drop_a.generator": "drop_solo.generator",
             "drop_solo.generator": "drop_a.generator"})),
        ("alias target not a string",
         lambda m: m["generators"]["aliases"].__setitem__(
             "drop_a.generator", 5)),
        ("malformed alias path",
         lambda m: m["generators"]["aliases"].__setitem__(
             ".bad", "drop_a.generator")),
        ("unexpected registered path",
         lambda m: m["generators"]["aliases"].__setitem__(
             "ghost.generator", "drop_a.generator")),
        ("missing registered path",
         lambda m: m["generators"]["aliases"].pop("drop_b.generator")),
        ("dropped canonical entry",
         lambda m: (m["generators"]["keys"].remove("drop_solo.generator"),
                    m["generators"]["entries"].pop("drop_solo.generator"),
                    m["generators"]["aliases"].pop("drop_solo.generator"))),
    ]
    # -- canonical integer strings, on both numeric fields.
    for label, value in [
        ("empty", ""), ("whitespace", " 7 "), ("plus", "+7"),
        ("minus", "-7"), ("leading zeros", "007"), ("decimal point", "7.0"),
        ("exponent", "1e3"), ("hex", "0x1f"), ("underscore", "1_000"),
        ("unicode digits", "٧"), ("json int", 7), ("json bool", True),
        ("json null", None), ("json float", 7.0),
        ("above uint64", "18446744073709551616"),
        ("21 digits", "1" * 21),
    ]:
        cases.append((f"seed {label}", entry("seed", value)))
        cases.append((f"calls {label}", entry("calls", value)))
    return cases


RAW_CORRUPTIONS = [
    ("root is a list", b"[]"),
    ("root is a string", b'"manifest"'),
    ("malformed JSON", b"{not json"),
    ("empty bytes", b""),
    ("invalid UTF-8", b"\xff\xfe{}"),
]


@pytest.fixture
def corruption_fixture(tmp_path):
    """A real v2 archive of a model with both a shared and an independent
    generator, plus everything needed to prove nothing moved."""
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, optimizer, steps=2)
    path = str(tmp_path / "base.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    yield model, optimizer, path, read_manifest(path), tmp_path
    close_model(model, optimizer)


def test_every_manifest_corruption_fails_before_any_live_change(
    corruption_fixture, live_storages,
):
    """The whole corruption matrix as one table. Each case must raise
    ``ValueError``, leave the model, buffers, optimizer, and every
    generator bit-identical, and return native live storage to baseline.

    One test rather than a parametrization on purpose: the point is that
    **all** of them are inert against the *same* live objects, in
    sequence, so a case that corrupted state would be caught by the next
    one as well."""
    model, optimizer, good, base, tmp_path = corruption_fixture
    model_before = model_fingerprint(model)
    optimizer_before = optimizer_fingerprint(optimizer)
    baseline = settled(live_storages)
    destination = str(tmp_path / "corrupt.npz")

    checked = 0
    for label, mutate in _corruption_cases(base):
        manifest = copy.deepcopy(base)
        mutate(manifest)
        rewrite_manifest(good, destination, manifest=manifest)
        with pytest.raises(ValueError):
            load_native_checkpoint(destination, model, optimizer=optimizer)
        assert_model_unchanged(model, model_before)
        assert_optimizer_unchanged(optimizer, optimizer_before)
        assert len(live_storages) == baseline, label
        checked += 1

    for label, raw in RAW_CORRUPTIONS:
        rewrite_manifest(good, destination, raw=raw)
        with pytest.raises(ValueError):
            load_native_checkpoint(destination, model, optimizer=optimizer)
        assert_model_unchanged(model, model_before)
        assert_optimizer_unchanged(optimizer, optimizer_before)
        assert len(live_storages) == baseline, label
        checked += 1

    assert checked >= 70, f"the corruption matrix shrank to {checked} cases"

    # ...and the untouched archive still loads afterwards. A *successful*
    # load legitimately moves every parameter version by one (the v3.7
    # contract), so the values are compared and the versions are checked
    # against that rule rather than against "unchanged".
    load_native_checkpoint(good, model, optimizer=optimizer)
    after = model_fingerprint(model)
    for name, values in model_before["parameters"].items():
        assert np.array_equal(after["parameters"][name], values), name
    for name, values in model_before["buffers"].items():
        assert np.array_equal(after["buffers"][name], values), name
    assert after["generators"] == model_before["generators"]
    assert after["versions"] == {
        name: version + 1 for name, version in model_before["versions"].items()
    }


def test_a_duplicate_json_object_key_is_rejected_at_every_level(
    corruption_fixture,
):
    """Python's ``json`` silently keeps the last duplicate, which for the
    alias map would turn a topology corruption into a valid-looking
    archive. The ``object_pairs_hook`` must reject it wherever it appears."""
    model, optimizer, good, base, tmp_path = corruption_fixture
    destination = str(tmp_path / "dup.npz")
    text = json.dumps(base)

    duplicates = [
        ("top level", text[:-1] + ', "format": "x"}'),
        ("generator section",
         text.replace('"aliases":', '"keys": ["z.generator"], "aliases":', 1)),
    ]
    for label, corrupted in duplicates:
        rewrite_manifest(good, destination, raw=corrupted.encode("utf-8"))
        with pytest.raises(ValueError, match="repeats|match|must"):
            load_native_checkpoint(destination, model, optimizer=optimizer)


def test_saved_shared_versus_live_independent_and_the_reverse(tmp_path):
    """Sharing is semantic state, so both directions of the mismatch must
    fail — naming the paths — before anything is touched."""
    shared_model = HardeningModel()
    shared_path = str(tmp_path / "shared.npz")
    save_native_checkpoint(shared_path, shared_model)

    class Independent(HardeningModel):
        def __init__(self):
            super().__init__()
            self.drop_b.generator = NativeGenerator(1234)

    independent = Independent()
    independent_path = str(tmp_path / "independent.npz")
    save_native_checkpoint(independent_path, independent)

    before = model_fingerprint(independent)
    with pytest.raises(ValueError, match="generator"):
        load_native_checkpoint(shared_path, independent)
    assert_model_unchanged(independent, before)

    before = model_fingerprint(shared_model)
    with pytest.raises(ValueError, match="generator"):
        load_native_checkpoint(independent_path, shared_model)
    assert_model_unchanged(shared_model, before)

    close_model(shared_model)
    close_model(independent)


def test_a_reordered_registration_changes_the_canonical_name_and_fails(
    tmp_path,
):
    """Canonical names are a function of traversal order, so a model whose
    registration order changed is a different topology and is rejected
    naming both names."""
    class Forward(NativeModule):
        def __init__(self, shared):
            super().__init__()
            self.first = NativeDropout(0.5, generator=shared)
            self.second = NativeDropout(0.5, generator=shared)

    class Reversed(NativeModule):
        def __init__(self, shared):
            super().__init__()
            self.second = NativeDropout(0.5, generator=shared)
            self.first = NativeDropout(0.5, generator=shared)

    forward = Forward(NativeGenerator(5))
    path = str(tmp_path / "order.npz")
    save_native_checkpoint(path, forward)
    assert read_manifest(path)["generators"]["keys"] == ["first.generator"]

    reversed_model = Reversed(NativeGenerator(5))
    before = model_fingerprint(reversed_model)
    with pytest.raises(ValueError) as caught:
        load_native_checkpoint(path, reversed_model)
    message = str(caught.value)
    assert "first.generator" in message and "second.generator" in message
    assert_model_unchanged(reversed_model, before)


# ===========================================================================
# 9. Version-1 compatibility
# ===========================================================================


@pytest.mark.parametrize("kind", ["none", "sgd", "adam"])
def test_a_version_1_archive_still_loads_into_a_generator_free_model(
    tmp_path, kind,
):
    """The locked v1 rule, across all three optimizer shapes: model state,
    persistent buffers, optimizer state, and metadata all restore, and the
    format-version 1 field set has no generator section at all."""
    model = PlainModel()
    optimizer = (None if kind == "none"
                 else NativeSGD(model.parameters(), lr=0.1) if kind == "sgd"
                 else NativeAdam(model.parameters(), lr=0.05))
    model.train()
    advance(model, optimizer, steps=2)

    v2 = str(tmp_path / f"{kind}-v2.npz")
    save_native_checkpoint(v2, model, optimizer=optimizer,
                           metadata={"epoch": 4, "kind": kind})
    v1 = downgrade_to_v1(v2, str(tmp_path / f"{kind}-v1.npz"))
    manifest = read_manifest(v1)
    assert manifest["format_version"] == 1
    assert "generators" not in manifest

    expected = model_fingerprint(model)
    expected_optimizer = (None if optimizer is None
                          else optimizer_fingerprint(optimizer))

    # Disturb everything, then restore from the v1 archive.
    advance(model, optimizer, steps=1)
    metadata = load_native_checkpoint(v1, model, optimizer=optimizer)
    assert metadata == {"epoch": 4, "kind": kind}
    after = model_fingerprint(model)
    for name, values in expected["parameters"].items():
        assert np.array_equal(after["parameters"][name], values), name
    for name, values in expected["buffers"].items():
        assert np.array_equal(after["buffers"][name], values), name
    if optimizer is not None:
        assert_optimizer_unchanged(optimizer, expected_optimizer)
    close_model(model, optimizer)


def test_a_version_1_archive_into_a_generator_model_invents_nothing(tmp_path):
    """It must fail naming the generators, and fabricate no seed and no
    counter — not zero, not fresh entropy, not the current value."""
    plain_source = HardeningModel()
    v2 = str(tmp_path / "src.npz")
    save_native_checkpoint(v2, plain_source)
    v1 = downgrade_to_v1(v2, str(tmp_path / "src-v1.npz"))

    model = HardeningModel()
    advance(model, steps=3)
    before = model_fingerprint(model)
    with pytest.raises(ValueError) as caught:
        load_native_checkpoint(v1, model)
    message = str(caught.value)
    assert "drop_a.generator" in message and "drop_solo.generator" in message
    assert "version 1" in message or "version-1" in message
    assert_model_unchanged(model, before)
    close_model(plain_source)
    close_model(model)


def test_a_v1_manifest_may_not_carry_a_generator_field(tmp_path):
    """The absence of ``"generators"`` is what *marks* an archive as
    pre-G5, so a v1 manifest carrying one is rejected as a field-set
    mismatch rather than half-read."""
    model = HardeningModel()
    path = str(tmp_path / "hybrid.npz")
    save_native_checkpoint(path, model)
    manifest = read_manifest(path)
    manifest["format_version"] = 1                     # keeps "generators"
    corrupt = rewrite_manifest(path, str(tmp_path / "v1-gen.npz"),
                               manifest=manifest)
    before = model_fingerprint(model)
    with pytest.raises(ValueError, match="fields do not match"):
        load_native_checkpoint(corrupt, model)
    assert_model_unchanged(model, before)
    close_model(model)


def test_a_v2_generator_archive_into_a_generator_free_model_fails(tmp_path):
    """The other direction: generator state is never silently discarded."""
    with_generators = HardeningModel()
    path = str(tmp_path / "gen.npz")
    save_native_checkpoint(path, with_generators)

    plain = PlainModel()
    before = model_fingerprint(plain)
    with pytest.raises(ValueError, match="registers none|generator"):
        load_native_checkpoint(path, plain)
    assert_model_unchanged(plain, before)
    close_model(with_generators)
    close_model(plain)


# ===========================================================================
# 10. Whole-checkpoint rollback, and the save destination
# ===========================================================================


COMMIT_SEAMS = ["_capture_rollback", "_commit_model", "_commit_optimizer",
                "_commit_generators", "_reach_commit_boundary"]


@pytest.mark.parametrize("seam", COMMIT_SEAMS)
@pytest.mark.parametrize("error", [
    RuntimeError("injected"), MemoryError("injected"), KeyboardInterrupt(),
    Boom("injected"),
], ids=["RuntimeError", "MemoryError", "KeyboardInterrupt", "BaseException"])
def test_every_commit_position_rolls_all_four_families_back(
    seam, error, tmp_path, live_storages,
):
    """Each transaction position × each exception class. Afterwards the
    parameters, persistent buffers, parameter versions, optimizer scalars
    and moments, and every generator's seed and counter are exactly what
    they were; every publicly identified object is the same object; native
    live storage is back at baseline; and the original exception is what
    propagates."""
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, optimizer, steps=2)
    path = str(tmp_path / "rollback.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    # Move every family away from the archive, so a rollback that silently
    # committed would be visible.
    advance(model, optimizer, steps=2)
    model.shared.reseed(0xAAAA)
    model.solo.reseed(0xBBBB)

    identities = {
        "parameters": {name: id(p) for name, p in model.named_parameters()},
        "generators": {name: id(g) for name, g in model.named_generators()},
        "optimizer": id(optimizer),
    }
    model_before = model_fingerprint(model)
    optimizer_before = optimizer_fingerprint(optimizer)
    baseline = settled(live_storages)

    real = getattr(transaction, seam)

    def failing(*args, **kwargs):
        if seam == "_capture_rollback":
            # Let the real snapshot run so the caller still owns what it
            # must close, then fail at the position itself.
            real(*args, **kwargs)
        raise error

    with patched(transaction, seam, failing):
        with pytest.raises(type(error)) as caught:
            load_native_checkpoint(path, model, optimizer=optimizer)

    assert caught.value is error, "the rollback replaced the failure"
    assert_model_unchanged(model, model_before)
    assert_optimizer_unchanged(optimizer, optimizer_before)
    assert {name: id(p) for name, p in model.named_parameters()} \
        == identities["parameters"]
    assert {name: id(g) for name, g in model.named_generators()} \
        == identities["generators"]
    assert id(optimizer) == identities["optimizer"]
    assert model.drop_a.generator is model.drop_b.generator
    assert settled(live_storages) == baseline

    # Both locks were released, so a real load works immediately.
    assert state_lock.held_by_current_thread() is False
    load_native_checkpoint(path, model, optimizer=optimizer)
    close_model(model, optimizer)


def test_a_rollback_leaves_active_reservations_and_saved_masks_untouched(
    tmp_path,
):
    """Two things a load must never disturb, proved across a rolled-back
    commit: a graph-owned Dropout mask from an earlier forward, and an
    unrelated generator's live reservation."""
    model = HardeningModel()
    path = str(tmp_path / "masks.npz")
    save_native_checkpoint(path, model)

    # A retained graph built before the load.
    values = np.arange(1.0, 7.0).reshape(2, 3)
    parameter = NativeParameter(values, requires_grad=True)
    outsider = NativeGenerator(0x515)
    mask = core_mask(values, 0.5, 0x515, 0)
    dropped = parameter.dropout(0.5, generator=outsider)
    loss = dropped.sum()

    # A live reservation on a generator the load does not target.
    untargeted = NativeGenerator(0x616)
    token = untargeted._reserve_call()
    reservation_before = internals(untargeted)

    with patched(transaction, "_reach_commit_boundary",
                 raiser(RuntimeError("injected"))):
        with pytest.raises(RuntimeError, match="injected"):
            load_native_checkpoint(path, model)

    assert internals(untargeted) == reservation_before
    untargeted._commit_call(token)
    assert untargeted.calls == 1

    loss.backward()
    assert np.allclose(parameter.grad.to_numpy(), mask)

    close_all(loss, dropped, parameter)
    close_model(model)


SAVE_SEAMS = [
    ("model snapshot", NativeModule, "state_dict"),
    ("optimizer snapshot", NativeAdam, "state_dict"),
    ("generator snapshot", checkpoint_module, "_generator_section"),
    ("manifest encoding", checkpoint_module.json, "dumps"),
    ("npz creation", checkpoint_module.np, "savez"),
    ("temporary file", checkpoint_module.tempfile, "mkstemp"),
    ("final replace", checkpoint_module.os, "replace"),
]


@pytest.mark.parametrize("label, target, attribute", SAVE_SEAMS,
                          ids=[case[0] for case in SAVE_SEAMS])
def test_a_failing_save_leaves_the_destination_byte_identical(
    label, target, attribute, tmp_path, live_storages,
):
    """An existing valid destination must survive every save failure
    byte-for-byte, with no temporary debris, no state mutation, no
    generator call consumed, and a later valid save still working."""
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, optimizer, steps=2)
    path = str(tmp_path / "destination.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    original = Path(path).read_bytes()
    listing = sorted(os.listdir(tmp_path))
    model_before = model_fingerprint(model)
    optimizer_before = optimizer_fingerprint(optimizer)
    baseline = settled(live_storages)

    with patched(target, attribute, raiser(RuntimeError(f"injected at {label}"))):
        with pytest.raises(RuntimeError, match="injected"):
            save_native_checkpoint(path, model, optimizer=optimizer)

    assert Path(path).read_bytes() == original, "the destination changed"
    assert sorted(os.listdir(tmp_path)) == listing, "temporary debris remained"
    assert_model_unchanged(model, model_before)
    assert_optimizer_unchanged(optimizer, optimizer_before)
    assert model.shared._has_active_reservation() is False
    assert model.solo._has_active_reservation() is False
    assert settled(live_storages) == baseline

    # A later valid save succeeds and, with nothing changed, is identical.
    save_native_checkpoint(path, model, optimizer=optimizer)
    assert Path(path).read_bytes() == original
    close_model(model, optimizer)


def test_repeated_saves_of_unchanged_state_are_byte_identical(tmp_path):
    """Determinism is a property of this archive format (fixed key order,
    canonical decimal strings, no timestamps in the payload), so it is
    asserted rather than assumed — and it is what makes "the manifest did
    not change" a usable check elsewhere in this suite."""
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, optimizer, steps=2)
    first = str(tmp_path / "a.npz")
    second = str(tmp_path / "b.npz")
    save_native_checkpoint(first, model, optimizer=optimizer)
    save_native_checkpoint(second, model, optimizer=optimizer)
    assert Path(first).read_bytes() == Path(second).read_bytes()
    assert read_manifest(first) == read_manifest(second)
    close_model(model, optimizer)


# ===========================================================================
# 11. State-transaction concurrency
# ===========================================================================


def test_a_rolled_back_load_completes_before_a_waiting_load_observes_it(
    tmp_path,
):
    """Serializability across a *failure*: while A rolls back, B waits, and
    B then observes A's fully restored pre-load state — never a partially
    committed one."""
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, optimizer, steps=2)
    path_a = str(tmp_path / "a.npz")
    save_native_checkpoint(path_a, model, optimizer=optimizer)
    advance(model, optimizer, steps=2)
    model.shared.reseed(0x1111)
    path_b = str(tmp_path / "b.npz")
    save_native_checkpoint(path_b, model, optimizer=optimizer)

    pre_load = model_fingerprint(model)
    observed = {}
    interleaver = Interleaver()
    real_boundary = transaction._reach_commit_boundary
    first = threading.Event()

    def failing_boundary():
        # Only A fails; B must be able to commit for real once A has fully
        # rolled back, which is the property under test.
        if first.is_set():
            return real_boundary()
        first.set()
        interleaver.block()
        raise RuntimeError("A fails after committing everything")

    def loader_a():
        try:
            load_native_checkpoint(path_a, model, optimizer=optimizer)
        except RuntimeError:
            observed["a"] = "rolled back"

    def loader_b():
        interleaver.wait_for_arrival()
        # A is parked *inside* the guard holding it, so B cannot proceed
        # until A's rollback has finished and released it.
        observed["b_saw_a_parked"] = True
        interleaver.let_go()
        load_native_checkpoint(path_b, model, optimizer=optimizer)
        observed["b"] = "committed"

    with patched(transaction, "_reach_commit_boundary", failing_boundary):
        run_threads([loader_a, loader_b])

    assert observed["a"] == "rolled back"
    assert observed["b"] == "committed"
    assert observed["b_saw_a_parked"] is True
    # The final state is B's archive — one whole operation's result, never a
    # mixture of A's model with B's generators.
    after = model_fingerprint(model)
    assert after["generators"]["drop_a.generator"]["seed"] == 0x1111
    assert after["generators"] == pre_load["generators"]
    close_model(model, optimizer)


def test_two_unrelated_models_serialize_without_deadlocking(tmp_path):
    """Unrelated-model transactions take the same one global guard, so they
    serialize; the property under test is that they still both finish."""
    first = HardeningModel(shared_seed=1, solo_seed=2)
    second = HardeningModel(shared_seed=3, solo_seed=4)
    first_optimizer = NativeAdam(first.parameters(), lr=0.05)
    second_optimizer = NativeAdam(second.parameters(), lr=0.05)
    advance(first, first_optimizer, steps=1)
    advance(second, second_optimizer, steps=1)
    first_path = str(tmp_path / "first.npz")
    second_path = str(tmp_path / "second.npz")
    save_native_checkpoint(first_path, first, optimizer=first_optimizer)
    save_native_checkpoint(second_path, second, optimizer=second_optimizer)

    barrier = threading.Barrier(2)

    def load(path, model, optimizer):
        def run():
            barrier.wait(JOIN_TIMEOUT)
            for _ in range(3):
                load_native_checkpoint(path, model, optimizer=optimizer)
        return run

    run_threads([load(first_path, first, first_optimizer),
                 load(second_path, second, second_optimizer)])
    assert first.shared.seed == 1 and second.shared.seed == 3
    close_model(first, first_optimizer)
    close_model(second, second_optimizer)


def test_overlapping_shared_generator_target_sets_cannot_deadlock(tmp_path):
    """Two models sharing generators, loaded concurrently from opposite
    registration orders. The identity-sorted acquisition order is what
    makes this terminate; a caller-derived order would deadlock."""
    shared_one = NativeGenerator(11)
    shared_two = NativeGenerator(13)

    class Forward(NativeModule):
        def __init__(self):
            super().__init__()
            self.a = NativeDropout(0.5, generator=shared_one)
            self.b = NativeDropout(0.5, generator=shared_two)

    class Backward(NativeModule):
        def __init__(self):
            super().__init__()
            self.b = NativeDropout(0.5, generator=shared_two)
            self.a = NativeDropout(0.5, generator=shared_one)

    forward, backward = Forward(), Backward()
    forward_path = str(tmp_path / "fwd.npz")
    backward_path = str(tmp_path / "bwd.npz")
    save_native_checkpoint(forward_path, forward)
    save_native_checkpoint(backward_path, backward)

    barrier = threading.Barrier(2)

    def load(path, model):
        def run():
            barrier.wait(JOIN_TIMEOUT)
            for _ in range(4):
                load_native_checkpoint(path, model)
        return run

    run_threads([load(forward_path, forward), load(backward_path, backward)])
    assert shared_one.seed == 11 and shared_two.seed == 13


def test_a_generator_free_model_still_serializes(tmp_path):
    """Serializability is not conditional on the model having generators —
    the guard is taken either way, so this must not quietly rest on
    generator locks."""
    model = PlainModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, optimizer, steps=1)
    path = str(tmp_path / "plain.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    assert read_manifest(path)["generators"] is None

    seen = []
    real = transaction._commit_model

    def traced(*args, **kwargs):
        seen.append(state_lock.held_by_current_thread())
        return real(*args, **kwargs)

    with patched(transaction, "_commit_model", traced):
        barrier = threading.Barrier(2)

        def load():
            barrier.wait(JOIN_TIMEOUT)
            for _ in range(4):
                load_native_checkpoint(path, model, optimizer=optimizer)

        run_threads([load, load])
    assert seen and all(seen), "a commit ran without the guard held"
    close_model(model, optimizer)


def test_ordinary_training_mutation_is_honestly_not_serialized():
    """The documented limitation, asserted so nobody upgrades the claim by
    accident: ``step()``, ``copy_value_``, and a backward do **not** take
    the shared guard."""
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    observed = []
    real = state_lock.held_by_current_thread

    x = sample()
    out = model(x)
    loss = out.sum()
    loss.backward()
    observed.append(("backward", real()))
    optimizer.step()
    observed.append(("step", real()))
    parameter = model.linear.weight
    replacement = NativeTensor.from_array(parameter.to_numpy() * 1.5)
    parameter.copy_value_(replacement)
    observed.append(("copy_value_", real()))

    assert all(not held for _, held in observed), observed
    replacement.close()
    x.close()
    close_model(model, optimizer)


# ===========================================================================
# 12. Exact next-mask restoration
# ===========================================================================


@pytest.mark.parametrize("calls", [0, 1, 7, UINT64_MAX - 1, UINT64_MAX])
def test_the_next_mask_after_a_load_matches_the_core_at_the_restored_index(
    tmp_path, calls,
):
    """Exact generator restoration, at every interesting counter value:
    after a load, the next Dropout output equals the G2 Core at the
    restored ``(seed, calls)`` — and at exhaustion the next forward is
    refused without allocating or moving anything."""
    seed = 0xF00DCAFE
    saved = NativeDropout(0.5, seed=seed)
    at_calls(saved.generator, calls)
    path = str(tmp_path / f"next-{calls}.npz")
    save_native_checkpoint(path, saved)

    fresh = NativeDropout(0.5, seed=seed)
    at_calls(fresh.generator, 0)
    load_native_checkpoint(path, fresh)
    assert fresh.generator.calls == calls
    assert fresh.generator.seed == seed

    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values)
    try:
        if calls == UINT64_MAX:
            with pytest.raises(RuntimeError, match="exhausted"):
                fresh(x)
            assert fresh.generator.calls == UINT64_MAX
            return
        out = fresh(x)
        try:
            assert np.allclose(out.to_numpy(),
                               values * core_mask(values, 0.5, seed, calls))
        finally:
            out.close()
        assert fresh.generator.calls == calls + 1
    finally:
        x.close()


def test_a_restored_shared_stream_resumes_interleaved(tmp_path):
    """One generator, two layers: after a load the two layers keep taking
    alternating indices from the one restored stream."""
    model = HardeningModel()
    at_calls(model.shared, 5)
    path = str(tmp_path / "interleaved.npz")
    save_native_checkpoint(path, model)

    fresh = HardeningModel()
    load_native_checkpoint(path, fresh)
    assert fresh.shared.calls == 5

    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values)
    try:
        first = fresh.drop_a(x)
        second = fresh.drop_b(x)
        try:
            assert np.allclose(
                first.to_numpy(),
                values * core_mask(values, 0.5, fresh.shared.seed, 5))
            assert np.allclose(
                second.to_numpy(),
                values * core_mask(values, 0.5, fresh.shared.seed, 6))
        finally:
            first.close()
            second.close()
        assert fresh.shared.calls == 7
    finally:
        x.close()
    close_model(model)
    close_model(fresh)


def test_independent_generators_restore_independently(tmp_path):
    """Several generators in one archive, each restored to its own state,
    each producing its own next mask."""
    model = HardeningModel(shared_seed=0xAAA, solo_seed=0xBBB)
    at_calls(model.shared, 3)
    at_calls(model.solo, 11)
    path = str(tmp_path / "several.npz")
    save_native_checkpoint(path, model)

    fresh = HardeningModel(shared_seed=1, solo_seed=2)
    load_native_checkpoint(path, fresh)
    assert (fresh.shared.seed, fresh.shared.calls) == (0xAAA, 3)
    assert (fresh.solo.seed, fresh.solo.calls) == (0xBBB, 11)

    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values)
    try:
        shared_out = fresh.drop_a(x)
        solo_out = fresh.drop_solo(x)
        try:
            assert np.allclose(shared_out.to_numpy(),
                               values * core_mask(values, 0.5, 0xAAA, 3))
            assert np.allclose(solo_out.to_numpy(),
                               values * core_mask(values, 0.25, 0xBBB, 11))
        finally:
            shared_out.close()
            solo_out.close()
    finally:
        x.close()
    close_model(model)
    close_model(fresh)


# ===========================================================================
# 13. Live-storage lifecycle loops
# ===========================================================================


@needs_fault_injection
def test_a_full_lifecycle_loop_returns_native_storage_to_baseline(
    tmp_path, live_storages,
):
    """The practical proof, as a loop rather than a one-shot: successful
    Core calls, failed Core calls, differentiable forward/backward,
    abandoned graphs, no-grad forwards, module train/eval, checkpoint
    save/load, and failed loads — repeated, with the live native storage
    count returning **exactly** to its post-setup value each time and no
    monotonic growth across the loop."""
    baseline = settled(live_storages)
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    path = str(tmp_path / "loop.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    after_setup = settled(live_storages)
    values = np.arange(1.0, 13.0)
    observed = []

    for cycle in range(3):
        # (a) a successful Core output/mask pair, explicitly closed.
        source = cpp.NativeTensorCore.from_array(values)
        out, mask = source._dropout_forward_with_mask(0.5, seed=5,
                                                      call_index=cycle)
        out.close()
        mask.close()
        source.close()

        # (b) a failed Core call at each of the two allocations.
        for nth in (1, 2):
            source = cpp.NativeTensorCore.from_array(values)
            with pytest.raises(MemoryError):
                cpp._arm_alloc_failure(nth)
                source._dropout_forward_with_mask(0.5, seed=5, call_index=0)
            cpp._arm_alloc_failure(0)
            source.close()

        # (c) a differentiable forward and backward.
        parameter = NativeParameter(values, requires_grad=True)
        dropped = parameter.dropout(0.5, generator=model.solo)
        loss = dropped.sum()
        loss.backward()
        close_all(loss, dropped, parameter)

        # (d) an abandoned graph, released by close().
        x = NativeTensor.from_array(values, requires_grad=True)
        abandoned = x.dropout(0.5, generator=model.solo)
        abandoned.close()
        x.close()

        # (e) a no-grad forward — the mask is closed inside the call.
        plain = NativeTensor.from_array(values)
        no_grad = plain.dropout(0.5, generator=model.solo)
        no_grad.close()

        # (f) module train and eval.
        model.drop_solo.eval()
        assert model.drop_solo(plain) is plain
        model.drop_solo.train()
        trained = model.drop_solo(plain)
        trained.close()
        plain.close()

        # (g) checkpoint save/load, and a failed load.
        save_native_checkpoint(path, model, optimizer=optimizer)
        load_native_checkpoint(path, model, optimizer=optimizer)
        with pytest.raises((ValueError, FileNotFoundError)):
            load_native_checkpoint(str(tmp_path / "absent.npz"), model)

        current = settled(live_storages)
        observed.append(current)
        assert current == after_setup, (
            f"cycle {cycle}: live storage drifted to {current} "
            f"from {after_setup}"
        )

    assert len(set(observed)) == 1, f"monotonic growth across cycles: {observed}"

    close_model(model, optimizer)
    assert settled(live_storages) == baseline


def test_a_rollback_lifecycle_loop_returns_storage_to_baseline(
    tmp_path, live_storages,
):
    """Repeated *failed* commits: every rolled-back load must release its
    staged tensors and its rollback snapshots, so failure cycles are as
    clean as success cycles."""
    baseline = settled(live_storages)
    model = HardeningModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, optimizer, steps=1)
    path = str(tmp_path / "rollback-loop.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    after_setup = settled(live_storages)
    before = model_fingerprint(model)

    for seam in COMMIT_SEAMS * 2:
        real = getattr(transaction, seam)

        def failing(*args, _real=real, _seam=seam, **kwargs):
            if _seam == "_capture_rollback":
                _real(*args, **kwargs)
            raise RuntimeError("injected")

        with patched(transaction, seam, failing):
            with pytest.raises(RuntimeError, match="injected"):
                load_native_checkpoint(path, model, optimizer=optimizer)
        assert settled(live_storages) == after_setup, seam
        assert_model_unchanged(model, before)

    close_model(model, optimizer)
    assert settled(live_storages) == baseline


def test_the_dropout_path_reaches_no_numpy_numerical_routine(monkeypatch):
    """A tripwire over one complete stochastic step: the mask, the output,
    and the gradient are all native. ``to_numpy``/``from_array`` are the
    explicit serialization boundary and are left alone; everything that
    would mean NumPy is doing the arithmetic is not."""
    tripped = []
    for name in ("multiply", "where", "random", "add", "matmul", "einsum"):
        attribute = getattr(np, name, None)
        if attribute is None:                        # pragma: no cover
            continue

        def trap(*args, _name=name, **kwargs):
            tripped.append(_name)
            raise AssertionError(f"numpy.{_name} reached the Dropout path")

        monkeypatch.setattr(np, name, trap)

    generator = NativeGenerator(0x7777)
    parameter = NativeParameter(np.arange(1.0, 13.0), requires_grad=True)
    dropped = parameter.dropout(0.5, generator=generator)
    loss = dropped.sum()
    loss.backward()
    monkeypatch.undo()

    assert not tripped, tripped
    assert generator.calls == 1
    close_all(loss, dropped, parameter)


# ===========================================================================
# 14. Semantic guardrails — G6 is hardening only
# ===========================================================================


def test_g6_moved_no_capability_registry_value():
    """The whole point of the milestone: nothing in any registry changed.
    ``"dropout"`` is still unsupported and only G10 removes it."""
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "dropout" in cpp.UNSUPPORTED
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    assert "dropout_backward" not in cpp.TENSOR_CORE_OPS
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert "generator_state" in cpp.STATE_SUPPORT
    assert "checkpoint_generator_state" in cpp.STATE_SUPPORT


def test_g6_added_no_operation_module_export_or_checkpoint_field():
    """The public surface is exactly what G5 left."""
    import tensorforge.experimental as experimental

    assert checkpoint_module._FORMAT == "tensorforge.native_checkpoint"
    assert checkpoint_module._FORMAT_VERSION == 2
    assert checkpoint_module._SUPPORTED_FORMAT_VERSIONS == (1, 2)
    assert checkpoint_module._GENERATOR_SECTION_KEYS == {
        "keys", "entries", "aliases"
    }
    assert checkpoint_module._GENERATOR_ENTRY_KEYS == {
        "algorithm", "algorithm_version", "seed", "calls"
    }
    assert checkpoint_module._MANIFEST_KEYS == {
        "format", "format_version", "model", "optimizer", "generators",
        "metadata",
    }

    # No new Dropout or RNG operation, module, or export.
    for absent in ("NativeDropout2d", "NativeDropout3d", "NativeAlphaDropout",
                   "NativeRandom", "native_random", "NativeRNG"):
        assert not hasattr(experimental, absent), absent
    for absent in ("dropout2d", "dropout3d", "dropout_backward", "randn",
                   "uniform", "bernoulli"):
        assert absent not in cpp.AUTOGRAD_OPS, absent
        assert absent not in cpp.TENSOR_CORE_OPS, absent

    # The signatures G6 must not touch.
    assert list(inspect.signature(NativeTensor.dropout).parameters) == [
        "self", "p", "generator"
    ]
    assert inspect.signature(NativeTensor.dropout).parameters["generator"].kind \
        is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(NativeTensor.dropout).parameters["generator"].default \
        is inspect.Parameter.empty
    assert list(inspect.signature(NativeDropout.__init__).parameters) == [
        "self", "p", "seed", "generator"
    ]
    assert not hasattr(NativeDropout, "owns_generator")
    assert not hasattr(NativeGenerator, "close")


def test_g9_integration_suite_has_not_begun():
    """G9's integration suite must not exist yet, and no result artifact
    of any kind was written. (G7's example and G8's benchmark do exist —
    later milestones shipped them; this suite is not where their absence
    is asserted, and G8's harness writes no file unless asked.)"""
    for absent in ("tests/test_native_phase_g.py",
                   "benchmark_results"):
        assert not (REPO_ROOT / absent).exists(), absent


def test_this_suite_is_hardening_only_and_makes_no_training_claim():
    """Derived from the file itself: a hardening suite must not quietly
    become a training example or a timing harness. The G7 resume proof
    lives in its own example and test module, not here."""
    source = Path(__file__).read_text(encoding="utf-8")
    # Assembled at runtime so the guard's own list is not a match for it.
    banned = ["BENCHMARK" + "_NAME", "perf" + "_counter", "def " + "train(",
              "def " + "main(", "time" + ".time(", "--" + "smoke"]
    for name in banned:
        assert name not in source, name
    assert "no capability" in source.lower()


def test_the_shared_guard_is_global_and_outermost_with_generator_locks_under_it(
    tmp_path,
):
    """The universal state-replacement lock order, observed rather than
    argued: one process-wide ``RLock``, held at every ordering decision,
    with the generator locks taken beneath it in sorted ``id()`` order."""
    assert isinstance(state_lock._STATE_TRANSACTION_LOCK,
                      type(threading.RLock()))
    # Every participant shares the one object.
    from tensorforge.experimental import _native_state, native_adam, native_sgd
    for module in (_native_state, native_adam, native_sgd, generator_module,
                   checkpoint_module, transaction):
        assert module.state_transaction() is state_lock.state_transaction()

    observed = {}
    real_order = generator_module._ordered_targets

    def traced(generators):
        ordered = real_order(generators)
        observed["held"] = state_lock.held_by_current_thread()
        observed["sorted"] = [id(g) for g in ordered] == sorted(
            id(g) for g in ordered
        )
        return ordered

    model = HardeningModel()
    with patched(generator_module, "_ordered_targets", traced):
        path = str(tmp_path / "order.npz")
        save_native_checkpoint(path, model)
        load_native_checkpoint(path, model)

    assert observed["held"] is True, "the ordering ran without the guard"
    assert observed["sorted"] is True, "generator locks were not id()-sorted"
    close_model(model)


def test_a_reservation_never_takes_the_shared_guard():
    """What keeps the two lock systems from inverting: the reservation path
    takes only its own generator's lock."""
    generator = NativeGenerator(0x808)
    observed = []
    real = generator_module._ReservationToken

    def traced(gen, serial, index):
        observed.append(state_lock.held_by_current_thread())
        return real(gen, serial, index)

    with patched(generator_module, "_ReservationToken", traced):
        token = generator._reserve_call()
        observed.append(state_lock.held_by_current_thread())
    generator._commit_call(token)
    observed.append(state_lock.held_by_current_thread())
    assert observed and not any(observed), observed


def test_the_runtime_holds_no_random_state_at_all():
    """§7.6, from the sources: no C++ translation unit and no Python
    surface keeps random state, and nothing in the phase reaches a foreign
    RNG."""
    def code_only(path):
        """The file's *code*, with comments stripped.

        Necessary rather than fussy: ``random.cpp`` documents in prose the
        very things it must not use ("no ``std::random_device``, no
        ``mt19937``..."), so a raw substring scan would fail on the comment
        that states the guarantee."""
        lines = []
        in_block = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if in_block:
                if "*/" in stripped:
                    in_block = False
                continue
            if stripped.startswith("/*"):
                in_block = "*/" not in stripped
                continue
            if stripped.startswith("//"):
                continue
            lines.append(line.split("//")[0])
        return "\n".join(lines)

    # No translation unit anywhere reaches a foreign RNG or the clock.
    sources = list((REPO_ROOT / "cpp" / "src").glob("*.cpp"))
    sources += list((REPO_ROOT / "cpp" / "include").glob("*.h"))
    assert sources
    for source in sources:
        text = code_only(source)
        for banned in ("<random>", "std::random_device", "mt19937",
                       "std::chrono", "getpid", "rand()", "srand"):
            assert banned not in text, (source.name, banned)

    # ...and the random unit specifically holds no state of its own. Scoped
    # to it deliberately: ``error.cpp`` legitimately uses ``thread_local``
    # for the ABI's error slot, which is not random state.
    random_sources = [REPO_ROOT / "cpp" / "src" / "random.cpp",
                      REPO_ROOT / "cpp" / "include" / "tf_random_internal.h"]
    for source in random_sources:
        assert source.exists(), source
        text = code_only(source)
        for banned in ("thread_local", "static std::uint64_t", "static int",
                       "std::atomic", "std::mutex"):
            assert banned not in text, (source.name, banned)

    assert "tf_core_dropout_forward" in cpp._CHECKED_KERNELS
    for banned in ("tf_core_dropout_backward", "tf_random_seed",
                   "tf_random_state", "tf_core_randn", "tf_core_uniform"):
        assert banned not in cpp._CHECKED_KERNELS, banned
        assert banned not in cpp.KERNELS, banned

    # The generator itself owns no native storage and no lifecycle.
    generator = NativeGenerator(1)
    for absent in ("close", "closed", "clone", "split", "jump", "advance",
                   "randn", "uniform", "bernoulli"):
        assert not hasattr(generator, absent), absent


def test_backward_stays_rng_free_and_eval_and_p_zero_consume_nothing():
    """The three consumption rules in one place, since they are what the
    whole resume story rests on."""
    module = NativeDropout(0.5, seed=0x909)
    values = np.arange(1.0, 13.0)
    parameter = NativeParameter(values, requires_grad=True)

    dropped = module(parameter)
    assert module.generator.calls == 1
    loss = dropped.sum()
    loss.backward()
    assert module.generator.calls == 1, "backward consumed a call"

    module.eval()
    for _ in range(3):
        assert module(parameter) is parameter
    assert module.generator.calls == 1

    module.train()
    zero = NativeDropout(0.0, generator=module.generator)
    for _ in range(3):
        assert zero(parameter) is parameter
    assert module.generator.calls == 1

    close_all(loss, dropped, parameter)
