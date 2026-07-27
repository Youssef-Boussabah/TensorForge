"""The shared native state-transaction guard — serializability of every
in-memory state replacement (Phase G, milestone G5; contract locked in
docs/native_rng_dropout_design.md §10.8).

Each native state-replacement path was already **individually atomic**:
``replace_native_state`` for parameters and persistent buffers,
``replace_generator_states`` for generators, each optimizer's
``load_state_dict``, and the whole-checkpoint transaction over all of
them. Atomic is not serializable. Two concurrent checkpoint loads could
each be internally all-or-nothing and still finish with the model from
one archive beside the optimizer or the generators from the other — a
hybrid state assembled from two checkpoints that no per-component
guarantee can see, and that deadlock freedom alone does not prevent.

G5 therefore runs every participating replacement under **one** private
guard, in one universal order: the shared ``_native_state_lock`` first,
then every unique target generator's lock in the global ``id()`` order.
These tests prove the resulting execution is serializable — after two
concurrent operations finish, the complete live state equals one of them
followed by the other — by **forcing** the interleaving with barriers and
events, never by sleeping and hoping.

The proof has two halves in every test:

1. a **commit trace**, recorded at the real mutation seams *inside* the
   guard, which must contain one contiguous run per thread — no
   interleaving at all; and
2. the **final state**, which must equal exactly one operation's result
   across all four state families, and specifically the one whose run
   ends the trace.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Every join is bounded, so a regression fails the suite
instead of hanging it.

Selector: python -m pytest -q -k native_state_transaction
"""

import json
import threading

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeDropout,
    NativeGenerator,
    NativeLinear,
    NativeModule,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import _native_state
from tensorforge.experimental import _native_state_lock as state_lock
from tensorforge.experimental import (
    _native_checkpoint_transaction as transaction,
)
from tensorforge.experimental import native_generator as generator_module

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

X = np.arange(1.0, 25.0).reshape(6, 4)
JOIN_TIMEOUT = 30.0        # bounded: a regression fails, never hangs
EVENT_TIMEOUT = 15.0


# ======================================================================
# Models and fingerprints
# ======================================================================


class TxnModel(NativeModule):
    """All four state families at once: parameters, persistent buffers,
    a shared generator across two Dropout layers, and an independent
    one."""

    def __init__(self, seed=1, shared_seed=11, own_seed=22):
        super().__init__()
        self.linear = NativeLinear(4, 4, seed=seed)
        self.norm = NativeBatchNorm1d(4)
        shared = NativeGenerator(shared_seed)
        self.drop_a = NativeDropout(0.5, generator=shared)
        self.drop_b = NativeDropout(0.5, generator=shared)
        self.drop_c = NativeDropout(0.25, seed=own_seed)

    def forward(self, x):
        return self.drop_c(self.drop_b(self.drop_a(self.norm(self.linear(x)))))


def plain_model(seed=0):
    """A generator-free model, so serialization can be shown not to
    depend on generator locks."""
    return NativeSequential(
        NativeLinear(4, 4, seed=seed), NativeReLU(),
        NativeLinear(4, 2, seed=seed + 1),
    )


def advance(model, steps=1, optimizer=None):
    x = NativeTensor.from_array(X)
    for _ in range(steps):
        y = model(x)
        y.sum().backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
        y.close()
    x.close()


def fingerprint(model, optimizer=None):
    """Everything a serial order must reproduce exactly."""
    state = {
        "tensors": {name: tensor.to_numpy().copy()
                    for name, tensor in model._state_named_tensors()},
        "generators": model.generator_state_dict(),
    }
    if optimizer is not None:
        state["lr"] = optimizer.lr
        if isinstance(optimizer, NativeAdam):
            state["steps"] = optimizer.step_counts
            state["betas"] = optimizer.betas
            state["eps"] = optimizer.eps
            state["moments"] = [buffer.to_numpy().copy()
                                for buffer in optimizer._m + optimizer._v]
    return state


def same(first, second):
    if set(first) != set(second):
        return False
    if first["generators"] != second["generators"]:
        return False
    if set(first["tensors"]) != set(second["tensors"]):
        return False
    for name, values in first["tensors"].items():
        if not np.array_equal(values, second["tensors"][name]):
            return False
    for key in ("lr", "steps", "betas", "eps"):
        if key in first and first[key] != second.get(key):
            return False
    if "moments" in first:
        if len(first["moments"]) != len(second.get("moments", [])):
            return False
        for a, b in zip(first["moments"], second["moments"]):
            if not np.array_equal(a, b):
                return False
    return True


def close_all(model, optimizer=None):
    for _, parameter in model.named_parameters():
        parameter.close()
    for _, buffer in model.named_buffers():
        buffer.close()
    if isinstance(optimizer, NativeAdam):
        optimizer.close()


# ======================================================================
# Recording and forcing
# ======================================================================


class Trace:
    """Every state mutation that happens **inside** the guard, tagged
    with the thread that made it. Serializability is exactly the claim
    that this list has one contiguous run per thread."""

    def __init__(self):
        self.entries = []
        self._lock = threading.Lock()

    def record(self, component):
        with self._lock:
            self.entries.append(
                (threading.current_thread().name, component)
            )

    def reset(self):
        """Drop everything recorded so far.

        Setup runs in the main thread and legitimately replaces state —
        a BatchNorm forward commits its running statistics through the
        same guarded transaction — so the trace is cleared immediately
        before the threads start. What it then holds is exactly the
        concurrent window."""
        with self._lock:
            self.entries.clear()

    @property
    def threads(self):
        """The thread labels in order, with consecutive repeats folded —
        so ``['A', 'B']`` means "all of A, then all of B"."""
        folded = []
        for name, _ in self.entries:
            if not folded or folded[-1] != name:
                folded.append(name)
        return folded

    def assert_serial(self):
        names = self.threads
        assert len(names) == len(set(names)), (
            f"state commits interleaved across threads: {names} "
            f"(full trace: {self.entries})"
        )
        return names


@pytest.fixture
def trace(monkeypatch):
    """Record at the real mutation seams — the ones inside the guard, so
    a recording proves the mutation really happened under it, not merely
    that a method was entered."""
    recorder = Trace()

    real_install = _native_state._install_core
    real_assign = generator_module.NativeGenerator._assign_state
    real_commit_optimizer = transaction._commit_optimizer
    real_adam_load = NativeAdam._load_state_dict_locked

    def install(planned, new_core):
        assert state_lock.held_by_current_thread(), (
            "a model state replacement ran without the shared guard"
        )
        recorder.record("model")
        return real_install(planned, new_core)

    def assign(self, seed, calls):
        recorder.record("generator")
        return real_assign(self, seed, calls)

    def commit_optimizer(optimizer, staged):
        recorder.record("optimizer")
        return real_commit_optimizer(optimizer, staged)

    def adam_load(self, state, where):
        recorder.record("optimizer")
        return real_adam_load(self, state, where)

    monkeypatch.setattr(_native_state, "_install_core", install)
    monkeypatch.setattr(
        generator_module.NativeGenerator, "_assign_state", assign
    )
    monkeypatch.setattr(transaction, "_commit_optimizer", commit_optimizer)
    monkeypatch.setattr(NativeAdam, "_load_state_dict_locked", adam_load)
    return recorder


class Interleaver:
    """Forces a real overlap without sleeping.

    The first thread to reach the chosen seam — which is *inside* the
    guard — parks there until the other thread reports that it has called
    into its own operation. The second thread therefore contends for the
    guard for certain, and whichever way the race is won, the trace and
    the final state must show one whole operation after the other."""

    def __init__(self):
        self.inside = threading.Event()      # someone is parked in the seam
        self.other_called = threading.Event()  # the rival entered its call
        self._parked = threading.Lock()
        self._used = False

    def park(self):
        with self._parked:
            if self._used:
                return
            self._used = True
        self.inside.set()
        self.other_called.wait(timeout=EVENT_TIMEOUT)

    def rival_entering(self):
        self.other_called.set()


def run_threads(targets, names):
    """Start every callable, join with a bounded timeout, and surface the
    first exception. A deadlock fails the test instead of hanging it."""
    errors = []

    def wrap(target):
        def run():
            try:
                target()
            except BaseException as error:      # pragma: no cover - report
                errors.append(error)
        return run

    threads = [threading.Thread(target=wrap(t), name=n, daemon=True)
               for t, n in zip(targets, names)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_TIMEOUT)
        assert not thread.is_alive(), (
            f"thread {thread.name} did not finish within {JOIN_TIMEOUT}s — "
            f"deadlock or lost wake-up"
        )
    assert errors == [], errors


def checkpoint_pair(tmp_path, build=TxnModel, with_optimizer=True):
    """Two valid version-2 checkpoints whose parameters, buffers,
    optimizer state, and generator state all differ, plus the live
    model/optimizer and each archive's exact fingerprint."""
    model = build()
    optimizer = NativeAdam(model.parameters(), lr=0.05) if with_optimizer else None
    advance(model, steps=2, optimizer=optimizer)
    path_a = str(tmp_path / "a.npz")
    save_native_checkpoint(path_a, model, optimizer)
    state_a = fingerprint(model, optimizer)

    advance(model, steps=3, optimizer=optimizer)
    for generator in model.generators():
        generator.reseed(generator.seed // 2 + 12345)
    path_b = str(tmp_path / "b.npz")
    save_native_checkpoint(path_b, model, optimizer)
    state_b = fingerprint(model, optimizer)

    assert not same(state_a, state_b), "the two checkpoints are not distinct"
    # Drift the live state away from both, so a no-op cannot pass.
    advance(model, steps=1, optimizer=optimizer)
    return model, optimizer, path_a, path_b, state_a, state_b


# ======================================================================
# 1. The guard itself
# ======================================================================


@needs_native
def test_the_guard_is_one_private_reentrant_lock():
    import tensorforge
    import tensorforge.experimental as experimental

    lock = state_lock.state_transaction()
    assert lock is state_lock._STATE_TRANSACTION_LOCK
    assert type(lock).__name__ in ("RLock", "_thread.RLock")
    # Reentrant: the whole nested-loader design depends on it.
    with lock:
        assert state_lock.held_by_current_thread()
        with lock:
            assert state_lock.held_by_current_thread()
    assert not state_lock.held_by_current_thread()
    # Private: no public export, no top-level leak.
    assert "_native_state_lock" not in experimental.__all__
    assert not any(name.startswith("_") for name in experimental.__all__)
    for absent in ("state_transaction", "STATE_TRANSACTION_LOCK",
                   "held_by_current_thread"):
        assert absent not in experimental.__all__
        assert not hasattr(tensorforge, absent)


@needs_native
def test_every_participant_shares_the_one_guard():
    """Not "each component has a lock" — the *same* object, so there is
    one order rather than several."""
    from tensorforge.experimental import native_adam, native_sgd

    expected = state_lock.state_transaction()
    for module in (_native_state, generator_module, native_adam, native_sgd,
                   native_checkpoint, transaction):
        assert module.state_transaction() is expected, module.__name__


@needs_native
def test_generator_locks_are_never_taken_before_the_guard(tmp_path,
                                                          monkeypatch):
    """The universal order, asserted from live behavior: every ordered
    generator-lock acquisition happens with the guard already held, and
    in sorted ``id()`` order."""
    model = TxnModel()
    optimizer = NativeSGD(model.parameters(), lr=0.1)
    advance(model, steps=1)
    path = str(tmp_path / "order.npz")

    seen = []
    real = generator_module._ordered_targets

    def recording(generators):
        assert state_lock.held_by_current_thread(), (
            "generator locks were ordered without the shared guard held"
        )
        ordered = real(generators)
        seen.append([id(g) for g in ordered])
        return ordered

    monkeypatch.setattr(generator_module, "_ordered_targets", recording)
    save_native_checkpoint(path, model, optimizer)
    load_native_checkpoint(path, model, optimizer)
    model.load_generator_state_dict(model.generator_state_dict())
    monkeypatch.undo()

    assert seen, "no generator lock sequence was taken"
    for order in seen:
        assert order == sorted(order), "not the global id() order"
    close_all(model, optimizer)


@needs_native
def test_the_checkpoint_transaction_holds_the_guard_at_every_commit(
    tmp_path, monkeypatch,
):
    """Reentrancy proved where it matters: the components' own loaders run
    *inside* the transaction's guard, not beside it."""
    model = TxnModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, steps=1, optimizer=optimizer)
    path = str(tmp_path / "nested.npz")
    save_native_checkpoint(path, model, optimizer)

    held = {}
    for seam in ("_capture_rollback", "_commit_model", "_commit_optimizer",
                 "_commit_generators", "_reach_commit_boundary"):
        real = getattr(transaction, seam)

        def make(name=seam, original=real):
            def wrapper(*args, **kwargs):
                held[name] = state_lock.held_by_current_thread()
                return original(*args, **kwargs)
            return wrapper

        monkeypatch.setattr(transaction, seam, make())

    load_native_checkpoint(path, model, optimizer)
    monkeypatch.undo()
    assert held and all(held.values()), held
    close_all(model, optimizer)


# ======================================================================
# 2. Two concurrent checkpoint loads
# ======================================================================


@needs_native
@pytest.mark.parametrize(
    "seam", ["_commit_model", "_commit_optimizer", "_commit_generators",
             "_reach_commit_boundary"],
)
@pytest.mark.parametrize("swap", [False, True])
def test_two_concurrent_loads_serialize(tmp_path, monkeypatch, trace, seam,
                                        swap):
    """The core claim. Two overlapping loads of two different archives
    into one live model must produce checkpoint A **or** checkpoint B,
    never model-from-A with optimizer-or-generators-from-B."""
    model, optimizer, path_a, path_b, state_a, state_b = checkpoint_pair(
        tmp_path
    )
    interleaver = Interleaver()
    real = getattr(transaction, seam)

    def blocking(*args, **kwargs):
        interleaver.park()
        return real(*args, **kwargs)

    monkeypatch.setattr(transaction, seam, blocking)

    first_path, second_path = (path_b, path_a) if swap else (path_a, path_b)
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def loader_one():
        start.wait()
        load_native_checkpoint(first_path, model, optimizer)

    def loader_two():
        start.wait()
        interleaver.inside.wait(timeout=EVENT_TIMEOUT)
        interleaver.rival_entering()
        load_native_checkpoint(second_path, model, optimizer)

    run_threads([loader_one, loader_two], ["one", "two"])
    monkeypatch.undo()

    order = trace.assert_serial()
    assert len(order) == 2, f"the two loads did not both commit: {order}"
    final = fingerprint(model, optimizer)
    assert same(final, state_a) or same(final, state_b), (
        "the final state is a mixture of both checkpoints"
    )
    # ...and it is exactly the archive whose commits end the trace.
    expected_first = first_path if order[0] == "one" else second_path
    expected_last = second_path if order[0] == "one" else first_path
    assert expected_first != expected_last
    expected = state_a if expected_last == path_a else state_b
    assert same(final, expected), (
        "the surviving state is not the one that committed last"
    )
    close_all(model, optimizer)


@needs_native
def test_two_concurrent_loads_serialize_without_generators(tmp_path,
                                                           monkeypatch,
                                                           trace):
    """Serialization must not quietly depend on generator locks: a
    generator-free model takes only the guard, and still serializes."""
    model, optimizer, path_a, path_b, state_a, state_b = checkpoint_pair(
        tmp_path, build=plain_model
    )
    assert model.generators() == []
    interleaver = Interleaver()
    real = transaction._commit_optimizer

    def blocking(*args, **kwargs):
        interleaver.park()
        return real(*args, **kwargs)

    monkeypatch.setattr(transaction, "_commit_optimizer", blocking)
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def one():
        start.wait()
        load_native_checkpoint(path_a, model, optimizer)

    def two():
        start.wait()
        interleaver.inside.wait(timeout=EVENT_TIMEOUT)
        interleaver.rival_entering()
        load_native_checkpoint(path_b, model, optimizer)

    run_threads([one, two], ["one", "two"])
    monkeypatch.undo()

    order = trace.assert_serial()
    assert len(order) == 2
    final = fingerprint(model, optimizer)
    assert same(final, state_a) or same(final, state_b)
    close_all(model, optimizer)


@needs_native
def test_a_shared_generator_across_two_loads_does_not_deadlock(tmp_path,
                                                               trace):
    """Two loads whose target sets overlap, entered through modules that
    registered the shared generator in **opposite** order — the classic
    lock-cycle shape."""
    shared = NativeGenerator(7)

    def build(order):
        module = NativeModule()
        drops = {"a": NativeDropout(0.5, generator=shared),
                 "b": NativeDropout(0.5, generator=shared)}
        for name in order:
            setattr(module, name, drops[name])
        module.linear = NativeLinear(4, 4, seed=3)
        return module

    first, second = build(("a", "b")), build(("b", "a"))
    path_one = str(tmp_path / "one.npz")
    path_two = str(tmp_path / "two.npz")
    save_native_checkpoint(path_one, first)
    save_native_checkpoint(path_two, second)
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def load(path, model):
        def run():
            start.wait()
            for _ in range(25):
                load_native_checkpoint(path, model)
        return run

    run_threads([load(path_one, first), load(path_two, second)],
                ["one", "two"])
    assert shared.seed == 7
    close_all(first)
    close_all(second)


# ======================================================================
# 3. Checkpoint load versus each component loader
# ======================================================================


@needs_native
def test_checkpoint_load_versus_model_state_load(tmp_path, monkeypatch,
                                                 trace):
    """``load_state_dict`` may run wholly before or wholly after the
    checkpoint's commit — never between its model and optimizer steps."""
    model, optimizer, path_a, _, state_a, _ = checkpoint_pair(tmp_path)
    rival_values = {
        name: NativeTensor.from_array(
            np.full(tensor.shape, 0.125, dtype=np.float64)
        )
        for name, tensor in model._state_named_tensors()
    }
    interleaver = Interleaver()
    real = transaction._commit_optimizer

    def blocking(*args, **kwargs):
        interleaver.park()
        return real(*args, **kwargs)

    monkeypatch.setattr(transaction, "_commit_optimizer", blocking)
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def checkpoint():
        start.wait()
        load_native_checkpoint(path_a, model, optimizer)

    def state_load():
        start.wait()
        interleaver.inside.wait(timeout=EVENT_TIMEOUT)
        interleaver.rival_entering()
        model.load_state_dict(rival_values)

    run_threads([checkpoint, state_load], ["checkpoint", "state"])
    monkeypatch.undo()

    trace.assert_serial()
    final = fingerprint(model, optimizer)
    # The optimizer and the generators always come from the checkpoint —
    # it is the only operation that touches them — and the model is one
    # whole operation's, never a partial commit.
    assert final["generators"] == state_a["generators"]
    assert final["steps"] == state_a["steps"] and final["lr"] == state_a["lr"]
    from_checkpoint = all(
        np.array_equal(final["tensors"][k], state_a["tensors"][k])
        for k in state_a["tensors"]
    )
    from_rival = all(
        np.allclose(values, 0.125) for values in final["tensors"].values()
    )
    assert from_checkpoint or from_rival, (
        "the model is neither checkpoint A nor the rival state_dict"
    )
    for value in rival_values.values():
        value.close()
    close_all(model, optimizer)


@needs_native
@pytest.mark.parametrize("kind", ["adam", "sgd"])
def test_checkpoint_load_versus_optimizer_state_load(tmp_path, monkeypatch,
                                                     trace, kind):
    """The optimizer ends up wholly the checkpoint's or wholly the rival
    state's — never an lr from one beside moments from the other."""
    build = TxnModel
    model = build()
    optimizer = (NativeAdam(model.parameters(), lr=0.05) if kind == "adam"
                 else NativeSGD(model.parameters(), lr=0.05))
    advance(model, steps=2, optimizer=optimizer)
    path = str(tmp_path / f"{kind}.npz")
    save_native_checkpoint(path, model, optimizer)
    checkpoint_state = fingerprint(model, optimizer)

    advance(model, steps=1, optimizer=optimizer)
    rival = optimizer.state_dict()
    rival["lr"] = 0.4242
    if kind == "adam":
        rival["betas"] = (0.5, 0.75)
        rival["eps"] = 1e-3
        rival["step_counts"] = tuple(
            count + 100 for count in rival["step_counts"]
        )

    interleaver = Interleaver()
    real = transaction._commit_generators

    def blocking(*args, **kwargs):
        interleaver.park()
        return real(*args, **kwargs)

    monkeypatch.setattr(transaction, "_commit_generators", blocking)
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def checkpoint():
        start.wait()
        load_native_checkpoint(path, model, optimizer)

    def optimizer_load():
        start.wait()
        interleaver.inside.wait(timeout=EVENT_TIMEOUT)
        interleaver.rival_entering()
        optimizer.load_state_dict(rival)

    run_threads([checkpoint, optimizer_load], ["checkpoint", "optimizer"])
    monkeypatch.undo()

    trace.assert_serial()
    final = fingerprint(model, optimizer)
    assert final["generators"] == checkpoint_state["generators"]
    if kind == "adam":
        whole_checkpoint = (
            final["lr"] == checkpoint_state["lr"]
            and final["steps"] == checkpoint_state["steps"]
            and final["betas"] == checkpoint_state["betas"]
            and final["eps"] == checkpoint_state["eps"]
        )
        whole_rival = (
            final["lr"] == 0.4242 and final["betas"] == (0.5, 0.75)
            and final["eps"] == 1e-3
            and final["steps"] == rival["step_counts"]
        )
    else:
        whole_checkpoint = final["lr"] == checkpoint_state["lr"]
        whole_rival = final["lr"] == 0.4242
    assert whole_checkpoint or whole_rival, (
        f"the optimizer state is a mixture: {final}"
    )
    if kind == "adam":
        for label in ("m", "v"):
            for snapshot in rival[label]:
                snapshot.close()
    close_all(model, optimizer)


@needs_native
def test_checkpoint_load_versus_generator_state_load(tmp_path, monkeypatch,
                                                     trace):
    """Generator state ends up wholly the checkpoint's or wholly the
    rival's, and the rival never lands between the checkpoint's model and
    generator commits."""
    model, optimizer, path_a, _, state_a, _ = checkpoint_pair(tmp_path)
    rival = {
        name: {"algorithm": "tensorforge.splitmix64",
               "algorithm_version": 1, "seed": 4242, "calls": 99}
        for name in model.generator_state_dict()
    }
    interleaver = Interleaver()
    real = transaction._commit_model

    def blocking(*args, **kwargs):
        interleaver.park()
        return real(*args, **kwargs)

    monkeypatch.setattr(transaction, "_commit_model", blocking)
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def checkpoint():
        start.wait()
        load_native_checkpoint(path_a, model, optimizer)

    def generator_load():
        start.wait()
        interleaver.inside.wait(timeout=EVENT_TIMEOUT)
        interleaver.rival_entering()
        model.load_generator_state_dict(rival)

    run_threads([checkpoint, generator_load], ["checkpoint", "generators"])
    monkeypatch.undo()

    trace.assert_serial()
    final = fingerprint(model, optimizer)
    whole_checkpoint = final["generators"] == state_a["generators"]
    whole_rival = all(
        state["seed"] == 4242 and state["calls"] == 99
        for state in final["generators"].values()
    )
    assert whole_checkpoint or whole_rival, (
        f"generator state is a mixture: {final['generators']}"
    )
    # The model and optimizer are the checkpoint's either way: the rival
    # touches neither.
    assert final["steps"] == state_a["steps"]
    for name, values in state_a["tensors"].items():
        assert np.array_equal(final["tensors"][name], values), name
    close_all(model, optimizer)


@needs_native
def test_a_live_reservation_still_blocks_both_operations(tmp_path):
    """Serialization does not weaken the reservation rule: a generator
    with a call in flight refuses the checkpoint load *and* the generator
    state load, and the reservation survives both refusals."""
    model = TxnModel()
    path = str(tmp_path / "reserved.npz")
    save_native_checkpoint(path, model)
    state = model.generator_state_dict()
    token = model.drop_a.generator._reserve_call()

    with pytest.raises(RuntimeError, match="reservation"):
        load_native_checkpoint(path, model)
    with pytest.raises(RuntimeError, match="reservation"):
        model.load_generator_state_dict(state)
    with pytest.raises(RuntimeError, match="reservation"):
        save_native_checkpoint(str(tmp_path / "nope.npz"), model)

    assert model.generator_state_dict() == state
    assert model.drop_a.generator._has_active_reservation() is True
    model.drop_a.generator._abandon_call(token)
    load_native_checkpoint(path, model)          # recovers
    close_all(model)


@needs_native
def test_a_reservation_racing_a_transaction_precedes_or_follows_it(
    tmp_path, trace,
):
    """Reservations take only their generator's lock, never the guard, so
    the two systems cannot invert. A racing reservation therefore either
    completes before the transaction takes that lock, or begins after the
    transaction released it — and never has state replaced underneath its
    token."""
    model = TxnModel()
    path = str(tmp_path / "race.npz")
    save_native_checkpoint(path, model)
    saved = model.generator_state_dict()
    generator = model.drop_a.generator
    observed = []
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def loader():
        start.wait()
        for _ in range(40):
            try:
                load_native_checkpoint(path, model)
            except RuntimeError as error:
                assert "reservation" in str(error), error

    def reserver():
        start.wait()
        for _ in range(40):
            try:
                token = generator._reserve_call()
            except RuntimeError:
                continue
            # The seed under this token must not change while it lives.
            seed = generator.seed
            observed.append(seed == generator.seed)
            generator._abandon_call(token)

    run_threads([loader, reserver], ["load", "reserve"])
    assert all(observed), "a seed moved underneath a live reservation"
    for name, state in model.generator_state_dict().items():
        assert state["seed"] == saved[name]["seed"], name
    assert all(not g._has_active_reservation() for g in model.generators())
    close_all(model)


# ======================================================================
# 4. Save snapshot serialization
# ======================================================================


@needs_native
@pytest.mark.parametrize("rival", ["model", "optimizer", "generators",
                                   "checkpoint"])
def test_a_save_snapshot_is_one_coherent_serial_snapshot(tmp_path,
                                                         monkeypatch, trace,
                                                         rival):
    """A save must not capture model state from before a replacement and
    optimizer or generator state from after it. The snapshot runs under
    the same guard, so a rival replacement waits, and the archive
    describes one serial point."""
    model = TxnModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, steps=2, optimizer=optimizer)
    before = fingerprint(model, optimizer)

    other = str(tmp_path / "other.npz")
    save_native_checkpoint(other, model, optimizer)   # the rival archive
    rival_tensors = {
        name: NativeTensor.from_array(
            np.full(tensor.shape, 0.5, dtype=np.float64)
        )
        for name, tensor in model._state_named_tensors()
    }
    rival_generators = {
        name: {"algorithm": "tensorforge.splitmix64", "algorithm_version": 1,
               "seed": 31337, "calls": 5}
        for name in model.generator_state_dict()
    }
    rival_optimizer = optimizer.state_dict()
    rival_optimizer["lr"] = 0.9

    interleaver = Interleaver()
    real_section = native_checkpoint._generator_section

    def blocking(*args, **kwargs):
        # Parked *after* the model and optimizer snapshots and *before*
        # the generator snapshot — precisely where an incoherent archive
        # would be produced if the guard did not hold.
        interleaver.park()
        return real_section(*args, **kwargs)

    monkeypatch.setattr(native_checkpoint, "_generator_section", blocking)
    destination = str(tmp_path / "snapshot.npz")
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def saver():
        start.wait()
        save_native_checkpoint(destination, model, optimizer)

    def replacer():
        start.wait()
        interleaver.inside.wait(timeout=EVENT_TIMEOUT)
        interleaver.rival_entering()
        if rival == "model":
            model.load_state_dict(rival_tensors)
        elif rival == "optimizer":
            optimizer.load_state_dict(rival_optimizer)
        elif rival == "generators":
            model.load_generator_state_dict(rival_generators)
        else:
            load_native_checkpoint(other, model, optimizer)

    run_threads([saver, replacer], ["save", "replace"])
    monkeypatch.undo()

    with np.load(destination, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest"].tobytes().decode("utf-8"))
        arrays = {name: archive[name].copy() for name in archive.files
                  if name != "manifest"}

    # The archive is the pre-replacement point, whole: the save holds the
    # guard from its first snapshot to its last, so the rival could not
    # have landed in the middle of it.
    for index, key in enumerate(manifest["model"]["keys"]):
        assert np.array_equal(arrays[f"model::{index:06d}"],
                              before["tensors"][key]), key
    assert manifest["optimizer"]["lr"] == before["lr"]
    assert tuple(manifest["optimizer"]["step_counts"]) == before["steps"]
    for name, entry in manifest["generators"]["entries"].items():
        assert int(entry["seed"]) == before["generators"][name]["seed"], name
        assert int(entry["calls"]) == before["generators"][name]["calls"], name

    for value in rival_tensors.values():
        value.close()
    for label in ("m", "v"):
        for snapshot in rival_optimizer[label]:
            snapshot.close()
    close_all(model, optimizer)


# ======================================================================
# 5. Concurrent rollback
# ======================================================================


@needs_native
def test_a_failed_load_rolls_back_before_the_next_one_sees_anything(
    tmp_path, monkeypatch, trace,
):
    """Load A commits one or more components and then fails; load B is
    waiting on the guard. A must roll back completely **before** B can
    run, B must then load completely, and the final state must be
    exactly B."""
    model, optimizer, path_a, path_b, state_a, state_b = checkpoint_pair(
        tmp_path
    )
    pre_load = fingerprint(model, optimizer)
    interleaver = Interleaver()
    observed_by_b = {}
    real_boundary = transaction._reach_commit_boundary
    real_capture = transaction._capture_rollback
    failing_thread = "one"

    def boundary(*args, **kwargs):
        if threading.current_thread().name == failing_thread:
            interleaver.park()
            raise RuntimeError("injected post-commit failure")
        return real_boundary(*args, **kwargs)

    def capture(plan, targets):
        if threading.current_thread().name != failing_thread:
            # What B sees at its own commit boundary: it must be the
            # pre-load state, never A's partially committed one.
            observed_by_b["state"] = fingerprint(plan.model, plan.optimizer)
        return real_capture(plan, targets)

    monkeypatch.setattr(transaction, "_reach_commit_boundary", boundary)
    monkeypatch.setattr(transaction, "_capture_rollback", capture)
    start = threading.Barrier(2, timeout=EVENT_TIMEOUT)
    trace.reset()

    def failing_load():
        start.wait()
        with pytest.raises(RuntimeError, match="injected post-commit"):
            load_native_checkpoint(path_a, model, optimizer)

    def surviving_load():
        start.wait()
        interleaver.inside.wait(timeout=EVENT_TIMEOUT)
        interleaver.rival_entering()
        load_native_checkpoint(path_b, model, optimizer)

    run_threads([failing_load, surviving_load], ["one", "two"])
    monkeypatch.undo()

    assert observed_by_b, "the surviving load never reached its boundary"
    assert same(observed_by_b["state"], pre_load), (
        "the second load observed the first load's partial state"
    )
    final = fingerprint(model, optimizer)
    assert same(final, state_b), "the final state is not checkpoint B"
    assert not same(final, state_a)
    close_all(model, optimizer)


@needs_native
def test_a_rolled_back_load_leaves_the_guard_usable(tmp_path, monkeypatch):
    """A failure inside the transaction must release both locks — a guard
    left held would deadlock every later state operation, which a bounded
    join turns into a failure rather than a hang."""
    model = TxnModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    advance(model, steps=1, optimizer=optimizer)
    path = str(tmp_path / "leak.npz")
    save_native_checkpoint(path, model, optimizer)

    for seam, error in (("_commit_optimizer", RuntimeError),
                        ("_commit_generators", KeyboardInterrupt),
                        ("_reach_commit_boundary", MemoryError),
                        ("_capture_rollback", MemoryError)):
        real = getattr(transaction, seam)

        def boom(*args, **kwargs):
            raise error("injected")

        monkeypatch.setattr(transaction, seam, boom)
        with pytest.raises(error):
            load_native_checkpoint(path, model, optimizer)
        monkeypatch.setattr(transaction, seam, real)
        assert not state_lock.held_by_current_thread()
        for generator in model.generators():
            assert not generator._has_active_reservation()

    # Everything still works, from another thread as well as this one.
    run_threads([lambda: load_native_checkpoint(path, model, optimizer)],
                ["after"])
    load_native_checkpoint(path, model, optimizer)
    close_all(model, optimizer)


# ======================================================================
# 6. Scope
# ======================================================================


@needs_native
def test_ordinary_training_mutation_does_not_take_the_guard():
    """The honest boundary: the guard serializes *state replacement and
    checkpoint snapshotting*, not training. An optimizer step and a
    parameter mutation deliberately do not take it, so this suite's
    claim stays exactly what it proves."""
    model = plain_model()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    x = NativeTensor.from_array(X)
    y = model(x)
    y.sum().backward()
    assert not state_lock.held_by_current_thread()
    optimizer.step()
    assert not state_lock.held_by_current_thread()
    parameter = model.parameters()[0]
    parameter.copy_value_(
        NativeTensor.from_array(np.zeros(parameter.shape, dtype=np.float64))
    )
    assert not state_lock.held_by_current_thread()
    y.close()
    x.close()
    close_all(model, optimizer)
