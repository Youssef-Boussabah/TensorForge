"""The cross-cutting adversarial matrix over the native data pipeline
(Phase J, milestone J7; docs/native_data_pipeline_design.md §4-§6, §7,
§9.1-§9.7, §10, §11, §12.1-§12.7, §13, §14, §15.1-§15.6, §16.1-§16.3,
§17.1-§17.5, §18, §19, §20, §21).

J1-J6 each proved their own object. **This module proves what happens
when every one of them is attacked at once**, and it proves it the way
I10 did: with a complete before/after fingerprint of the observable
world around every rejection and every injected failure, and with an
independent non-vacuity control for every injection and every parser.

The one sentence the whole module exists to defend:

    **A committed batch position moves if and only if the caller
    received the batch — and a request that is refused, or a delivery
    that fails, leaves the entire observable world exactly as it found
    it.**

What is proved here, and why it is not a duplicate of J3/J4/J5
---------------------------------------------------------------

* **§12.7 in full.** J4 proved a rejected *loader* load mutates nothing.
  J7 proves the whole §12.7 list — no consumed position, no claim or
  pending record, no unclosed ``NativeTensor``, no persistent native
  allocation, no dataset/sampler/loader mutation, **no
  ``NativeParameter``, buffer, version, gradient, optimizer, or
  registered ``NativeGenerator`` touched**, no file written, and no
  global or module state changed — around *every* rejection family, with
  one reusable fingerprint rather than a per-test spot check.
* **§17.2 and §17.3 row by row.** J3 injected most of the iteration
  rows; J7 adds the ones it did not and re-asserts every one under the
  full fingerprint. In particular the **commit step is made to fail
  after the candidate position has really been applied** — J3's
  injection raises *instead of* the assignment, J7's raises *after* it —
  and the host gather, the native allocation, the host→native transfer,
  and the target copy are separated into four distinct injections rather
  than one.
* **A ``BaseException`` at the seam**, not only an ``Exception``, so the
  unconditional ``finally`` is proved unconditional.
* **A checkpoint taken immediately after a failed delivery**, through a
  real version-3 archive, restored into a wholly fresh model, optimizer,
  generator, dataset, sampler, and loader, and proved to resume the
  **same candidate batch** with bit-identical features and targets.
* **§16 as a boundary rather than a feature**: reentrancy is refused
  deterministically from real same-thread seams; concurrency is asserted
  *documented and unprotected*. **No test here starts a thread, and no
  lock exists to find.**

Discipline, inherited from J6 and I10 and not relaxed anywhere here:
numeric equality is raw IEEE-754 bits through ``uint32``/``uint64``
views, never a tolerance; each dtype is compared only against itself;
every injection has a non-vacuity control proving the patched path
really ran; every parser has a negative control proving it can fail; no
test asserts an exact error message, a dict ordering, a timing, a speed,
or a garbage-collection event.

**J7 adds no production code.** Every seam it uses already exists and is
private — ``_deliver_batch``, ``_claim_batch``, ``_publish_pending``,
``_commit_pending``, ``_assign_state``, ``_begin_iteration``,
``feature_batch``/``target_batch``, and the backend's existing
thread-local allocation-failure arm. No production fault hook, no public
transaction inspector, no live-storage counter, and no test-only export
was added for it.

Selector: python -m pytest -q tests/test_native_data_hardening.py
"""

import ast
import collections
import gc
import inspect
import json
import os
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import tensorforge
import tensorforge.experimental as experimental
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam, NativeBatchNorm1d, NativeBatchSampler, NativeDataLoader,
    NativeDropout, NativeGenerator, NativeLinear, NativeModule,
    NativeParameter, NativeReLU, NativeSGD, NativeTensor,
    NativeTensorDataset, load_native_checkpoint, save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint as checkpoint_module
from tensorforge.experimental import native_data_loader as loader_module
from tensorforge.experimental import native_dataset as dataset_module
from tensorforge.experimental import native_optimizer_state as optimizer_state
from tensorforge.experimental import native_sampler as sampler_module

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "src" / "tensorforge" / "experimental"

# The four Phase-J production modules. Every source-level absence check in
# this module runs over exactly these, and over nothing else — a scan that
# quietly widened to the whole package would stop being about Phase J.
PIPELINE_MODULES = ("native_dataset.py", "native_sampler.py",
                    "native_data_loader.py", "_native_permutation.py")

needs_backend = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="the build has no deterministic allocation-failure arm",
)


# ===========================================================================
# 0. Fixtures, builders, and the reusable world fingerprint
# ===========================================================================

@pytest.fixture(autouse=True)
def _disarm_allocation_faults():
    """No injected allocation failure survives a test, whatever it did.

    Armed **and** disarmed here as well as in every arming test's own
    ``finally``, so a test that dies between the two cannot leave the
    backend armed for the next one (§"Allocation-failure strategy").
    """
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages():
    """The ids of every ``NativeStorage`` currently open — the project's
    deterministic instrumentation for native-allocation lifetime, used
    unchanged since Phase C. **There is no public counter and J7 adds
    none.**

    Installed with an explicit save/restore rather than through
    ``monkeypatch``, deliberately: many tests here call
    ``monkeypatch.undo()`` in the middle to take their injection back out,
    and a tracker installed through the same ``monkeypatch`` would be
    silently uninstalled with it — leaving every later ``close()``
    unrecorded and every live-storage assertion vacuous.
    """
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    cpp.NativeStorage.__init__ = tracked_init
    cpp.NativeStorage.close = tracked_close
    try:
        yield open_ids
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


def test_the_live_storage_tracker_survives_a_monkeypatch_undo(monkeypatch,
                                                              live_storages):
    """Negative control for the fixture above, and for every live-storage
    assertion in this module: a mid-test ``monkeypatch.undo()`` must not
    disarm the tracker."""
    if not cpp.is_available():
        pytest.skip("the tracker needs the built backend to allocate")
    monkeypatch.setattr(loader_module, "_FORMAT_VERSION", 1)
    monkeypatch.undo()
    baseline = settled(live_storages)
    tensor = NativeTensor.from_array(np.zeros((2, 2)))
    assert settled(live_storages) == baseline + 1
    tensor.close()
    assert settled(live_storages) == baseline


def settled(live_storages):
    """The live-storage count after a collection. Collection *settles* the
    count; it is never the proof that anything was released — every test
    here closes what it owns explicitly first."""
    gc.collect()
    return len(live_storages)


def bits(values):
    """Raw IEEE-754 bits of a host array, as unsigned integers of the
    matching width. **Never a tolerance**: float32 through ``uint32``,
    float64 through ``uint64``, each compared only against itself."""
    array = np.ascontiguousarray(values)
    if array.dtype == np.float32:
        return array.view(np.uint32).tolist()
    assert array.dtype == np.float64, array.dtype
    return array.view(np.uint64).tolist()


def host_arrays(samples=8, width=2):
    """Deterministic source values whose every row identifies itself, so a
    batch can be checked against the indices that produced it."""
    features = (np.arange(samples * width, dtype=np.float64)
                .reshape(samples, width))
    targets = np.arange(samples, dtype=np.int64) % 3
    return features, targets


def make_dataset(samples=8, width=2, dtype=None, offset=0.0):
    features, targets = host_arrays(samples, width)
    return NativeTensorDataset(features + offset, targets, dtype=dtype)


def make_loader(samples=8, dataset=None, width=2, dtype=None, **kwargs):
    """``(loader, sampler, dataset)`` so a test can reach every level."""
    kwargs.setdefault("batch_size", 3)
    if dataset is None:
        dataset = make_dataset(samples, width, dtype)
    sampler = NativeBatchSampler(dataset, **kwargs)
    return NativeDataLoader(sampler), sampler, dataset


def position(sampler):
    return (sampler.epoch, sampler.cursor)


# --- the fingerprint, in four object-shaped halves plus two global ones ----
#
# Semantic helper functions rather than one opaque blob, so a harmless
# internal reformatting does not force a rewrite of the matrix, and so a
# failure names the component that moved.

def dataset_view(dataset, materialize=None):
    """Everything §4-§6 makes observable about a dataset.

    ``materialize`` is an optional index tuple whose batch must still be
    producible; its **bits** join the fingerprint, so "the dataset can
    still answer the next legal request, identically" is part of the
    comparison rather than a separate assertion.
    """
    view = {
        "id": id(dataset),
        "closed": dataset.closed,
        "samples": dataset.samples,
        "len": len(dataset),
        "feature_shape": dataset.feature_shape,
        "dtype": dataset.dtype,
        "device": dataset.device,
        "fingerprint": dataset.fingerprint,
        "identity": dataset.identity(),
        "repr": repr(dataset),
    }
    if materialize is not None and not dataset.closed:
        features = dataset.feature_batch(materialize)
        try:
            view["batch_bits"] = bits(features.to_numpy())
            view["batch_shape"] = features.shape
        finally:
            features.close()
        view["batch_targets"] = dataset.target_batch(materialize).tolist()
    return view


def sampler_view(sampler):
    """Everything §7/§11.2 makes observable about a sampler, plus the
    private bookkeeping §12.7 says a rejection may not disturb.

    The permutation cache is included **by behavior and by key**: the key
    proves no drift, and ``epoch_permutation()`` proves the value the
    cache would serve is still the one a pure re-derivation gives.
    """
    return {
        "id": id(sampler),
        "dataset_id": id(sampler.dataset),
        "batch_size": sampler.batch_size,
        "shuffle": sampler.shuffle,
        "seed": sampler.seed,
        "drop_last": sampler.drop_last,
        "epoch": sampler.epoch,
        "cursor": sampler.cursor,
        "batches_per_epoch": sampler.batches_per_epoch,
        "remaining": sampler.remaining,
        "next_batch_indices": sampler.next_batch_indices(),
        "epoch_permutation": sampler.epoch_permutation(),
        "plan": sampler.plan(),
        "state": None if sampler._has_transaction() else sampler.state_dict(),
        "repr": repr(sampler),
        # Private, and read only to prove nothing moved (§12.7).
        "cache_key": sampler._cache_key,
        "cache_order": sampler._cache_order,
        "transaction": transaction_view(sampler._transaction),
        "next_serial": sampler._next_serial,
        "active_iterations": frozenset(sampler._active_iterations),
        "next_token": sampler._next_iteration_token,
    }


def transaction_view(transaction):
    """The integer half of an in-flight record, or ``None``."""
    if transaction is None:
        return None
    return (transaction.serial, transaction.owner, transaction.status,
            transaction.before, transaction.after, transaction.indices)


def loader_view(loader):
    """Everything §3.5/§11.3/§15 makes observable about a loader."""
    return {
        "id": id(loader),
        "sampler_id": id(loader.sampler),
        "dataset_id": id(loader.dataset),
        "closed": loader.closed,
        "state": (None if loader.sampler._has_transaction()
                  else loader.state_dict()),
        "repr": repr(loader),
        "iterator_id": id(loader._iterator),
    }


def iterator_view(iterator):
    """The iterator's public status and the private slots §15 governs."""
    if iterator is None:
        return None
    return {
        "id": id(iterator),
        "closed": iterator._closed,
        "superseded": iterator._superseded,
        "exhausted": iterator._exhausted,
        "to_yield": iterator._to_yield,
        "txn_serial": iterator._txn_serial,
        "has_features": iterator._features is not None,
        "has_targets": iterator._targets is not None,
        "token": iterator._token,
        "is_current": iterator._loader._iterator is iterator,
        "participating": iterator._token in iterator._sampler._active_iterations,
        "repr": repr(iterator),
    }


class Sentinels:
    """Unrelated native state that a Phase-J rejection may never touch
    (§12.7): a registered parameter with a gradient, a persistent buffer,
    a registered generator with a nontrivial call count, and a live
    optimizer holding moments for that parameter.

    Held together only so a test can build one and close it explicitly.
    No production analogue exists or is implied.
    """

    __slots__ = ("model", "optimizer", "generator", "parameter")

    def __init__(self):
        self.model = SentinelModel()
        self.generator = self.model.drop.generator
        self.parameter = self.model.linear.weight
        self.optimizer = NativeAdam(self.model.parameters(), lr=0.01)
        # Give every family something nontrivial to preserve: a real
        # gradient, real Adam moments, real running statistics, and a
        # generator that has actually drawn.
        self.model.train()
        features = NativeTensor.from_array(
            np.linspace(-1.0, 1.0, 12).reshape(3, 4))
        try:
            out = self.model(features)
            loss = out.sum()
            loss.backward()
            self.optimizer.step()
            loss.close()
            out.close()
        finally:
            features.close()

    def close(self):
        """Explicit cleanup in the established order: the optimizer's
        moments, then every unique parameter and buffer. Nothing here
        relies on garbage collection."""
        self.optimizer.close()
        seen = set()
        for _, tensor in (list(self.model.named_parameters())
                          + list(self.model.named_buffers())):
            if tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                tensor.close()


class SentinelModel(NativeModule):
    """Trainable parameters, a persistent buffer, and a registered
    generator — one object covering every §12.7 family at once."""

    def __init__(self):
        super().__init__()
        self.linear = NativeLinear(4, 3, seed=11)
        self.norm = NativeBatchNorm1d(3)
        self.relu = NativeReLU()
        self.drop = NativeDropout(0.25, seed=909)

    def forward(self, x):
        return self.drop(self.relu(self.norm(self.linear(x))))


def sentinel_view(sentinels):
    """Every §12.7 family, by value and by version."""
    if sentinels is None:
        return None
    model = sentinels.model
    parameters = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        parameters[name] = (
            bits(parameter.to_numpy()),
            parameter.version,
            parameter.requires_grad,
            None if grad is None else bits(grad.to_numpy()),
        )
    buffers = {}
    for name, buffer in model.named_buffers():
        buffers[name] = (bits(buffer.to_numpy()) if buffer is not None
                         else None)
    return {
        "parameters": parameters,
        "buffers": buffers,
        "generator": sentinels.generator.state(),
        "generator_seed": sentinels.generator.seed,
        "generator_calls": sentinels.generator.calls,
        "optimizer": json.dumps(sentinels.optimizer.state_dict(),
                                sort_keys=True, default=repr),
        "training": model.training,
    }


def registry_view():
    """The capability, schema, and export inventories no Phase-J operation
    may move (§3, §11, §13.1, §22.3)."""
    return {
        "supported_dtypes": cpp.SUPPORTED_DTYPES,
        "supported_devices": cpp.SUPPORTED_DEVICES,
        "unsupported": cpp.UNSUPPORTED,
        "raw_kernel_dtypes": cpp.RAW_KERNEL_DTYPES,
        "default_dtype": cpp.normalize_dtype(None),
        "backend_info": json.dumps(cpp.backend_info(), sort_keys=True,
                                   default=repr),
        "checkpoint_format": checkpoint_module._FORMAT,
        "checkpoint_version": checkpoint_module._FORMAT_VERSION,
        "checkpoint_versions": checkpoint_module._SUPPORTED_FORMAT_VERSIONS,
        "optimizer_state_version": optimizer_state.FORMAT_VERSION,
        "loader_format": loader_module._FORMAT,
        "loader_version": loader_module._FORMAT_VERSION,
        "loader_versions": loader_module._SUPPORTED_FORMAT_VERSIONS,
        "loader_fields": loader_module._STATE_FIELDS,
        "sampler_format": sampler_module._FORMAT,
        "sampler_version": sampler_module._FORMAT_VERSION,
        "sampler_versions": sampler_module._SUPPORTED_FORMAT_VERSIONS,
        "sampler_fields": sampler_module._STATE_FIELDS,
        "dataset_fields": sampler_module._DATASET_FIELDS,
        "identity_dtypes": sampler_module._IDENTITY_DTYPES,
        "experimental_all": tuple(experimental.__all__),
        "stable_all": tuple(tensorforge.__all__),
    }


def globals_view(directory=None):
    """The process-level state §12.7 says a Phase-J operation never
    touches. Deliberately **not** a claim of complete process purity: it
    names the globals the design actually says the phase does not use.

    Garbage-collection timing, allocator internals, and object ids of
    unrelated objects are excluded on purpose — they are not contracts.
    """
    numpy_state = np.random.get_state()
    view = {
        "python_random": random.getstate(),
        "numpy_random": (numpy_state[0], numpy_state[1].tolist(),
                         numpy_state[2], numpy_state[3], numpy_state[4]),
        "environ": dict(os.environ),
        "cwd": os.getcwd(),
        "registries": registry_view(),
    }
    if directory is not None:
        view["files"] = sorted(
            (str(path.relative_to(directory)), path.stat().st_size)
            for path in Path(directory).rglob("*") if path.is_file()
        )
    return view


def world(*, loader=None, sampler=None, dataset=None, iterator=None,
          sentinels=None, materialize=None, directory=None):
    """One comparable snapshot of everything a rejected or failed
    operation must leave untouched."""
    return {
        "dataset": None if dataset is None else dataset_view(dataset,
                                                             materialize),
        "sampler": None if sampler is None else sampler_view(sampler),
        "loader": None if loader is None else loader_view(loader),
        "iterator": iterator_view(iterator),
        "sentinels": sentinel_view(sentinels),
        "globals": globals_view(directory),
    }


def without_sampler_keys(snapshot, *keys):
    """A world snapshot with named sampler fields removed.

    Used for exactly one contracted exception. A **failed delivery** mints
    a serial and never gives it back — the never-reused rule of §9.4,
    Phase 1 — so ``next_serial`` legitimately advances across one. Every
    other field must still be identical, and the serial itself is asserted
    separately and explicitly rather than quietly excluded.
    """
    trimmed = dict(snapshot)
    sampler = dict(trimmed["sampler"])
    for key in keys:
        sampler.pop(key)
    trimmed["sampler"] = sampler
    return trimmed


def code_identifiers(relative):
    """Every identifier a module's **executable code** names.

    A source-text scan would be wrong: these modules explain at length
    what they deliberately do *not* do, so a prose mention of
    ``threading`` would fail a substring check that is supposed to be
    about behavior. Reading the AST asks the question that was meant, and
    it reads keyword-argument names too.
    """
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.add((node.module or "").split(".")[0])
            for alias in node.names:
                names.add(alias.name)
    return names


class Boom(Exception):
    """The one injected failure type, so a test can never mistake an
    accidental production error for its own injection."""


class Abort(BaseException):
    """A ``BaseException`` that is deliberately **not** an ``Exception``,
    so the unconditional rollback is proved unconditional rather than
    proved against the easy case."""


class GatherBomb:
    """Stands in for a dataset snapshot whose fancy-index gather raises.

    This is how the **M1 host gather** is made to fail *distinctly* from
    the native allocation and from the transfer: nothing native has been
    touched when it raises, which is exactly the §17.3 row it stands for.
    Test-side only — it replaces a private slot for the duration of one
    call and is restored in a ``finally``.
    """

    __slots__ = ("calls",)

    def __init__(self):
        self.calls = []

    def __getitem__(self, key):
        self.calls.append(tuple(key) if isinstance(key, list) else key)
        raise Boom("injected: host gather")


# ===========================================================================
# 1. The world fingerprint is itself non-vacuous
# ===========================================================================
#
# Every rejection test below is an equality between two fingerprints. That
# is evidence only if the fingerprint can actually *change*, so each
# component is proved able to notice the mutation it exists to notice.

@needs_backend
def test_the_dataset_component_notices_every_dataset_change():
    dataset = make_dataset(8)
    other = make_dataset(8, offset=1.0)
    try:
        base = dataset_view(dataset, materialize=(0, 1))
        assert dataset_view(dataset, materialize=(0, 1)) == base
        assert dataset_view(other, materialize=(0, 1)) != base
        assert dataset_view(dataset, materialize=(2, 3)) != base
        dataset.close()
        assert dataset_view(dataset) != base
    finally:
        dataset.close()
        other.close()


@needs_backend
@pytest.mark.parametrize("mutate", [
    pytest.param(lambda s: s._assign_state(s.seed + 1, s.shuffle,
                                           s.batch_size, s.drop_last,
                                           s.epoch, s.cursor), id="seed"),
    pytest.param(lambda s: s._assign_state(s.seed, not s.shuffle,
                                           s.batch_size, s.drop_last,
                                           s.epoch, s.cursor), id="shuffle"),
    pytest.param(lambda s: s._assign_state(s.seed, s.shuffle, 2,
                                           s.drop_last, s.epoch, 0),
                 id="batch_size"),
    pytest.param(lambda s: s._assign_state(s.seed, s.shuffle, s.batch_size,
                                           not s.drop_last, s.epoch,
                                           s.cursor), id="drop_last"),
    pytest.param(lambda s: s._assign_state(s.seed, s.shuffle, s.batch_size,
                                           s.drop_last, s.epoch + 1,
                                           s.cursor), id="epoch"),
    pytest.param(lambda s: s._assign_state(s.seed, s.shuffle, s.batch_size,
                                           s.drop_last, s.epoch, 1),
                 id="cursor"),
    pytest.param(lambda s: s._begin_iteration(), id="active_iteration"),
])
def test_the_sampler_component_notices_every_field(mutate):
    """Negative control for the sampler half of the fingerprint: each of
    the six configuration and position fields, and the private
    active-iteration bookkeeping, must be visible to it."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    try:
        base = sampler_view(sampler)
        assert sampler_view(sampler) == base
        mutate(sampler)
        assert sampler_view(sampler) != base
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_the_loader_and_iterator_components_notice_their_changes():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    try:
        base = loader_view(loader)
        assert loader_view(loader) == base
        iterator = iter(loader)
        assert loader_view(loader) != base          # the iterator slot moved
        iterator_base = iterator_view(iterator)
        assert iterator_view(iterator) == iterator_base
        features, _ = next(iterator)
        features.close()
        assert iterator_view(iterator) != iterator_base   # countdown moved
        superseded = iterator
        iter(loader)
        assert iterator_view(superseded)["superseded"] is True
        superseded.close()
        assert iterator_view(superseded)["closed"] is True
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_the_sentinel_component_notices_every_family():
    """Negative control for §12.7's "unrelated state" half: a parameter
    value and version, a gradient, a buffer, a generator, and the
    optimizer's own state must each be visible to the fingerprint."""
    sentinels = Sentinels()
    try:
        base = sentinel_view(sentinels)
        assert sentinel_view(sentinels) == base
        # A parameter value replacement moves value **and** version.
        parameter = sentinels.parameter
        replacement = NativeTensor.from_array(
            np.zeros(parameter.shape, dtype=np.float64))
        try:
            parameter.copy_value_(replacement)
        finally:
            replacement.close()
        moved = sentinel_view(sentinels)
        assert moved != base
        assert moved["parameters"]["linear.weight"][1] != \
            base["parameters"]["linear.weight"][1]
        # A generator draw moves the generator half on its own.
        before_generator = sentinel_view(sentinels)["generator"]
        sentinels.generator.reseed(4242)
        assert sentinel_view(sentinels)["generator"] != before_generator
    finally:
        sentinels.close()


def test_the_globals_component_notices_a_global_rng_move():
    base = globals_view()
    assert globals_view() == base
    state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.random()
        assert globals_view() != base
        random.setstate(state)
        assert globals_view() == base
        np.random.random()
        assert globals_view() != base
    finally:
        random.setstate(state)
        np.random.set_state(numpy_state)
    assert globals_view() == base


def test_the_globals_component_notices_a_written_file(tmp_path):
    base = globals_view(directory=tmp_path)
    assert base["files"] == []
    (tmp_path / "written.bin").write_bytes(b"x")
    assert globals_view(directory=tmp_path) != base
    assert globals_view(directory=tmp_path)["files"] == [("written.bin", 1)]


def test_the_registry_component_notices_a_moved_registry(monkeypatch):
    base = registry_view()
    assert registry_view() == base
    monkeypatch.setattr(loader_module, "_FORMAT_VERSION", 2)
    assert registry_view() != base
    monkeypatch.undo()
    assert registry_view() == base


@needs_backend
def test_the_identifier_scanner_can_actually_find_something():
    """Negative control for the AST scanner every absence check uses."""
    names = code_identifiers("src/tensorforge/experimental/"
                             "native_data_loader.py")
    for present in ("NativeBatchSampler", "_deliver_batch", "close",
                    "_rollback", "state_dict", "load_state_dict"):
        assert present in names, present
    # ...and it reads keyword-argument names, which a Name/Attribute-only
    # walk would miss entirely.
    tree = ast.parse("f(threading=1)")
    keywords = {node.arg for node in ast.walk(tree)
                if isinstance(node, ast.keyword)}
    assert keywords == {"threading"}


@needs_backend
def test_the_bit_helper_really_distinguishes_values():
    """Negative control for the bit view: it must separate values a
    tolerance would call equal, at both widths."""
    for dtype, tiny in ((np.float64, 1e-300), (np.float32, np.float32(1e-40))):
        a = np.array([1.0, 0.0], dtype=dtype)
        b = np.array([1.0, 0.0], dtype=dtype)
        assert bits(a) == bits(b)
        b[1] = tiny
        assert bits(a) != bits(b)
        # -0.0 and 0.0 compare equal numerically and differ in bits.
        c = np.array([0.0], dtype=dtype)
        d = np.array([-0.0], dtype=dtype)
        assert (c == d).all()
        assert bits(c) != bits(d)


# ===========================================================================
# 2. Dataset-construction failures — every §17.2 row
# ===========================================================================
#
# Three rows, three genuinely different injection points, and the same six
# questions at each: did the intended point really fail, is any partially
# constructed object observable, does any snapshot reference survive,
# were the caller's arrays touched, did anything global move, and does a
# retry with the injection removed produce the exact expected identity.

def constructor_frame_locals(error):
    """The ``NativeTensorDataset.__init__`` frame's locals, taken from the
    raised exception's own traceback.

    This is how "**no reference survives**" (§17.2) is proved rather than
    asserted: a traceback keeps every frame alive, so if the constructor
    left a snapshot bound to a local it would be reachable from right
    here. Finding ``None`` in both is the evidence.
    """
    traceback = error.__traceback__
    frames = []
    while traceback is not None:
        frame = traceback.tb_frame
        if (frame.f_code.co_name == "__init__"
                and frame.f_code.co_filename.endswith("native_dataset.py")):
            frames.append(frame)
        traceback = traceback.tb_next
    assert frames, "the constructor frame is not on the traceback"
    return frames[-1].f_locals


@needs_backend
@pytest.mark.parametrize("features, targets, dtype, error", [
    pytest.param(None, None, "float16", ValueError, id="dtype_value"),
    pytest.param(None, None, 64, TypeError, id="dtype_type"),
    pytest.param([[1.0, 2.0]], None, None, TypeError, id="features_not_array"),
    pytest.param(np.zeros((0, 2)), None, None, ValueError, id="no_samples"),
    pytest.param(np.zeros((4, 0)), None, None, ValueError, id="zero_width"),
    pytest.param(np.zeros(4, dtype=np.int64), None, None, TypeError,
                 id="integer_features"),
    pytest.param(np.array(1.0), None, None, ValueError, id="scalar_features"),
    pytest.param(None, [0, 1, 2, 3], None, TypeError, id="targets_not_array"),
    pytest.param(None, np.zeros((4, 1), dtype=np.int64), None,
                 ValueError, id="targets_rank"),
    pytest.param(None, np.array([True, False, True, False]), None, TypeError,
                 id="bool_targets"),
    pytest.param(None, np.zeros(4, dtype=np.float64), None, TypeError,
                 id="float_targets"),
    pytest.param(None, np.zeros(3, dtype=np.int64), None, ValueError,
                 id="length_mismatch"),
    pytest.param(None, np.array([0, -1, 2, 3], dtype=np.int64), None,
                 ValueError, id="negative_target"),
    pytest.param(None, np.array([0, 1, 2, 2 ** 63], dtype=np.uint64), None,
                 ValueError, id="target_above_int64"),
])
def test_a_construction_rejected_before_any_snapshot_allocates_nothing(
        monkeypatch, live_storages, tmp_path, features, targets, dtype,
        error):
    """§17.2 row 1: input normalization or validation fails, so nothing has
    been allocated and no object exists.

    Non-vacuity is structural rather than incidental: the fingerprint is
    the **last** step of a successful construction, so proving it was
    never reached proves the failure really was in the validation phase.
    """
    good_features, good_targets = host_arrays(4, 2)
    if features is None:
        features = good_features
    if targets is None:
        targets = good_targets
    feature_bits = (bits(features)
                    if isinstance(features, np.ndarray)
                    and features.dtype == np.float64 else None)
    target_copy = (targets.copy() if isinstance(targets, np.ndarray)
                   else list(targets))

    fingerprints = []
    original = dataset_module._fingerprint

    def counted(*args):
        fingerprints.append(args)
        return original(*args)

    monkeypatch.setattr(dataset_module, "_fingerprint", counted)
    baseline = settled(live_storages)
    before = globals_view(directory=tmp_path)
    with pytest.raises(error):
        NativeTensorDataset(features, targets, dtype=dtype)
    # The failure was before the snapshot phase: the fingerprint step,
    # which is the last thing a successful construction does, never ran.
    assert fingerprints == [], "the construction got past validation"
    monkeypatch.undo()
    assert settled(live_storages) == baseline
    assert globals_view(directory=tmp_path) == before
    # The caller's arrays are exactly as they were handed over.
    if isinstance(targets, np.ndarray):
        assert np.array_equal(targets, target_copy)
        assert targets.flags.writeable == target_copy.flags.writeable
    if feature_bits is not None:
        assert bits(features) == feature_bits
    # Retry with valid inputs succeeds and gives the contracted identity.
    dataset = NativeTensorDataset(good_features, good_targets)
    try:
        assert dataset.identity() == {
            "samples": 4, "feature_shape": [2], "feature_dtype": "float64",
            "fingerprint": dataset.fingerprint,
        }
        assert len(dataset.fingerprint) == 64
    finally:
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_failure_between_the_two_snapshots_releases_the_first(
        monkeypatch, live_storages, tmp_path, dtype):
    """§17.2 row 2: the feature snapshot exists and the target snapshot
    fails, so the feature snapshot must be released **before the exception
    leaves the constructor**."""
    features, targets = host_arrays(6, 2)
    feature_bits = bits(features)
    target_values = targets.tolist()
    baseline = settled(live_storages)
    before = globals_view(directory=tmp_path)

    calls = []
    original = np.array

    def counted(values, *args, **kwargs):
        result = original(values, *args, **kwargs)
        calls.append((result.shape, str(result.dtype)))
        if len(calls) == 2:                    # the target snapshot
            raise Boom("injected: target snapshot")
        return result

    monkeypatch.setattr(np, "array", counted)
    try:
        with pytest.raises(Boom, match="injected") as caught:
            NativeTensorDataset(features, targets, dtype=dtype)
    finally:
        monkeypatch.undo()
    # Non-vacuity, and the exact position: the first snapshot really was
    # built, at the **chosen** dtype, before the second one failed.
    assert len(calls) == 2, calls
    assert calls[0] == ((6, 2), dtype)
    assert calls[1] == ((6,), "int64")
    # No reference survives, even through the traceback that kept the
    # constructor frame alive.
    locals_after = constructor_frame_locals(caught.value)
    assert locals_after["feature_snapshot"] is None
    assert locals_after["target_snapshot"] is None

    # Construction allocates no native storage at all, so a failed one
    # cannot move the count; and nothing global moved.
    assert settled(live_storages) == baseline
    assert globals_view(directory=tmp_path) == before
    assert bits(features) == feature_bits
    assert targets.tolist() == target_values
    # Retry with the injection removed produces the exact identity.
    dataset = NativeTensorDataset(features, targets, dtype=dtype)
    try:
        assert dataset.identity()["feature_dtype"] == dtype
        assert dataset.samples == 6
        assert dataset.feature_shape == (2,)
    finally:
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_fingerprint_failure_releases_both_snapshots(
        monkeypatch, live_storages, tmp_path):
    """§17.2 row 3: both snapshots exist and the digest fails."""
    features, targets = host_arrays(6, 2)
    feature_bits = bits(features)
    baseline = settled(live_storages)
    before = globals_view(directory=tmp_path)
    seen = []

    def boom(dtype, feature_snapshot, target_snapshot):
        seen.append((dtype, feature_snapshot.shape,
                     str(feature_snapshot.dtype), target_snapshot.shape,
                     str(target_snapshot.dtype)))
        raise Boom("injected: fingerprint")

    monkeypatch.setattr(dataset_module, "_fingerprint", boom)
    with pytest.raises(Boom, match="injected") as caught:
        NativeTensorDataset(features, targets)
    # Non-vacuity: both snapshots really existed, at the right shapes and
    # dtypes, when the digest was attempted.
    assert seen == [("float64", (6, 2), "float64", (6,), "int64")]
    locals_after = constructor_frame_locals(caught.value)
    assert locals_after["feature_snapshot"] is None
    assert locals_after["target_snapshot"] is None
    monkeypatch.undo()
    assert settled(live_storages) == baseline
    assert globals_view(directory=tmp_path) == before
    assert bits(features) == feature_bits
    dataset = NativeTensorDataset(features, targets)
    try:
        assert len(dataset.fingerprint) == 64
    finally:
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_no_partially_constructed_dataset_is_reachable_after_any_row(
        monkeypatch, live_storages):
    """The common consequence of all three rows: the name a caller was
    assigning to is never bound, and no class or module registry exists
    through which a half-built object could be found."""
    baseline = settled(live_storages)
    features, targets = host_arrays(4, 2)
    dataset = "unbound"

    def boom(*args):
        raise Boom("injected: fingerprint")

    monkeypatch.setattr(dataset_module, "_fingerprint", boom)
    with pytest.raises(Boom):
        dataset = NativeTensorDataset(features, targets)
    monkeypatch.undo()
    assert dataset == "unbound"
    # No class-level registry, cache, or instance list exists to hold one.
    for forbidden in ("instances", "_instances", "registry", "_registry",
                      "_cache", "datasets", "_datasets"):
        assert not hasattr(NativeTensorDataset, forbidden), forbidden
    for forbidden in ("_datasets", "_loaders", "_samplers", "_registry",
                      "DATASETS", "LOADERS"):
        assert not hasattr(dataset_module, forbidden), forbidden
        assert not hasattr(loader_module, forbidden), forbidden
        assert not hasattr(sampler_module, forbidden), forbidden
    assert settled(live_storages) == baseline


# ===========================================================================
# 3. Malformed sampler state — every field, every wrong type (§11.2, §12.4)
# ===========================================================================
#
# The table below is the matrix, and it is deliberately exhaustive rather
# than a sample: every root key, every dataset key, every wrong type each
# one can take, every out-of-range value, every identity mismatch, and the
# joint zero-batch rule. Each row names the exception **class** and a
# distinguishing **field path** — never a whole message, which is not a
# contract.
#
# Every row is also a §12.7 assertion: the complete dataset, sampler,
# loader, and global fingerprint must be identical afterwards.


def replace(state, key, value):
    """A copy of ``state`` with one root field replaced."""
    changed = dict(state)
    changed[key] = value
    return changed


def replace_dataset(state, key, value):
    """A copy of ``state`` with one **dataset-block** field replaced."""
    identity = dict(state["dataset"])
    identity[key] = value
    return replace(state, "dataset", identity)


def drop(state, key):
    changed = dict(state)
    del changed[key]
    return changed


def add(state, key, value="extra"):
    changed = dict(state)
    changed[key] = value
    return changed


# --- containers -----------------------------------------------------------

SAMPLER_CONTAINER_FAULTS = [
    pytest.param(None, TypeError, id="none"),
    pytest.param([], TypeError, id="list"),
    pytest.param((), TypeError, id="tuple"),
    pytest.param("{}", TypeError, id="str"),
    pytest.param(7, TypeError, id="int"),
    pytest.param(b"{}", TypeError, id="bytes"),
    pytest.param(set(), TypeError, id="set"),
]


class MappingLike(dict):
    """A ``dict`` **subclass**, which the exact-type contract refuses.

    Not a convenience wrapper: it is the case §11.5 exists for. A subclass
    may override ``__getitem__``, so accepting one would mean the values
    validated are not necessarily the values committed.
    """


# --- the field matrix -----------------------------------------------------
#
# Each entry is ``(id, build, error, field)`` where ``build`` takes a valid
# state and returns the malformed one.

SAMPLER_FIELD_FAULTS = [
    # format
    ("format_type_int", lambda s: replace(s, "format", 1), TypeError,
     r"'format'"),
    ("format_type_bytes", lambda s: replace(s, "format", b"x"), TypeError,
     r"'format'"),
    ("format_type_none", lambda s: replace(s, "format", None), TypeError,
     r"'format'"),
    ("format_value_loader", lambda s: replace(s, "format",
                                              loader_module._FORMAT),
     ValueError, r"format mismatch"),
    ("format_value_empty", lambda s: replace(s, "format", ""), ValueError,
     r"format mismatch"),
    # format_version
    ("version_type_str", lambda s: replace(s, "format_version", "1"),
     TypeError, r"'format_version'"),
    ("version_type_float", lambda s: replace(s, "format_version", 1.0),
     TypeError, r"'format_version'"),
    ("version_type_bool_true", lambda s: replace(s, "format_version", True),
     TypeError, r"'format_version'"),
    ("version_type_bool_false", lambda s: replace(s, "format_version", False),
     TypeError, r"'format_version'"),
    ("version_type_numpy", lambda s: replace(s, "format_version",
                                             np.int64(1)),
     TypeError, r"'format_version'"),
    ("version_value_two", lambda s: replace(s, "format_version", 2),
     ValueError, r"unsupported sampler state format version"),
    ("version_value_zero", lambda s: replace(s, "format_version", 0),
     ValueError, r"unsupported sampler state format version"),
    ("version_value_negative", lambda s: replace(s, "format_version", -1),
     ValueError, r"unsupported sampler state format version"),
    # the dataset block's own container
    ("dataset_type_list", lambda s: replace(s, "dataset", []), TypeError,
     r"'dataset'"),
    ("dataset_type_none", lambda s: replace(s, "dataset", None), TypeError,
     r"'dataset'"),
    ("dataset_type_str", lambda s: replace(s, "dataset", "{}"), TypeError,
     r"'dataset'"),
    ("dataset_type_subclass",
     lambda s: replace(s, "dataset", MappingLike(s["dataset"])), TypeError,
     r"'dataset'"),
    # dataset.samples
    ("samples_type_str", lambda s: replace_dataset(s, "samples", "8"),
     TypeError, r"dataset\.samples"),
    ("samples_type_float", lambda s: replace_dataset(s, "samples", 8.0),
     TypeError, r"dataset\.samples"),
    ("samples_type_bool", lambda s: replace_dataset(s, "samples", True),
     TypeError, r"dataset\.samples"),
    ("samples_type_numpy",
     lambda s: replace_dataset(s, "samples", np.int64(8)), TypeError,
     r"dataset\.samples"),
    ("samples_zero", lambda s: replace_dataset(s, "samples", 0), ValueError,
     r"dataset\.samples"),
    ("samples_negative", lambda s: replace_dataset(s, "samples", -1),
     ValueError, r"dataset\.samples"),
    # dataset.feature_shape
    ("shape_type_str", lambda s: replace_dataset(s, "feature_shape", "2"),
     TypeError, r"dataset\.feature_shape"),
    ("shape_type_dict", lambda s: replace_dataset(s, "feature_shape", {}),
     TypeError, r"dataset\.feature_shape"),
    ("shape_type_int", lambda s: replace_dataset(s, "feature_shape", 2),
     TypeError, r"dataset\.feature_shape"),
    ("shape_type_none", lambda s: replace_dataset(s, "feature_shape", None),
     TypeError, r"dataset\.feature_shape"),
    ("shape_element_float",
     lambda s: replace_dataset(s, "feature_shape", [2.0]), TypeError,
     r"dataset\.feature_shape\[0\]"),
    ("shape_element_str",
     lambda s: replace_dataset(s, "feature_shape", ["2"]), TypeError,
     r"dataset\.feature_shape\[0\]"),
    ("shape_element_bool",
     lambda s: replace_dataset(s, "feature_shape", [True]), TypeError,
     r"dataset\.feature_shape\[0\]"),
    ("shape_element_zero",
     lambda s: replace_dataset(s, "feature_shape", [0]), ValueError,
     r"dataset\.feature_shape\[0\]"),
    ("shape_element_negative",
     lambda s: replace_dataset(s, "feature_shape", [-2]), ValueError,
     r"dataset\.feature_shape\[0\]"),
    ("shape_wrong_rank",
     lambda s: replace_dataset(s, "feature_shape", [2, 1]), ValueError,
     r"per-sample shape"),
    ("shape_wrong_width",
     lambda s: replace_dataset(s, "feature_shape", [3]), ValueError,
     r"per-sample shape"),
    # dataset.feature_dtype
    ("dtype_type_int", lambda s: replace_dataset(s, "feature_dtype", 64),
     TypeError, r"dataset\.feature_dtype"),
    ("dtype_type_none", lambda s: replace_dataset(s, "feature_dtype", None),
     TypeError, r"dataset\.feature_dtype"),
    ("dtype_value_float16",
     lambda s: replace_dataset(s, "feature_dtype", "float16"), ValueError,
     r"dataset\.feature_dtype"),
    ("dtype_value_bfloat16",
     lambda s: replace_dataset(s, "feature_dtype", "bfloat16"), ValueError,
     r"dataset\.feature_dtype"),
    ("dtype_value_int64",
     lambda s: replace_dataset(s, "feature_dtype", "int64"), ValueError,
     r"dataset\.feature_dtype"),
    ("dtype_value_capitalized",
     lambda s: replace_dataset(s, "feature_dtype", "Float64"), ValueError,
     r"dataset\.feature_dtype"),
    ("dtype_value_other_width",
     lambda s: replace_dataset(s, "feature_dtype", "float32"), ValueError,
     r"feature dtype"),
    # dataset.fingerprint
    ("fingerprint_type_int",
     lambda s: replace_dataset(s, "fingerprint", 0), TypeError,
     r"dataset\.fingerprint"),
    ("fingerprint_type_bytes",
     lambda s: replace_dataset(s, "fingerprint", b"a" * 64), TypeError,
     r"dataset\.fingerprint"),
    ("fingerprint_too_short",
     lambda s: replace_dataset(s, "fingerprint", "ab"), ValueError,
     r"dataset\.fingerprint"),
    ("fingerprint_too_long",
     lambda s: replace_dataset(s, "fingerprint", "a" * 65), ValueError,
     r"dataset\.fingerprint"),
    ("fingerprint_uppercase",
     lambda s: replace_dataset(s, "fingerprint",
                               s["dataset"]["fingerprint"].upper()),
     ValueError, r"dataset\.fingerprint"),
    ("fingerprint_non_hex",
     lambda s: replace_dataset(s, "fingerprint", "z" * 64), ValueError,
     r"dataset\.fingerprint"),
    ("fingerprint_mismatch",
     lambda s: replace_dataset(s, "fingerprint", "0" * 64), ValueError,
     r"content fingerprints differ"),
    # seed
    ("seed_type_str", lambda s: replace(s, "seed", "7"), TypeError,
     r"'seed'"),
    ("seed_type_float", lambda s: replace(s, "seed", 7.0), TypeError,
     r"'seed'"),
    ("seed_type_bool", lambda s: replace(s, "seed", True), TypeError,
     r"'seed'"),
    ("seed_type_numpy", lambda s: replace(s, "seed", np.uint64(7)),
     TypeError, r"'seed'"),
    ("seed_negative", lambda s: replace(s, "seed", -1), ValueError,
     r"'seed'"),
    ("seed_above_uint64", lambda s: replace(s, "seed", 2 ** 64), ValueError,
     r"'seed'"),
    # shuffle
    ("shuffle_type_int_one", lambda s: replace(s, "shuffle", 1), TypeError,
     r"'shuffle'"),
    ("shuffle_type_int_zero", lambda s: replace(s, "shuffle", 0), TypeError,
     r"'shuffle'"),
    ("shuffle_type_str", lambda s: replace(s, "shuffle", "true"), TypeError,
     r"'shuffle'"),
    ("shuffle_type_none", lambda s: replace(s, "shuffle", None), TypeError,
     r"'shuffle'"),
    ("shuffle_type_numpy", lambda s: replace(s, "shuffle", np.bool_(True)),
     TypeError, r"'shuffle'"),
    # batch_size
    ("batch_size_type_str", lambda s: replace(s, "batch_size", "3"),
     TypeError, r"'batch_size'"),
    ("batch_size_type_float", lambda s: replace(s, "batch_size", 3.0),
     TypeError, r"'batch_size'"),
    ("batch_size_type_bool", lambda s: replace(s, "batch_size", True),
     TypeError, r"'batch_size'"),
    ("batch_size_zero", lambda s: replace(s, "batch_size", 0), ValueError,
     r"'batch_size'"),
    ("batch_size_negative", lambda s: replace(s, "batch_size", -3),
     ValueError, r"'batch_size'"),
    # drop_last
    ("drop_last_type_int", lambda s: replace(s, "drop_last", 0), TypeError,
     r"'drop_last'"),
    ("drop_last_type_str", lambda s: replace(s, "drop_last", "false"),
     TypeError, r"'drop_last'"),
    ("drop_last_type_none", lambda s: replace(s, "drop_last", None),
     TypeError, r"'drop_last'"),
    # epoch
    ("epoch_type_str", lambda s: replace(s, "epoch", "0"), TypeError,
     r"'epoch'"),
    ("epoch_type_float", lambda s: replace(s, "epoch", 0.0), TypeError,
     r"'epoch'"),
    ("epoch_type_bool", lambda s: replace(s, "epoch", False), TypeError,
     r"'epoch'"),
    ("epoch_negative", lambda s: replace(s, "epoch", -1), ValueError,
     r"'epoch'"),
    ("epoch_above_uint64", lambda s: replace(s, "epoch", 2 ** 64),
     ValueError, r"'epoch'"),
    # cursor
    ("cursor_type_str", lambda s: replace(s, "cursor", "0"), TypeError,
     r"'cursor'"),
    ("cursor_type_float", lambda s: replace(s, "cursor", 0.0), TypeError,
     r"'cursor'"),
    ("cursor_type_bool", lambda s: replace(s, "cursor", True), TypeError,
     r"'cursor'"),
    ("cursor_negative", lambda s: replace(s, "cursor", -1), ValueError,
     r"'cursor'"),
    ("cursor_equals_count", lambda s: replace(s, "cursor", 3), ValueError,
     r"'cursor'"),
    ("cursor_above_count", lambda s: replace(s, "cursor", 4), ValueError,
     r"'cursor'"),
    # the §7.5 joint rule, and the dataset identity fields
    ("zero_batch_joint",
     lambda s: replace(replace(s, "batch_size", 99), "drop_last", True),
     ValueError, r"exceeds the dataset"),
    ("samples_mismatch", lambda s: replace_dataset(s, "samples", 9),
     ValueError, r"written for a dataset of"),
]

SAMPLER_KEY_FAULTS = (
    [(f"missing_{key}", (lambda key: lambda s: drop(s, key))(key),
      ValueError, r"exactly the keys")
     for key in sampler_module._STATE_FIELDS]
    + [("extra_root_key", lambda s: add(s, "workers", 4), ValueError,
        r"exactly the keys")]
    + [(f"dataset_missing_{key}",
        (lambda key: lambda s: replace(s, "dataset",
                                       drop(s["dataset"], key)))(key),
        ValueError, r"'dataset' must have exactly the keys")
       for key in sampler_module._DATASET_FIELDS]
    + [("dataset_extra_key",
        lambda s: replace(s, "dataset", add(s["dataset"], "device", "cpu")),
        ValueError, r"'dataset' must have exactly the keys")]
)

ALL_SAMPLER_FAULTS = SAMPLER_FIELD_FAULTS + SAMPLER_KEY_FAULTS

SAMPLER_FAULT_PARAMS = [
    pytest.param(build, error, field, id=name)
    for name, build, error, field in ALL_SAMPLER_FAULTS
]


def hardened_pipeline(**kwargs):
    """A loader over an 8-sample, 2-feature dataset with three batches per
    epoch — the shape every state row below is written against."""
    kwargs.setdefault("batch_size", 3)
    kwargs.setdefault("shuffle", True)
    kwargs.setdefault("seed", 7)
    return make_loader(8, **kwargs)


@needs_backend
@pytest.mark.parametrize("build, error, field", SAMPLER_FAULT_PARAMS)
def test_a_malformed_sampler_state_is_refused_and_mutates_nothing(
        live_storages, tmp_path, build, error, field):
    """Every field, every wrong type, every out-of-range value, and every
    identity mismatch — each rejected with the contracted **class** and a
    message naming the field path, and each leaving the complete
    observable world byte-identical (§12.7)."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        state = sampler.state_dict()
        malformed = build(state)
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       materialize=(0, 3), directory=tmp_path)
        with pytest.raises(error, match=field):
            sampler.load_state_dict(malformed)
        assert world(loader=loader, sampler=sampler, dataset=dataset,
                     materialize=(0, 3), directory=tmp_path) == before
        assert settled(live_storages) == baseline
        # ...and the sampler still works: the untouched valid state loads.
        assert sampler.load_state_dict(state) is None
        assert sampler.state_dict() == state
    finally:
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("container, error", SAMPLER_CONTAINER_FAULTS)
def test_a_non_dict_sampler_state_is_refused(live_storages, tmp_path,
                                             container, error):
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       materialize=(0, 3), directory=tmp_path)
        with pytest.raises(error, match=r"must be a dict"):
            sampler.load_state_dict(container)
        assert world(loader=loader, sampler=sampler, dataset=dataset,
                     materialize=(0, 3), directory=tmp_path) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_a_dict_subclass_is_not_a_sampler_state(live_storages):
    """§11.5's exact-type contract at the root: a subclass may override
    ``__getitem__``, so the values validated would not have to be the
    values committed."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        before = sampler_view(sampler)
        subclassed = MappingLike(sampler.state_dict())
        assert subclassed == sampler.state_dict()      # equal, and refused
        with pytest.raises(TypeError, match=r"must be a dict"):
            sampler.load_state_dict(subclassed)
        assert sampler_view(sampler) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_a_tuple_feature_shape_is_the_one_documented_container_latitude():
    """§12.4 step 9: ``feature_shape`` accepts a ``tuple`` as well as a
    ``list``, because a caller may legitimately have rebuilt the
    container. Nothing else in either schema has that latitude."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        state = sampler.state_dict()
        assert type(state["dataset"]["feature_shape"]) is list
        widened = replace_dataset(state, "feature_shape",
                                  tuple(state["dataset"]["feature_shape"]))
        assert sampler.load_state_dict(widened) is None
        # ...and the emitted state is still a list, never a tuple.
        assert type(sampler.state_dict()["dataset"]["feature_shape"]) is list
    finally:
        loader.close()
        dataset.close()


# --- validation precedence -------------------------------------------------
#
# Combined faults, each proving the **earlier** rule wins. The later fault
# in every pair is one that would raise a different, uniquely identifiable
# error if it had been reached, so "the earlier one won" is evidence
# rather than a coincidence of ordering.

SAMPLER_PRECEDENCE = [
    ("container_beats_keys", lambda s: [1, 2, 3], TypeError,
     r"must be a dict"),
    ("keys_beat_format",
     lambda s: drop(replace(s, "format", 1), "cursor"), ValueError,
     r"exactly the keys"),
    ("format_type_beats_format_value",
     lambda s: replace(replace(s, "format", 1), "format_version", 99),
     TypeError, r"'format'"),
    ("format_value_beats_version",
     lambda s: replace(replace(s, "format", "wrong"), "format_version", 99),
     ValueError, r"format mismatch"),
    ("version_type_beats_version_value",
     lambda s: replace(replace(s, "format_version", "2"), "seed", -1),
     TypeError, r"'format_version'"),
    ("version_beats_dataset",
     lambda s: replace(replace(s, "format_version", 2), "dataset", []),
     ValueError, r"unsupported sampler state format version"),
    ("dataset_block_beats_configuration",
     lambda s: replace(replace_dataset(s, "samples", "8"), "seed", -1),
     TypeError, r"dataset\.samples"),
    ("dataset_identity_beats_configuration",
     lambda s: replace(replace_dataset(s, "fingerprint", "0" * 64),
                       "batch_size", 0),
     ValueError, r"content fingerprints differ"),
    ("identity_structure_beats_digest",
     lambda s: replace_dataset(replace_dataset(s, "samples", 9),
                               "fingerprint", "0" * 64),
     ValueError, r"written for a dataset of"),
    ("identity_shape_beats_dtype",
     lambda s: replace_dataset(replace_dataset(s, "feature_shape", [5]),
                               "feature_dtype", "float32"),
     ValueError, r"per-sample shape"),
    ("identity_dtype_beats_digest",
     lambda s: replace_dataset(replace_dataset(s, "feature_dtype", "float32"),
                               "fingerprint", "0" * 64),
     ValueError, r"feature dtype"),
    ("all_types_beat_all_ranges",
     lambda s: replace(replace(s, "seed", 1.0), "epoch", 2 ** 70), TypeError,
     r"'seed'"),
    ("seed_type_beats_cursor_type",
     lambda s: replace(replace(s, "seed", "7"), "cursor", "9"), TypeError,
     r"'seed'"),
    ("shuffle_type_beats_batch_size_type",
     lambda s: replace(replace(s, "shuffle", 1), "batch_size", "3"),
     TypeError, r"'shuffle'"),
    ("range_order_seed_then_batch_size",
     lambda s: replace(replace(s, "seed", -1), "batch_size", 0), ValueError,
     r"'seed'"),
    ("batch_size_range_beats_epoch_range",
     lambda s: replace(replace(s, "batch_size", 0), "epoch", 2 ** 70),
     ValueError, r"'batch_size'"),
    ("joint_rule_beats_cursor",
     lambda s: replace(replace(replace(s, "batch_size", 99), "drop_last",
                               True), "cursor", 77),
     ValueError, r"exceeds the dataset"),
    ("cursor_is_last",
     lambda s: replace(s, "cursor", 99), ValueError, r"'cursor'"),
]


@needs_backend
@pytest.mark.parametrize("build, error, field", [
    pytest.param(build, error, field, id=name)
    for name, build, error, field in SAMPLER_PRECEDENCE
])
def test_the_sampler_validation_precedence_is_exact(live_storages, tmp_path,
                                                    build, error, field):
    """A state with two faults reports the **more fundamental** one, and
    the later field is therefore never reached."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       materialize=(0, 3), directory=tmp_path)
        with pytest.raises(error, match=field):
            sampler.load_state_dict(build(sampler.state_dict()))
        assert world(loader=loader, sampler=sampler, dataset=dataset,
                     materialize=(0, 3), directory=tmp_path) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_the_precedence_table_is_not_vacuous():
    """Negative control: each *later* fault in the pairs above really is a
    fault, so "the earlier one won" is evidence rather than an accident of
    which fields happen to be invalid."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        state = sampler.state_dict()
        for build, error in (
            (lambda s: replace(s, "format_version", 99), ValueError),
            (lambda s: replace(s, "dataset", []), TypeError),
            (lambda s: replace(s, "seed", -1), ValueError),
            (lambda s: replace(s, "cursor", "9"), TypeError),
            (lambda s: replace(s, "batch_size", "3"), TypeError),
            (lambda s: replace(s, "epoch", 2 ** 70), ValueError),
            (lambda s: replace(s, "cursor", 77), ValueError),
            (lambda s: replace_dataset(s, "fingerprint", "0" * 64),
             ValueError),
            (lambda s: replace_dataset(s, "feature_dtype", "float32"),
             ValueError),
        ):
            with pytest.raises(error):
                sampler.load_state_dict(build(state))
        # ...and the untouched state is genuinely valid, so the matrix is
        # measuring rejection rather than a state that never loads.
        assert sampler.load_state_dict(state) is None
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_each_dataset_identity_mismatch_is_rejected_independently(
        live_storages):
    """§12.4 step 8, one field at a time: a state that differs in exactly
    one identity field is refused for exactly that reason, and the three
    structural fields are checked before the digest."""
    loader, sampler, dataset = hardened_pipeline()
    other_samples = make_dataset(9, 2)
    other_shape = make_dataset(8, 3)
    other_dtype = make_dataset(8, 2, dtype="float32")
    other_values = make_dataset(8, 2, offset=1.0)
    try:
        baseline = settled(live_storages)
        base = sampler.state_dict()
        for foreign, pattern in (
            (other_samples, r"written for a dataset of"),
            (other_shape, r"per-sample shape"),
            (other_dtype, r"feature dtype"),
            (other_values, r"content fingerprints differ"),
        ):
            state = replace(base, "dataset", foreign.identity())
            before = sampler_view(sampler)
            with pytest.raises(ValueError, match=pattern):
                sampler.load_state_dict(state)
            assert sampler_view(sampler) == before
        # An equal-but-distinct dataset is accepted: identity is content,
        # not object identity.
        twin = make_dataset(8, 2)
        try:
            assert twin is not dataset
            assert twin.identity() == dataset.identity()
            assert sampler.load_state_dict(
                replace(base, "dataset", twin.identity())) is None
        finally:
            twin.close()
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()
        for extra in (other_samples, other_shape, other_dtype, other_values):
            extra.close()


# ===========================================================================
# 4. Malformed loader state — the wrapper and the delegated nest (§12.5)
# ===========================================================================
#
# Two halves, and keeping them apart is the point: the **wrapper's** own
# three-key schema is validated here, and the **whole** of the nested
# sampler validation is delegated. The delegation is proved by running the
# entire §3 matrix again through the wrapper and getting the identical
# class, the identical field path, and the identical precedence — which is
# what "one rule, one authority" has to mean if it means anything.


class StrLike(str):
    """A ``str`` subclass, which the exact-type contract refuses."""


def wrap(sampler_state):
    """A well-formed loader wrapper around ``sampler_state``."""
    return {
        "format": loader_module._FORMAT,
        "format_version": loader_module._FORMAT_VERSION,
        "sampler": sampler_state,
    }


LOADER_CONTAINER_FAULTS = [
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param((), id="tuple"),
    pytest.param("{}", id="str"),
    pytest.param(7, id="int"),
    pytest.param(b"{}", id="bytes"),
    pytest.param(set(), id="set"),
    pytest.param(3.5, id="float"),
]

LOADER_FIELD_FAULTS = [
    # format
    ("format_type_int", lambda s: replace(wrap(s), "format", 1), TypeError,
     r"'format'"),
    ("format_type_bytes",
     lambda s: replace(wrap(s), "format", b"tensorforge"), TypeError,
     r"'format'"),
    ("format_type_none", lambda s: replace(wrap(s), "format", None),
     TypeError, r"'format'"),
    ("format_type_str_subclass",
     lambda s: replace(wrap(s), "format", StrLike(loader_module._FORMAT)),
     TypeError, r"'format'"),
    ("format_value_sampler_tag",
     lambda s: replace(wrap(s), "format", sampler_module._FORMAT),
     ValueError, r"format mismatch"),
    ("format_value_empty", lambda s: replace(wrap(s), "format", ""),
     ValueError, r"format mismatch"),
    ("format_value_whitespace",
     lambda s: replace(wrap(s), "format", loader_module._FORMAT + " "),
     ValueError, r"format mismatch"),
    ("format_value_checkpoint_tag",
     lambda s: replace(wrap(s), "format", checkpoint_module._FORMAT),
     ValueError, r"format mismatch"),
    # format_version
    ("version_type_str", lambda s: replace(wrap(s), "format_version", "1"),
     TypeError, r"'format_version'"),
    ("version_type_float", lambda s: replace(wrap(s), "format_version", 1.0),
     TypeError, r"'format_version'"),
    ("version_type_bool_true",
     lambda s: replace(wrap(s), "format_version", True), TypeError,
     r"'format_version'"),
    ("version_type_bool_false",
     lambda s: replace(wrap(s), "format_version", False), TypeError,
     r"'format_version'"),
    ("version_type_numpy",
     lambda s: replace(wrap(s), "format_version", np.int64(1)), TypeError,
     r"'format_version'"),
    ("version_type_none",
     lambda s: replace(wrap(s), "format_version", None), TypeError,
     r"'format_version'"),
    ("version_value_two", lambda s: replace(wrap(s), "format_version", 2),
     ValueError, r"unsupported loader state format version"),
    ("version_value_zero", lambda s: replace(wrap(s), "format_version", 0),
     ValueError, r"unsupported loader state format version"),
    ("version_value_negative",
     lambda s: replace(wrap(s), "format_version", -1), ValueError,
     r"unsupported loader state format version"),
    ("version_value_three",
     lambda s: replace(wrap(s), "format_version", 3), ValueError,
     r"unsupported loader state format version"),
    # the nested container
    ("sampler_type_list", lambda s: replace(wrap(s), "sampler", []),
     TypeError, r"'sampler' must be a dict"),
    ("sampler_type_none", lambda s: replace(wrap(s), "sampler", None),
     TypeError, r"'sampler' must be a dict"),
    ("sampler_type_str",
     lambda s: replace(wrap(s), "sampler", json.dumps(s)), TypeError,
     r"'sampler' must be a dict"),
    ("sampler_type_subclass",
     lambda s: replace(wrap(s), "sampler", MappingLike(s)), TypeError,
     r"'sampler' must be a dict"),
    ("sampler_type_tuple",
     lambda s: replace(wrap(s), "sampler", tuple(s.items())), TypeError,
     r"'sampler' must be a dict"),
]

LOADER_KEY_FAULTS = (
    [(f"missing_{key}", (lambda key: lambda s: drop(wrap(s), key))(key),
      ValueError, r"exactly the keys")
     for key in loader_module._STATE_FIELDS]
    + [("extra_root_key", lambda s: add(wrap(s), "epoch", 0), ValueError,
        r"exactly the keys"),
       ("duplicated_sampler_field",
        lambda s: add(wrap(s), "cursor", 0), ValueError, r"exactly the keys"),
       ("empty_root", lambda s: {}, ValueError, r"exactly the keys")]
)

ALL_LOADER_FAULTS = LOADER_FIELD_FAULTS + LOADER_KEY_FAULTS


@needs_backend
@pytest.mark.parametrize("build, error, field", [
    pytest.param(build, error, field, id=name)
    for name, build, error, field in ALL_LOADER_FAULTS
])
def test_a_malformed_loader_wrapper_is_refused_and_mutates_nothing(
        live_storages, tmp_path, build, error, field):
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        valid = loader.state_dict()
        malformed = build(valid["sampler"])
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       materialize=(0, 3), directory=tmp_path)
        with pytest.raises(error, match=field):
            loader.load_state_dict(malformed)
        assert world(loader=loader, sampler=sampler, dataset=dataset,
                     materialize=(0, 3), directory=tmp_path) == before
        assert settled(live_storages) == baseline
        assert loader.load_state_dict(valid) is None
        assert loader.state_dict() == valid
    finally:
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("container", LOADER_CONTAINER_FAULTS)
def test_a_non_dict_loader_state_is_refused(live_storages, tmp_path,
                                            container):
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       materialize=(0, 3), directory=tmp_path)
        with pytest.raises(TypeError, match=r"loader state must be a dict"):
            loader.load_state_dict(container)
        assert world(loader=loader, sampler=sampler, dataset=dataset,
                     materialize=(0, 3), directory=tmp_path) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_a_mapping_that_is_not_exactly_a_dict_is_refused(live_storages):
    """Exact-type discipline at the wrapper root: a ``dict`` subclass, an
    ``OrderedDict``, a ``mappingproxy``, and a ``defaultdict`` are each
    **refused rather than converted** — a subclass could override
    ``__getitem__``, and a ``defaultdict`` would turn a missing key into a
    default, which §11.5 forbids in the strongest terms."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        valid = loader.state_dict()
        before = sampler_view(sampler)
        for wrong in (MappingLike(valid),
                      collections.OrderedDict(valid),
                      types.MappingProxyType(valid),
                      collections.defaultdict(int, valid),
                      collections.UserDict(valid)):
            assert dict(wrong) == valid, type(wrong).__name__
            with pytest.raises(TypeError, match=r"must be a dict"):
                loader.load_state_dict(wrong)
            assert sampler_view(sampler) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("build, error, field", SAMPLER_FAULT_PARAMS)
def test_every_nested_sampler_rule_still_applies_through_the_wrapper(
        live_storages, tmp_path, build, error, field):
    """The delegation, proved rather than asserted: the **entire** §3
    matrix, run again through ``loader.load_state_dict``, gives the
    identical exception class and the identical field path — because the
    loader restates none of it and calls the sampler's one
    validation-only seam."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        valid = loader.state_dict()
        malformed = wrap(build(valid["sampler"]))
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       materialize=(0, 3), directory=tmp_path)
        with pytest.raises(error, match=field):
            loader.load_state_dict(malformed)
        assert world(loader=loader, sampler=sampler, dataset=dataset,
                     materialize=(0, 3), directory=tmp_path) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("build, error, field", [
    pytest.param(build, error, field, id=name)
    for name, build, error, field in SAMPLER_PRECEDENCE
    if name != "container_beats_keys"
])
def test_the_nested_precedence_survives_the_wrapper(build, error, field):
    loader, sampler, dataset = hardened_pipeline()
    try:
        before = sampler_view(sampler)
        valid = loader.state_dict()
        with pytest.raises(error, match=field):
            loader.load_state_dict(wrap(build(valid["sampler"])))
        assert sampler_view(sampler) == before
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_the_wrapper_is_validated_before_the_nest(live_storages):
    """§12.5's order: a wrapper fault outranks every nested one, so a
    state that is wrong at both levels reports the outer fault and the
    nested validation is never reached."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        broken_nest = replace(loader.state_dict()["sampler"], "cursor", 99)
        before = sampler_view(sampler)
        for build, error, field in (
            (lambda n: replace(wrap(n), "format", 1), TypeError, r"'format'"),
            (lambda n: replace(wrap(n), "format", "wrong"), ValueError,
             r"format mismatch"),
            (lambda n: replace(wrap(n), "format_version", "1"), TypeError,
             r"'format_version'"),
            (lambda n: replace(wrap(n), "format_version", 2), ValueError,
             r"unsupported loader state format version"),
            (lambda n: drop(wrap(n), "format"), ValueError,
             r"exactly the keys"),
        ):
            with pytest.raises(error, match=field):
                loader.load_state_dict(build(broken_nest))
            assert sampler_view(sampler) == before
        # Non-vacuity: the nested fault really is a fault on its own.
        with pytest.raises(ValueError, match=r"'cursor'"):
            loader.load_state_dict(wrap(broken_nest))
        assert sampler_view(sampler) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_the_two_state_shapes_are_not_interchangeable(live_storages):
    """The format tags exist so a loader state and a sampler state cannot
    be confused. Both directions, and neither mutates anything."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        loader_state = loader.state_dict()
        sampler_state = sampler.state_dict()
        assert loader_state["sampler"] == sampler_state
        before = sampler_view(sampler)
        # Neither is rejected by its tag here — the **key set** differs, and
        # the key set is checked first, which is itself the contracted
        # order. The tag is what catches the case where the key sets *do*
        # coincide, which is the nested-state confusion below.
        with pytest.raises(ValueError, match=r"exactly the keys"):
            loader.load_state_dict(sampler_state)
        with pytest.raises(ValueError, match=r"exactly the keys"):
            sampler.load_state_dict(loader_state)
        # The tag doing its own job: a wrapper carrying the *sampler's*
        # tag has the right key set and is refused by the tag alone.
        with pytest.raises(ValueError, match=r"format mismatch"):
            loader.load_state_dict(
                replace(loader_state, "format", sampler_module._FORMAT))
        # ...and a sampler state wearing the loader's tag, likewise.
        with pytest.raises(ValueError, match=r"format mismatch"):
            sampler.load_state_dict(
                replace(sampler_state, "format", loader_module._FORMAT))
        # ...and a checkpoint archive's manifest is neither.
        with pytest.raises((TypeError, ValueError)):
            loader.load_state_dict({"format": checkpoint_module._FORMAT,
                                    "format_version": 3, "sampler": {}})
        assert sampler_view(sampler) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("malformed", [
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param("not a state", id="str"),
    pytest.param({"format": 1}, id="broken_dict"),
    pytest.param({}, id="empty_dict"),
])
def test_the_closed_guard_outranks_every_schema_rule(malformed,
                                                     live_storages):
    """§12.5 step 1: a closed loader refuses **before ``state`` is
    inspected at all**, which is why a malformed argument produces a
    ``RuntimeError`` here rather than the ``TypeError`` it would produce
    on an open loader."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        # Non-vacuity: on an open loader the same argument is a schema
        # error, so the RuntimeError below really is the guard.
        with pytest.raises((TypeError, ValueError)):
            loader.load_state_dict(malformed)
        valid = loader.state_dict()
        loader.close()
        before = sampler_view(sampler)
        with pytest.raises(RuntimeError, match=r"closed"):
            loader.load_state_dict(malformed)
        with pytest.raises(RuntimeError, match=r"closed"):
            loader.load_state_dict(valid)          # even its own state
        assert sampler_view(sampler) == before
        # state_dict() stays readable after close, and nothing reopened.
        assert loader.state_dict() == valid
        assert loader.closed is True
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("malformed", [
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param({}, id="empty_dict"),
])
def test_the_active_iteration_guard_outranks_every_schema_rule(
        malformed, live_storages):
    """§12.5 step 3: a live iterator refuses before the state is read, on
    both the loader and the sampler, and releasing the participation makes
    the identical call legal again."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        valid = loader.state_dict()
        iterator = iter(loader)
        before = sampler_view(sampler)
        with pytest.raises(RuntimeError, match=r"iterator"):
            loader.load_state_dict(malformed)
        with pytest.raises(RuntimeError, match=r"iterator"):
            loader.load_state_dict(valid)
        with pytest.raises(RuntimeError, match=r"iterator"):
            sampler.load_state_dict(valid["sampler"])
        assert sampler_view(sampler) == before
        iterator.close()
        # ...and now the same valid call succeeds, so the guard was the
        # only thing refusing it.
        assert loader.load_state_dict(valid) is None
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


# ===========================================================================
# 5. §12.7 in full — a rejection touches nothing, anywhere
# ===========================================================================
#
# §3 and §4 proved the pipeline's own fingerprint. This section adds the
# half §12.7 names that lives **outside** the pipeline: a registered
# parameter's value, version, and gradient; a persistent buffer; a live
# optimizer's moments; a registered generator's stream; the filesystem;
# and the process globals. One sentinel graph, every rejection family.

@needs_backend
def test_no_rejected_operation_touches_unrelated_native_state(
        live_storages, tmp_path):
    """The whole §12.7 list, around every rejection family at once.

    A rejected construction, a rejected sampler load, a rejected loader
    load, and a rejected batch request must each leave the parameters,
    versions, gradients, buffers, optimizer moments, registered generator,
    filesystem, and globals exactly as they were.
    """
    loader, sampler, dataset = hardened_pipeline()
    sentinels = Sentinels()
    try:
        baseline = settled(live_storages)
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       sentinels=sentinels, materialize=(0, 3),
                       directory=tmp_path)
        valid_sampler_state = sampler.state_dict()
        valid_loader_state = loader.state_dict()

        # 1. A rejected construction.
        with pytest.raises(TypeError):
            NativeTensorDataset([[1.0]], np.zeros(1, dtype=np.int64))
        with pytest.raises(ValueError):
            NativeTensorDataset(np.zeros((0, 2)), np.zeros(0, dtype=np.int64))
        # 2. A rejected sampler construction.
        with pytest.raises(ValueError):
            NativeBatchSampler(dataset, batch_size=0)
        with pytest.raises(ValueError):
            NativeBatchSampler(dataset, batch_size=99, drop_last=True)
        # 3. A rejected loader construction.
        with pytest.raises(TypeError):
            NativeDataLoader(dataset)
        # 4. A rejected sampler state load.
        with pytest.raises(ValueError):
            sampler.load_state_dict(replace(valid_sampler_state, "cursor", 9))
        # 5. A rejected loader state load.
        with pytest.raises(TypeError):
            loader.load_state_dict(replace(valid_loader_state, "sampler", []))
        # 6. A rejected batch request, at every §12.6 step.
        for indices, error in (((), ValueError), ((8,), ValueError),
                               ((-1,), ValueError), (("0",), TypeError),
                               ((True,), TypeError), (None, TypeError),
                               (np.zeros((2, 2), dtype=np.int64), ValueError),
                               (np.array([True, False]), TypeError)):
            with pytest.raises(error):
                dataset.feature_batch(indices)
            with pytest.raises(error):
                dataset.target_batch(indices)
        # 7. A rejected planning request.
        with pytest.raises(TypeError):
            sampler.plan(epoch="0")
        with pytest.raises(ValueError):
            sampler.epoch_permutation(epoch=-1)

        assert world(loader=loader, sampler=sampler, dataset=dataset,
                     sentinels=sentinels, materialize=(0, 3),
                     directory=tmp_path) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()
        sentinels.close()


@needs_backend
def test_a_failed_delivery_is_held_to_the_same_list(monkeypatch,
                                                    live_storages, tmp_path):
    """§12.7's closing sentence: **a failed delivery is held to exactly
    the rejected-request list**, so the only difference between the two is
    that one happened later."""
    loader, sampler, dataset = hardened_pipeline()
    sentinels = Sentinels()
    try:
        baseline = settled(live_storages)
        iterator = iter(loader)
        features, targets = next(iterator)
        features.close()
        # The fingerprint is taken *with* a live iterator, so the iterator
        # half is compared too.
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       iterator=iterator, sentinels=sentinels,
                       materialize=(0, 3), directory=tmp_path)
        pre_storage = settled(live_storages)
        serial_before = sampler._next_serial
        seen = []

        def boom(record):
            seen.append((position(sampler), sampler._transaction.status))
            raise Boom("injected: delivery seam")

        monkeypatch.setattr(loader_module, "_deliver_batch", boom)
        with pytest.raises(Boom, match="injected"):
            next(iterator)
        monkeypatch.undo()
        assert seen and seen[0][1] == "committed", "the seam never ran"
        after = world(loader=loader, sampler=sampler, dataset=dataset,
                      iterator=iterator, sentinels=sentinels,
                      materialize=(0, 3), directory=tmp_path)
        # Everything is identical except the one thing the contract says
        # must move: the serial counter, because a serial is **never
        # reused** even by a transaction that failed.
        assert without_sampler_keys(after, "next_serial") == \
            without_sampler_keys(before, "next_serial")
        assert sampler._next_serial == serial_before + 1
        assert settled(live_storages) == pre_storage
        # And the retry succeeds, so the failure really was recoverable.
        features, targets = next(iterator)
        features.close()
        iterator.close()
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()
        sentinels.close()


# ===========================================================================
# 6. Alias and independence boundaries (§5.1-§5.3, §10.1-§10.5, §11.4)
# ===========================================================================

@needs_backend
def test_caller_mutations_after_construction_change_nothing(live_storages):
    """§5.1/§5.2: the snapshots are **copies taken once**, so a caller who
    keeps mutating their arrays reaches neither the values nor the
    digest."""
    features, targets = host_arrays(6, 2)
    baseline = settled(live_storages)
    dataset = NativeTensorDataset(features, targets)
    try:
        before = dataset_view(dataset, materialize=(0, 1, 5))
        features[0, 0] = -12345.0
        features[:] = features + 1.0
        targets[:] = (targets + 1) % 3
        assert dataset_view(dataset, materialize=(0, 1, 5)) == before
        # ...including the fingerprint, which is a property of the values
        # the dataset copied and not of the arrays it was handed.
        assert dataset.fingerprint == before["fingerprint"]
    finally:
        dataset.close()
    # Closing the dataset drops its own references and touches nothing of
    # the caller's.
    assert np.isfinite(features).all()
    assert targets.tolist() == ((np.arange(6) % 3 + 1) % 3).tolist()
    assert settled(live_storages) == baseline


@needs_backend
def test_two_equal_datasets_are_independent_objects_with_equal_identity():
    """§6: identity is **content**, not object identity — and two datasets
    built from the same values share no snapshot."""
    first = make_dataset(8, 2)
    second = make_dataset(8, 2)
    try:
        assert first is not second
        assert first.identity() == second.identity()
        assert first.fingerprint == second.fingerprint
        assert first._features is not second._features
        assert first._targets is not second._targets
        assert not np.shares_memory(first._features, second._features)
        assert not np.shares_memory(first._targets, second._targets)
        first.close()
        # Closing one leaves the other completely usable.
        assert second.closed is False
        batch = second.feature_batch((0, 1))
        try:
            assert batch.shape == (2, 2)
        finally:
            batch.close()
    finally:
        first.close()
        second.close()


@needs_backend
def test_identity_and_state_containers_are_fresh_and_independent():
    """§5.6/§11.4: every container a caller is handed is new, so editing
    one reaches nothing — not the object, not the cache, not a previous
    result."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        first = dataset.identity()
        second = dataset.identity()
        assert first == second
        assert first is not second
        assert first["feature_shape"] is not second["feature_shape"]
        first["samples"] = 999
        first["feature_shape"].append(99)
        first["fingerprint"] = "0" * 64
        assert dataset.identity() == second
        assert dataset.samples == 8

        sampler_first = sampler.state_dict()
        sampler_second = sampler.state_dict()
        assert sampler_first == sampler_second
        assert sampler_first is not sampler_second
        assert sampler_first["dataset"] is not sampler_second["dataset"]
        assert (sampler_first["dataset"]["feature_shape"]
                is not sampler_second["dataset"]["feature_shape"])

        loader_first = loader.state_dict()
        loader_second = loader.state_dict()
        assert loader_first == loader_second
        assert loader_first is not loader_second
        assert loader_first["sampler"] is not loader_second["sampler"]
        assert (loader_first["sampler"]["dataset"]
                is not loader_second["sampler"]["dataset"])
        assert (loader_first["sampler"]["dataset"]["feature_shape"]
                is not loader_second["sampler"]["dataset"]["feature_shape"])

        # Editing every level of a returned state reaches nothing.
        before = world(loader=loader, sampler=sampler, dataset=dataset)
        loader_first["format"] = "hijacked"
        loader_first["sampler"]["cursor"] = 99
        loader_first["sampler"]["dataset"]["samples"] = 0
        loader_first["sampler"]["dataset"]["feature_shape"].clear()
        sampler_first["epoch"] = 77
        assert world(loader=loader, sampler=sampler, dataset=dataset) == before
        assert loader.state_dict() == loader_second
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_loader_state_carries_no_permutation_payload_or_process_local_value():
    """§11.4: the state is a position, not a dataset. It must carry no
    permutation, no values, no NumPy object, no serial, no token, and
    nothing that grows with the sample count."""
    small, small_sampler, small_dataset = hardened_pipeline()
    big, big_sampler, big_dataset = make_loader(512, batch_size=3,
                                                shuffle=True, seed=7)
    try:
        state = small.state_dict()
        big_state = big.state_dict()
        # Identical shape at 8 and at 512 samples.
        assert json.dumps(state, sort_keys=True).count(":") == \
            json.dumps(big_state, sort_keys=True).count(":")
        assert len(json.dumps(big_state)) - len(json.dumps(state)) < 40

        permutation = list(small_sampler.epoch_permutation())
        flat = json.dumps(state, sort_keys=True)
        assert json.dumps(permutation) not in flat
        # No NumPy object, no callable, no bytes anywhere in the tree.
        def walk(value):
            yield value
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from walk(item)
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)

        for value in walk(state):
            assert type(value) in (dict, list, str, int, bool), type(value)
            assert not isinstance(value, np.generic)
            assert not isinstance(value, np.ndarray)
            assert not callable(value)
        # A JSON round trip is lossless, so nothing process-local rode in.
        assert json.loads(json.dumps(state)) == state
    finally:
        small.close()
        small_dataset.close()
        big.close()
        big_dataset.close()


@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_every_batch_owns_independent_storage(live_storages, dtype):
    """§10.1/§10.5: each delivered batch is its own owning object, aliasing
    neither the dataset, nor a previous batch, nor the next one."""
    loader, sampler, dataset = make_loader(8, batch_size=3, dtype=dtype)
    try:
        baseline = settled(live_storages)
        iterator = iter(loader)
        first_features, first_targets = next(iterator)
        second_features, second_targets = next(iterator)
        # Two live batches, two live storages.
        assert settled(live_storages) == baseline + 2
        assert first_features is not second_features
        assert first_targets is not second_targets
        assert not np.shares_memory(first_targets, second_targets)
        assert not np.shares_memory(first_targets, dataset._targets)
        assert not np.shares_memory(second_targets, dataset._targets)
        # Targets are read-only and independently owning.
        for targets in (first_targets, second_targets):
            assert targets.dtype == np.int64
            assert targets.flags["C_CONTIGUOUS"]
            assert targets.flags.writeable is False
            with pytest.raises(ValueError):
                targets[0] = 0
        # Closing one reaches nothing else.
        first_values = first_features.to_numpy().copy()
        second_values = second_features.to_numpy().copy()
        first_features.close()
        assert settled(live_storages) == baseline + 1
        assert second_features.closed is False
        assert bits(second_features.to_numpy()) == bits(second_values)
        assert dataset.closed is False
        assert loader.closed is False
        assert sampler.next_batch_indices() == sampler.plan()[2]
        second_features.close()
        assert settled(live_storages) == baseline
        assert bits(first_values) != bits(second_values)
        iterator.close()
    finally:
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_two_batches_with_equal_values_are_still_distinct_objects(
        live_storages):
    """Equal contents never mean a shared object: the dataset keeps no
    cache and hands out no second reference to one tensor."""
    dataset = make_dataset(8, 2)
    try:
        baseline = settled(live_storages)
        first = dataset.feature_batch((1, 2))
        second = dataset.feature_batch((1, 2))
        try:
            assert first is not second
            assert bits(first.to_numpy()) == bits(second.to_numpy())
            assert settled(live_storages) == baseline + 2
            first.close()
            assert second.closed is False
            assert bits(second.to_numpy()) == bits(
                dataset.feature_batch((1, 2)).to_numpy())
        finally:
            first.close()
            second.close()
        gc.collect()
    finally:
        dataset.close()


@needs_backend
def test_duplicate_indices_gather_twice_without_sharing_a_row(live_storages):
    """§10.4: a gather is a gather. Duplicates are permitted, order is
    exact, and the duplicated rows are separate memory."""
    dataset = make_dataset(8, 2)
    try:
        baseline = settled(live_storages)
        features = dataset.feature_batch((3, 3, 0, 3))
        targets = dataset.target_batch((3, 3, 0, 3))
        try:
            values = features.to_numpy()
            assert features.shape == (4, 2)
            assert bits(values[0]) == bits(values[1]) == bits(values[3])
            assert bits(values[2]) != bits(values[0])
            assert targets.tolist() == [3 % 3, 3 % 3, 0, 3 % 3]
            # Writing one duplicated row does not change the other: the
            # gather copied, it did not broadcast one row four ways.
            host = values.copy()
            host[0, 0] = 42.0
            assert host[1, 0] != 42.0
        finally:
            features.close()
        assert settled(live_storages) == baseline
        # The sampler itself never produces a duplicate.
        loader, sampler, _ = make_loader(dataset=dataset, batch_size=3,
                                         shuffle=True, seed=7)
        try:
            flat = [index for group in sampler.plan() for index in group]
            assert sorted(flat) == list(range(8))
        finally:
            loader.close()
    finally:
        dataset.close()


@needs_backend
def test_a_delivered_batch_is_retained_by_neither_loader_nor_iterator(
        live_storages):
    """§9.4 Phase 6 / §15.6: after the seam returns, neither object holds a
    reference — so neither can close it, and neither tries."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        iterator = iter(loader)
        features, targets = next(iterator)
        assert iterator._features is None
        assert iterator._targets is None
        assert iterator._txn_serial == 0
        assert sampler._transaction is None
        # No attribute anywhere on either object refers to the batch.
        for holder in (iterator, loader, sampler, dataset):
            for slot in type(holder).__slots__:
                value = getattr(holder, slot, None)
                assert value is not features, (type(holder).__name__, slot)
                assert value is not targets, (type(holder).__name__, slot)
        assert settled(live_storages) == baseline + 1
        iterator.close()
        loader.close()
        dataset.close()
        # Every close path has run, and the batch is untouched.
        assert features.closed is False
        assert settled(live_storages) == baseline + 1
        values = features.to_numpy()
        assert values.shape == (3, 2)
        assert targets.shape == (3,)
        features.close()
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_a_restored_loader_adopts_values_without_adopting_objects():
    """§12.4/§12.5: configuration and position are adopted; the dataset
    identity is validated and **never** adopted; every object identity is
    preserved absolutely."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=False,
                                           seed=0, drop_last=False)
    donor_dataset = make_dataset(8, 2)
    donor_sampler = NativeBatchSampler(donor_dataset, batch_size=2,
                                       shuffle=True, seed=123,
                                       drop_last=True)
    donor = NativeDataLoader(donor_sampler)
    try:
        state = donor.state_dict()
        state["sampler"]["epoch"] = 5
        state["sampler"]["cursor"] = 2
        identity_before = dataset.identity()
        ids = (id(loader), id(sampler), id(dataset), id(loader.sampler),
               id(loader.dataset), id(sampler.dataset))
        assert loader.load_state_dict(state) is None
        # Adopted.
        assert (sampler.seed, sampler.shuffle, sampler.batch_size,
                sampler.drop_last, sampler.epoch, sampler.cursor) == \
            (123, True, 2, True, 5, 2)
        # Not adopted: identity, and every object.
        assert dataset.identity() == identity_before
        assert (id(loader), id(sampler), id(dataset), id(loader.sampler),
                id(loader.dataset), id(sampler.dataset)) == ids
        assert loader.sampler is sampler
        assert loader.dataset is dataset
        assert sampler.dataset is dataset
        assert loader._iterator is None
        # The donor is untouched by having been read.
        assert donor.state_dict()["sampler"]["epoch"] == 0
    finally:
        loader.close()
        dataset.close()
        donor.close()
        donor_dataset.close()


# ===========================================================================
# 7. The batch-transaction failure matrix — every §17.3 row
# ===========================================================================
#
# One shape per row: capture the whole world, record the candidate
# indices and their oracle values, inject at the exact intended point,
# prove non-vacuity from **inside** the injection, let the real cleanup
# run, compare the whole world again, retry, and prove the retry returns
# the same indices and the same bits in a **fresh** allocation.
#
# The four Phase-2 rows are deliberately four different injections — host
# gather, native allocation, host-to-native transfer, and target copy — so
# no failure is ever labelled as another.

class TransactionCase:
    """One loader under test, with the oracle for its next batch."""

    __slots__ = ("loader", "sampler", "dataset", "iterator", "indices",
                 "feature_bits", "target_values", "baseline", "before",
                 "serial_before")

    def __init__(self, live_storages, tmp_path, sentinels=None, dtype=None,
                 advance=0, **kwargs):
        self.loader, self.sampler, self.dataset = hardened_pipeline(
            dtype=dtype, **kwargs)
        self.baseline = settled(live_storages)
        self.iterator = iter(self.loader)
        for _ in range(advance):
            features, _ = next(self.iterator)
            features.close()
        self.indices = self.sampler.next_batch_indices()
        oracle = self.dataset.feature_batch(self.indices)
        try:
            self.feature_bits = bits(oracle.to_numpy())
        finally:
            oracle.close()
        self.target_values = self.dataset.target_batch(self.indices).tolist()
        self.serial_before = self.sampler._next_serial
        self.before = world(loader=self.loader, sampler=self.sampler,
                            dataset=self.dataset, iterator=self.iterator,
                            sentinels=sentinels, materialize=self.indices,
                            directory=tmp_path)

    def assert_nothing_consumed(self, live_storages, tmp_path,
                                sentinels=None, minted=None):
        """Every measure §17.3 names, in one place.

        ``minted`` is how many serials the row is contracted to consume: a
        Phase-1 failure before the claim consumes **none**, and every
        later row consumes exactly one, because a serial is never reused.
        """
        after = world(loader=self.loader, sampler=self.sampler,
                      dataset=self.dataset, iterator=self.iterator,
                      sentinels=sentinels, materialize=self.indices,
                      directory=tmp_path)
        assert without_sampler_keys(after, "next_serial") == \
            without_sampler_keys(self.before, "next_serial")
        if minted is not None:
            assert self.sampler._next_serial == self.serial_before + minted
        assert self.sampler._transaction is None
        assert self.iterator._features is None
        assert self.iterator._targets is None
        assert self.iterator._txn_serial == 0
        assert self.loader.closed is False
        assert settled(live_storages) == self.baseline
        assert self.sampler.next_batch_indices() == self.indices

    def retry(self, live_storages, rolled_back=None):
        """The retry half: the same indices, the same bits, and a **fresh**
        allocation, because the rolled-back one was closed."""
        features, targets = next(self.iterator)
        try:
            assert self.sampler.next_batch_indices() != self.indices \
                or self.sampler.batches_per_epoch == 1
            assert bits(features.to_numpy()) == self.feature_bits
            assert targets.tolist() == self.target_values
            if rolled_back is not None:
                assert rolled_back.closed is True
                assert features is not rolled_back
            assert features.closed is False
        finally:
            features.close()
        assert settled(live_storages) == self.baseline
        return features

    def close(self, live_storages=None):
        self.iterator.close()
        self.loader.close()
        self.dataset.close()
        if live_storages is not None:
            assert settled(live_storages) == self.baseline


# --- Phase 1: planning and claim ------------------------------------------

@needs_backend
def test_a_planning_failure_publishes_no_claim_and_mints_no_serial(
        monkeypatch, live_storages, tmp_path):
    """§17.3 row 1, before the claim: the plan is a pure function, so a
    failure computing it leaves no record and **not even a serial**."""
    case = TransactionCase(live_storages, tmp_path)
    seen = []
    try:
        def boom(self):
            seen.append((self._transaction, position(self)))
            raise Boom("injected: candidate planning")

        monkeypatch.setattr(NativeBatchSampler, "next_batch_indices", boom)
        with pytest.raises(Boom, match="injected"):
            next(case.iterator)
        monkeypatch.undo()
        # Non-vacuity: the injection ran, with no claim standing yet.
        assert len(seen) == 1 and seen[0][0] is None
        case.assert_nothing_consumed(live_storages, tmp_path, minted=0)
        case.retry(live_storages)
    finally:
        monkeypatch.undo()
        case.close(live_storages)


@needs_backend
def test_a_candidate_position_failure_publishes_no_claim(monkeypatch,
                                                         live_storages,
                                                         tmp_path):
    """§17.3 row 1, later in the same phase: the plan succeeded and the
    **candidate position** computation failed. The claim is written last,
    so there is still nothing to clean up."""
    case = TransactionCase(live_storages, tmp_path)
    seen = []
    try:
        def boom(self, epoch, cursor):
            seen.append((self._transaction, epoch, cursor))
            raise Boom("injected: candidate position")

        monkeypatch.setattr(NativeBatchSampler, "_next_position", boom)
        with pytest.raises(Boom, match="injected"):
            next(case.iterator)
        monkeypatch.undo()
        assert seen == [(None, 0, 0)], seen
        case.assert_nothing_consumed(live_storages, tmp_path, minted=0)
        case.retry(live_storages)
    finally:
        monkeypatch.undo()
        case.close(live_storages)


def test_the_claim_is_published_as_the_last_act_of_phase_one():
    """The structural reason there is no "failed after publishing the
    claim, before Phase 2" row to inject.

    ``_claim_batch`` writes ``self._transaction`` as its **last**
    statement before ``return``, and the two statements between that
    return and ``__next__``'s ``try`` are a slot assignment and a local
    binding — neither of which can raise. So the window is not merely
    untested: it does not exist. This is asserted from the AST rather than
    argued in prose, and the negative controls below prove the parser can
    fail.
    """
    source = inspect.getsource(NativeBatchSampler._claim_batch)
    body = ast.parse(source.lstrip()).body[0].body
    assert isinstance(body[-1], ast.Return)
    publication = body[-2]
    assert isinstance(publication, ast.Assign)
    target = publication.targets[0]
    assert isinstance(target, ast.Attribute) and target.attr == "_transaction"

    # ...and the caller's window between the claim and the guarded block.
    next_source = inspect.getsource(loader_module._NativeBatchIterator
                                    .__next__)
    statements = ast.parse(next_source.lstrip()).body[0].body
    guarded = [index for index, node in enumerate(statements)
               if isinstance(node, ast.Try)]
    assert len(guarded) == 1, "the transaction is not one guarded block"
    claim_index = max(
        index for index, node in enumerate(statements[:guarded[0]])
        if isinstance(node, ast.Assign)
        and "_claim_batch" in ast.dump(node)
    )
    between = statements[claim_index + 1:guarded[0]]
    for node in between:
        assert isinstance(node, ast.Assign), ast.dump(node)
        assert isinstance(node.value, ast.Constant) \
            or isinstance(node.value, ast.Name), ast.dump(node)
    # The guarded block ends in an unconditional finally.
    assert statements[guarded[0]].finalbody, "the rollback is not in a finally"


def test_the_claim_structure_parser_can_actually_fail():
    """Negative control for the two AST checks above."""
    reordered = ("def _claim_batch(self, owner):\n"
                 "    self._transaction = 1\n"
                 "    self._next_serial += 1\n"
                 "    return 1, ()\n")
    body = ast.parse(reordered).body[0].body
    assert isinstance(body[-1], ast.Return)
    assert not isinstance(body[-2], ast.Assign) \
        or getattr(body[-2].targets[0], "attr", None) != "_transaction"
    unguarded = ("def __next__(self):\n"
                 "    serial = self._sampler._claim_batch(self._token)\n"
                 "    self.materialize()\n"
                 "    try:\n"
                 "        pass\n"
                 "    finally:\n"
                 "        pass\n")
    statements = ast.parse(unguarded).body[0].body
    guarded = [index for index, node in enumerate(statements)
               if isinstance(node, ast.Try)]
    between = statements[1:guarded[0]]
    assert any(not isinstance(node, ast.Assign) for node in between)


@needs_backend
def test_a_claim_standing_with_nothing_constructed_is_cleared(
        monkeypatch, live_storages, tmp_path):
    """§17.3 row 1's cleanup half: the claim is published, the very first
    thing after it fails, and nothing native exists yet."""
    case = TransactionCase(live_storages, tmp_path)
    seen = []
    original = NativeTensorDataset.feature_batch
    try:
        def boom(self, indices):
            transaction = case.sampler._transaction
            seen.append((transaction.status, tuple(indices),
                         position(case.sampler), settled(live_storages),
                         case.iterator._features))
            raise Boom("injected: immediately after the claim")

        monkeypatch.setattr(NativeTensorDataset, "feature_batch", boom)
        with pytest.raises(Boom, match="injected"):
            next(case.iterator)
        monkeypatch.setattr(NativeTensorDataset, "feature_batch", original)
        # Non-vacuity: the claim was standing, nothing had advanced, and
        # no native storage existed.
        assert seen == [("claim", case.indices, (0, 0), case.baseline, None)]
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
        case.retry(live_storages)
    finally:
        monkeypatch.undo()
        case.close(live_storages)


# --- Phase 2: the four materialization rows, kept apart -------------------

@needs_backend
def test_a_host_gather_failure_allocates_nothing_native(
        monkeypatch, live_storages, tmp_path):
    """§17.3 row 2 (M1): the **host** gather raises, so the host→native
    boundary is never reached and no native storage is ever created.

    Distinguished from the allocation row below by exactly that: the
    counter on ``NativeTensor.from_array`` stays at zero.
    """
    case = TransactionCase(live_storages, tmp_path)
    entries = []
    original_from_array = NativeTensor.from_array
    bomb = GatherBomb()
    snapshot = case.dataset._features
    try:
        def counted(values, dtype=None, device="cpu", requires_grad=False):
            entries.append((values.shape, str(values.dtype)))
            return original_from_array(values, dtype=dtype, device=device,
                                       requires_grad=requires_grad)

        monkeypatch.setattr(NativeTensor, "from_array", staticmethod(counted))
        case.dataset._features = bomb
        with pytest.raises(Boom, match="host gather"):
            next(case.iterator)
        case.dataset._features = snapshot
        monkeypatch.undo()
        # Non-vacuity: the gather really was attempted, with the claimed
        # indices — and the native boundary was never reached.
        assert bomb.calls == [tuple(case.indices)]
        assert entries == [], entries
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
        case.retry(live_storages)
    finally:
        case.dataset._features = snapshot
        monkeypatch.undo()
        case.close(live_storages)


@needs_backend
@needs_fault_injection
def test_a_native_allocation_failure_consumes_nothing(monkeypatch,
                                                      live_storages,
                                                      tmp_path):
    """§17.3 row 3 (M2, allocation): the **backend's own** thread-local
    allocation-failure arm, not a monkeypatched stand-in.

    Distinguished from the gather row by the opposite evidence: the
    host→native boundary *was* reached, with the gathered host array, and
    it is the allocation inside it that failed.
    """
    case = TransactionCase(live_storages, tmp_path)
    entries = []
    original_from_array = NativeTensor.from_array
    try:
        def watched(values, dtype=None, device="cpu", requires_grad=False):
            entries.append((values.shape, str(values.dtype), dtype))
            return original_from_array(values, dtype=dtype, device=device,
                                       requires_grad=requires_grad)

        monkeypatch.setattr(NativeTensor, "from_array", staticmethod(watched))
        cpp._arm_alloc_failure(1)
        try:
            with pytest.raises(MemoryError):
                next(case.iterator)
        finally:
            cpp._arm_alloc_failure(0)
            cpp._require_library().tf_clear_error()
        monkeypatch.undo()
        # Non-vacuity: the arm fired at the host→native boundary, on the
        # gathered rows, at the dataset's dtype.
        assert entries == [((3, 2), "float64", "float64")], entries
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
        # ...and the identical call succeeds once the arm is disarmed, so
        # the failure was the injection and nothing else.
        case.retry(live_storages)
    finally:
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()
        monkeypatch.undo()
        case.close(live_storages)


@needs_backend
def test_a_host_to_native_transfer_failure_closes_its_own_storage(
        monkeypatch, live_storages, tmp_path):
    """§17.3 row 4 (M2, transfer): the storage was created and the copy
    into it failed, so ``from_array`` closes it before the exception
    leaves — and the peak observed **inside** the injection proves the
    storage really existed."""
    case = TransactionCase(live_storages, tmp_path)
    peaks = []
    try:
        def boom(self, values):
            peaks.append((len(live_storages), self.size, self.dtype,
                          case.iterator._features))
            raise Boom("injected: host-to-native transfer")

        monkeypatch.setattr(cpp.NativeStorage, "copy_from", boom)
        with pytest.raises(Boom, match="transfer"):
            next(case.iterator)
        monkeypatch.undo()
        # Non-vacuity, and the exact position: one more storage than the
        # baseline existed at the moment the transfer failed, and no
        # feature tensor had been published to the iterator yet.
        assert len(peaks) == 1, peaks
        assert peaks[0][0] == case.baseline + 1
        assert peaks[0][1] == 3 * 2 and peaks[0][2] == "float64"
        assert peaks[0][3] is None
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
        case.retry(live_storages)
    finally:
        monkeypatch.undo()
        case.close(live_storages)


@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_target_copy_failure_closes_the_feature_tensor(
        monkeypatch, live_storages, tmp_path, dtype):
    """§17.3 row 5 (M3) — **the one Phase-2 cleanup Phase J writes
    itself**: the feature tensor exists and is iterator-owned, and the
    iterator must close it before re-raising."""
    case = TransactionCase(live_storages, tmp_path, dtype=dtype)
    bomb = GatherBomb()
    snapshot = case.dataset._targets
    seen = []
    captured = []
    try:
        def watching_target_batch(self, indices):
            iterator = case.iterator
            seen.append((iterator._features is not None,
                         iterator._features.closed,
                         iterator._features.dtype,
                         position(case.sampler),
                         case.sampler._transaction.status,
                         settled(live_storages)))
            captured.append(iterator._features)
            self._targets = bomb              # fail inside the real gather
            try:
                return original_target_batch(self, indices)
            finally:
                self._targets = snapshot

        original_target_batch = NativeTensorDataset.target_batch
        monkeypatch.setattr(NativeTensorDataset, "target_batch",
                            watching_target_batch)
        with pytest.raises(Boom, match="host gather"):
            next(case.iterator)
        monkeypatch.undo()
        # Non-vacuity: the feature tensor was genuinely allocated, open,
        # at the right dtype, and live storage had really risen by one.
        assert len(seen) == 1, seen
        assert seen[0][:3] == (True, False, dtype)
        assert seen[0][3] == (0, 0) and seen[0][4] == "claim"
        assert seen[0][5] == case.baseline + 1
        assert bomb.calls == [tuple(case.indices)]
        # It was closed before the exception escaped, and no target object
        # survives on the iterator.
        assert captured[0].closed is True
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
        case.retry(live_storages, rolled_back=captured[0])
    finally:
        case.dataset._targets = snapshot
        monkeypatch.undo()
        case.close(live_storages)


# --- Phase 3: publication --------------------------------------------------

@needs_backend
def test_a_publication_failure_closes_the_batch_and_clears_the_claim(
        monkeypatch, live_storages, tmp_path):
    """§17.3 row 6: both halves of the batch exist and the pending record
    cannot be published."""
    case = TransactionCase(live_storages, tmp_path)
    seen = []
    captured = []
    try:
        def boom(self, serial, owner):
            seen.append((position(case.sampler), self._transaction.status,
                         case.iterator._features is not None,
                         case.iterator._targets is not None,
                         settled(live_storages)))
            captured.append(case.iterator._features)
            raise Boom("injected: publication")

        monkeypatch.setattr(NativeBatchSampler, "_publish_pending", boom)
        with pytest.raises(Boom, match="injected"):
            next(case.iterator)
        monkeypatch.undo()
        assert len(seen) == 1, seen
        assert seen[0][:4] == ((0, 0), "claim", True, True)
        assert seen[0][4] == case.baseline + 1
        assert captured[0].closed is True
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
        case.retry(live_storages, rolled_back=captured[0])
    finally:
        monkeypatch.undo()
        case.close(live_storages)


# --- Phase 4, step 1: the commit, made to fail *after* it applied ---------

@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_commit_failure_after_the_candidate_was_applied_restores_it(
        monkeypatch, live_storages, tmp_path, dtype):
    """§17.3's Phase-4 row, at the **commit** rather than the seam, and
    injected so the candidate assignment genuinely happens first.

    The distinction matters and is the reason this test exists beside
    J3's: raising *instead of* the assignment proves the rollback copes
    with a commit that never ran, while raising *after* it proves the
    rollback copes with one that did. Only the second exercises the
    restore path with a position that really moved.
    """
    case = TransactionCase(live_storages, tmp_path, dtype=dtype, advance=1)
    original = NativeBatchSampler._assign_state
    calls = []
    try:
        def wrapper(self, *values):
            original(self, *values)            # apply it for real, first
            calls.append((values[-2:], position(self),
                          None if self._transaction is None
                          else self._transaction.status))
            if len(calls) == 1:
                raise Boom("injected: after the candidate was applied")

        monkeypatch.setattr(NativeBatchSampler, "_assign_state", wrapper)
        with pytest.raises(Boom, match="injected"):
            next(case.iterator)
        monkeypatch.undo()
        # Non-vacuity, and the exact ordering: the candidate really was
        # committed and visible, and then the restore put the exact
        # pre-delivery position back — **before** any resource cleanup.
        assert len(calls) == 2, calls
        assert calls[0][0] == (0, 2) and calls[0][1] == (0, 2)
        assert calls[0][2] == "committed"
        assert calls[1][0] == (0, 1) and calls[1][1] == (0, 1)
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
        case.retry(live_storages)
    finally:
        monkeypatch.undo()
        case.close(live_storages)


@needs_backend
def test_the_rollback_restores_the_position_before_it_releases_resources(
        monkeypatch, live_storages, tmp_path):
    """§9.4 Phase 5's contracted **order**: restore first, because that
    step cannot fail, then clear the record, then close the tensor."""
    case = TransactionCase(live_storages, tmp_path)
    order = []
    original_assign = NativeBatchSampler._assign_state
    original_close = NativeTensor.close
    try:
        def watched_assign(self, *values):
            original_assign(self, *values)
            order.append(("position", values[-2:]))

        def watched_close(self):
            order.append(("close", self.closed))
            return original_close(self)

        def boom(record):
            order.append(("seam", position(case.sampler)))
            raise Boom("injected: delivery seam")

        monkeypatch.setattr(NativeBatchSampler, "_assign_state",
                            watched_assign)
        monkeypatch.setattr(NativeTensor, "close", watched_close)
        monkeypatch.setattr(loader_module, "_deliver_batch", boom)
        with pytest.raises(Boom, match="injected"):
            next(case.iterator)
        monkeypatch.undo()
        # commit → seam → restore → close, in that order and no other.
        assert order[0] == ("position", (0, 1))
        assert order[1] == ("seam", (0, 1))
        assert order[2] == ("position", (0, 0))
        assert order[3][0] == "close" and order[3][1] is False
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
    finally:
        monkeypatch.undo()
        case.close(live_storages)


# --- Phase 4, step 2: the delivery seam ------------------------------------

@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_delivery_seam_failure_consumes_nothing_at_all(
        monkeypatch, live_storages, tmp_path, dtype):
    """The load-bearing row, under the complete world fingerprint: the
    candidate is committed, the seam raises, and **nothing whatsoever**
    is different afterwards except a serial that may never be reused."""
    sentinels = Sentinels()
    case = TransactionCase(live_storages, tmp_path, sentinels=sentinels,
                           dtype=dtype, advance=1)
    seen = []
    captured = []
    try:
        def boom(record):
            transaction = case.sampler._transaction
            seen.append((position(case.sampler), transaction.status,
                         transaction.before[-2:], transaction.after[-2:],
                         record._features.dtype, settled(live_storages),
                         refuses_state(case.loader),
                         refuses_state(case.sampler)))
            captured.append(record._features)
            raise Boom("injected: delivery seam")

        monkeypatch.setattr(loader_module, "_deliver_batch", boom)
        with pytest.raises(Boom, match="injected"):
            next(case.iterator)
        monkeypatch.undo()
        # Non-vacuity, and every fact the contract names about the moment
        # of failure: the candidate is applied, the record is committed,
        # the batch exists, and **no snapshot can observe the position**.
        assert len(seen) == 1, seen
        assert seen[0][0] == (0, 2)            # candidate visible inside
        assert seen[0][1] == "committed"
        assert seen[0][2] == (0, 1) and seen[0][3] == (0, 2)
        assert seen[0][4] == dtype
        assert seen[0][5] == case.baseline + 1
        assert seen[0][6] is True and seen[0][7] is True
        assert captured[0].closed is True
        case.assert_nothing_consumed(live_storages, tmp_path,
                                     sentinels=sentinels, minted=1)
        case.retry(live_storages, rolled_back=captured[0])
    finally:
        monkeypatch.undo()
        case.close(live_storages)
        sentinels.close()


def refuses_state(obj):
    """Whether ``obj.state_dict()`` refuses right now (§9.5)."""
    try:
        obj.state_dict()
    except RuntimeError:
        return True
    return False


# --- the BaseException path ------------------------------------------------

@needs_backend
@pytest.mark.parametrize("phase", ["commit", "seam"])
def test_a_base_exception_still_triggers_the_full_rollback(
        monkeypatch, live_storages, tmp_path, phase):
    """The ``finally`` is unconditional, so a ``BaseException`` that is
    **not** an ``Exception`` — the shape an asynchronous interruption
    takes — rolls the transaction back exactly as an ordinary one does.

    Nothing in production catches it: it propagates to the caller
    unchanged, which is why ``pytest.raises(Abort)`` is the assertion.
    """
    case = TransactionCase(live_storages, tmp_path, advance=1)
    seen = []
    captured = []
    original_assign = NativeBatchSampler._assign_state
    try:
        if phase == "commit":
            def wrapper(self, *values):
                original_assign(self, *values)
                seen.append((values[-2:], position(self)))
                if len(seen) == 1:
                    captured.append(case.iterator._features)
                    raise Abort("injected: BaseException at the commit")

            monkeypatch.setattr(NativeBatchSampler, "_assign_state", wrapper)
        else:
            def boom(record):
                seen.append((position(case.sampler),
                             case.sampler._transaction.status))
                captured.append(record._features)
                raise Abort("injected: BaseException at the seam")

            monkeypatch.setattr(loader_module, "_deliver_batch", boom)

        with pytest.raises(Abort, match="injected"):
            next(case.iterator)
        monkeypatch.undo()
        assert seen, "the BaseException injection never ran"
        assert not isinstance(Abort("x"), Exception)
        assert captured[0] is not None and captured[0].closed is True
        case.assert_nothing_consumed(live_storages, tmp_path, minted=1)
        case.retry(live_storages, rolled_back=captured[0])
    finally:
        monkeypatch.undo()
        case.close(live_storages)


# --- the epoch-overflow boundary ------------------------------------------

@needs_backend
def test_the_epoch_overflow_boundary_refuses_and_stays_deterministic(
        live_storages, tmp_path):
    """§7.4: advancing past ``2**64 - 1`` raises and moves nothing — no
    claim, no serial, no allocation, and certainly no wrapped epoch. The
    position is reached through a **validated state load**, never by
    assigning a private field."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        state = loader.state_dict()
        state["sampler"]["epoch"] = 2 ** 64 - 1
        state["sampler"]["cursor"] = sampler.batches_per_epoch - 1
        assert loader.load_state_dict(state) is None
        iterator = iter(loader)
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       iterator=iterator, materialize=(0, 1),
                       directory=tmp_path)
        serial_before = sampler._next_serial
        # The refusal, repeatedly: it is not an exhaustion and not a close.
        for _ in range(3):
            with pytest.raises(RuntimeError, match=r"epoch"):
                next(iterator)
            assert world(loader=loader, sampler=sampler, dataset=dataset,
                         iterator=iterator, materialize=(0, 1),
                         directory=tmp_path) == before
        assert sampler.epoch == 2 ** 64 - 1
        assert sampler._next_serial == serial_before      # no serial minted
        assert sampler._transaction is None
        assert settled(live_storages) == baseline
        # Stepping back one batch makes the identical call legal again, so
        # the refusal was the boundary and nothing else.
        state["sampler"]["cursor"] = 0
        iterator.close()
        assert loader.load_state_dict(state) is None
        features, _ = next(iter(loader))
        features.close()
        assert sampler.cursor == 1
        assert sampler.epoch == 2 ** 64 - 1
    finally:
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


# --- iterator creation -----------------------------------------------------

@needs_backend
def test_a_failed_iterator_creation_releases_its_participation(
        monkeypatch, live_storages, tmp_path):
    """§17.3's last row: participation is taken before the countdown is
    captured, so a failure between the two must give it back."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       materialize=(0, 1), directory=tmp_path)
        token_before = sampler._next_iteration_token
        seen = []

        def boom(self):
            seen.append(frozenset(self._active_iterations))
            raise Boom("injected: iterator creation")

        monkeypatch.setattr(NativeBatchSampler, "remaining", property(boom))
        with pytest.raises(Boom, match="injected"):
            iter(loader)
        monkeypatch.undo()
        # Non-vacuity: participation really had been taken when it failed.
        assert len(seen) == 1 and len(seen[0]) == 1, seen
        # ...and it was released, leaving no partial iterator anywhere.
        assert sampler._active_iterations == set()
        assert loader._iterator is None
        assert sampler._transaction is None
        assert loader.closed is False
        # The token counter advanced, and that is the contract: a token is
        # **never reused**, so a failed iterator's token can never be
        # handed out again. Everything else is identical.
        assert sampler._next_iteration_token == token_before + 1
        after = world(loader=loader, sampler=sampler, dataset=dataset,
                      materialize=(0, 1), directory=tmp_path)
        assert without_sampler_keys(after, "next_token") == \
            without_sampler_keys(before, "next_token")
        assert settled(live_storages) == baseline
        # A state load is legal again, which it would not be if the
        # participation had leaked.
        assert loader.load_state_dict(loader.state_dict()) is None
        # ...and a retry creates a perfectly ordinary iterator.
        iterator = iter(loader)
        features, _ = next(iterator)
        features.close()
        iterator.close()
        assert settled(live_storages) == baseline
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


# ===========================================================================
# 8. The reentrancy refusal matrix (§9.5, §16.1, §16.2)
# ===========================================================================
#
# The claim guards **reentrancy**, not concurrency, so every probe below
# runs on the calling thread, from inside a real transaction, at a named
# phase. Nothing here starts a thread, and nothing here claims a race is
# safe — §16's concurrency half is asserted as *documented and
# unprotected* in section 13.

TRANSACTION_PHASES = ("claim", "pending", "committed")


def reentrant_probe(monkeypatch, loader, probe, phase):
    """Run ``probe(status, position)`` from inside the transaction, at the
    requested phase, and return what it collected.

    ``claim`` — published, nothing constructed.
    ``pending`` — both halves built, **position not yet applied**.
    ``committed`` — candidate applied, batch not yet delivered.
    """
    sampler = loader.sampler
    collected = []

    def run():
        transaction = sampler._transaction
        collected.append(probe(transaction.status, position(sampler)))

    if phase == "claim":
        original = NativeTensorDataset.feature_batch

        def at_claim(self, indices):
            run()
            return original(self, indices)

        monkeypatch.setattr(NativeTensorDataset, "feature_batch", at_claim)
    elif phase == "pending":
        original = NativeBatchSampler._commit_pending

        def at_pending(self, serial, owner):
            run()
            return original(self, serial, owner)

        monkeypatch.setattr(NativeBatchSampler, "_commit_pending", at_pending)
    else:
        def at_committed(record):
            run()
            return record._features, record._targets

        monkeypatch.setattr(loader_module, "_deliver_batch", at_committed)

    features, targets = next(iter(loader))
    features.close()
    monkeypatch.undo()
    assert len(collected) == 1, f"the {phase} probe never ran"
    return collected[0]


@needs_backend
@pytest.mark.parametrize("phase", TRANSACTION_PHASES)
def test_the_three_transaction_phases_are_each_really_reached(monkeypatch,
                                                              phase):
    """Non-vacuity for the whole matrix below: each named phase exists,
    is distinguishable, and carries the position the contract says."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        seen = reentrant_probe(monkeypatch, loader,
                               lambda status, where: (status, where), phase)
        expected = {"claim": ("claim", (0, 0)),
                    "pending": ("pending", (0, 0)),
                    "committed": ("committed", (0, 1))}[phase]
        assert seen == expected
        assert position(sampler) == (0, 1)
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("phase", TRANSACTION_PHASES)
def test_every_refused_reentrant_operation_raises_and_mutates_nothing(
        monkeypatch, live_storages, phase):
    """§9.5's refusal rows, all of them, at all three phases.

    Every one raises ``RuntimeError``, and the malformed arguments prove
    the **guard** answered rather than the schema: on an idle loader those
    same arguments are a ``TypeError``.
    """
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        valid_loader_state = loader.state_dict()
        valid_sampler_state = sampler.state_dict()

        def probe(status, where):
            observed = {"status": status, "position": where,
                        "serial": sampler._next_serial,
                        "record": sampler._transaction}
            iterator = loader._iterator
            # A second __next__ on the same iterator.
            with pytest.raises(RuntimeError):
                next(iterator)
            # A new iterator: supersession is refused rather than performed.
            with pytest.raises(RuntimeError):
                iter(loader)
            # Both state reads.
            with pytest.raises(RuntimeError):
                loader.state_dict()
            with pytest.raises(RuntimeError):
                sampler.state_dict()
            # Both state loads, with valid **and** malformed arguments —
            # the malformed ones are never inspected.
            for state in (valid_loader_state, None, [], {"format": 1}):
                with pytest.raises(RuntimeError):
                    loader.load_state_dict(state)
            for state in (valid_sampler_state, None, "nope", 7):
                with pytest.raises(RuntimeError):
                    sampler.load_state_dict(state)
            observed["after"] = {"serial": sampler._next_serial,
                                 "record": sampler._transaction,
                                 "position": position(sampler),
                                 "iterator": loader._iterator is iterator,
                                 "active": len(sampler._active_iterations)}
            return observed

        seen = reentrant_probe(monkeypatch, loader, probe, phase)
        # The original transaction survived every refusal, untouched: same
        # record object, same serial counter, same position, same iterator
        # in the slot, and exactly one participation.
        assert seen["after"]["record"] is seen["record"]
        assert seen["after"]["serial"] == seen["serial"]
        assert seen["after"]["position"] == seen["position"]
        assert seen["after"]["iterator"] is True
        assert seen["after"]["active"] == 1
        # ...and the batch was delivered exactly once afterwards.
        assert position(sampler) == (0, 1)
        assert sampler._transaction is None
        assert settled(live_storages) == baseline
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("phase", TRANSACTION_PHASES)
def test_the_permitted_reentrant_operations_stay_permitted(monkeypatch,
                                                           phase):
    """§9.5's other half: the plain integer reads and the three pure
    planners are **permitted** at every phase, and answer exactly what
    they answer outside one."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        plan = sampler.plan(0)
        permutation = sampler.epoch_permutation(0)

        def probe(status, where):
            return (sampler.epoch, sampler.cursor, sampler.remaining,
                    sampler.batches_per_epoch, sampler.batch_size,
                    sampler.seed, sampler.shuffle, sampler.drop_last,
                    sampler.epoch_permutation(0), sampler.plan(0),
                    sampler.next_batch_indices(), repr(sampler),
                    repr(loader), loader.closed, loader.sampler is sampler,
                    loader.dataset is dataset, len(dataset),
                    dataset.identity(), repr(dataset))

        seen = reentrant_probe(monkeypatch, loader, probe, phase)
        assert seen[8] == permutation
        assert seen[9] == plan
        assert seen[13] is False and seen[14] is True and seen[15] is True
        assert seen[16] == 8
        assert seen[17] == dataset.identity()
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("phase", TRANSACTION_PHASES)
def test_a_reentrant_iterator_close_resolves_exactly_its_transaction(
        monkeypatch, live_storages, phase):
    """§9.5's one row that is **performed** rather than refused: close is
    never refused, because it is the recovery path.

    Whichever of the close and the transaction's own ``finally`` reaches
    the rollback first performs it; the second matches nothing. The outer
    ``__next__`` therefore never returns a closed or unowned batch.
    """
    loader, sampler, dataset = hardened_pipeline()
    delivered = []
    try:
        baseline = settled(live_storages)
        # One genuinely delivered batch first, so "an earlier delivery is
        # untouched" is part of the claim rather than an empty statement.
        iterator = iter(loader)
        first_features, first_targets = next(iterator)
        delivered.append(first_features)
        first_bits = bits(first_features.to_numpy())
        pre_state = loader.state_dict()
        pre_position = position(sampler)
        pre_indices = sampler.next_batch_indices()
        pre_dataset = dataset_view(dataset, materialize=(0, 1))
        seen = []

        def close_from_inside(*args):
            live = loader._iterator
            seen.append((sampler._transaction.status, position(sampler)))
            live.close()
            seen.append((sampler._transaction, position(sampler),
                         live._features, live._closed))

        if phase == "claim":
            original = NativeTensorDataset.feature_batch

            def hook(self, indices):
                close_from_inside()
                return original(self, indices)

            monkeypatch.setattr(NativeTensorDataset, "feature_batch", hook)
        elif phase == "pending":
            original = NativeBatchSampler._commit_pending

            def hook(self, serial, owner):
                close_from_inside()
                return original(self, serial, owner)

            monkeypatch.setattr(NativeBatchSampler, "_commit_pending", hook)
        else:
            def hook(record):
                close_from_inside()
                return record._features, record._targets

            monkeypatch.setattr(loader_module, "_deliver_batch", hook)

        with pytest.raises(RuntimeError):
            next(iterator)
        monkeypatch.undo()
        # Non-vacuity: the reentrant close really ran, at the named phase,
        # and it resolved the record and released the resource.
        assert len(seen) == 2, seen
        assert seen[0][0] == phase
        assert seen[1][0] is None
        assert seen[1][2] is None and seen[1][3] is True
        # **No second batch was consumed**: the position, the state, the
        # next candidate, and the dataset are exactly what they were after
        # the first delivery.
        assert position(sampler) == pre_position
        assert loader.state_dict() == pre_state
        assert sampler.next_batch_indices() == pre_indices
        assert dataset_view(dataset, materialize=(0, 1)) == pre_dataset
        # What close *is* contracted to change, and only that: the
        # iterator detached and gave back its participation.
        assert loader._iterator is None
        assert sampler._active_iterations == set()
        assert loader.closed is False
        # A second rollback for the same transaction matches nothing.
        assert iterator._txn_serial == 0
        assert sampler._transaction is None
        # The earlier delivered batch is untouched by all of it.
        assert first_features.closed is False
        assert bits(first_features.to_numpy()) == first_bits
        assert first_targets.tolist() == first_targets.tolist()
        assert settled(live_storages) == baseline + 1
    finally:
        monkeypatch.undo()
        for tensor in delivered:
            tensor.close()
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
@pytest.mark.parametrize("phase", TRANSACTION_PHASES)
def test_a_reentrant_loader_close_resolves_through_its_iterator(
        monkeypatch, live_storages, phase):
    """The loader delegates to its iterator's close, so the same rollback
    runs — and the loader ends up closed, with no participation left."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        before = without_sampler_keys(
            world(loader=loader, sampler=sampler, dataset=dataset),
            "next_serial", "next_token")
        seen = []

        def close_from_inside(*args):
            seen.append((sampler._transaction.status, position(sampler)))
            loader.close()
            seen.append((sampler._transaction, position(sampler),
                         loader.closed, frozenset(sampler._active_iterations)))

        if phase == "claim":
            original = NativeTensorDataset.feature_batch

            def hook(self, indices):
                close_from_inside()
                return original(self, indices)

            monkeypatch.setattr(NativeTensorDataset, "feature_batch", hook)
        elif phase == "pending":
            original = NativeBatchSampler._commit_pending

            def hook(self, serial, owner):
                close_from_inside()
                return original(self, serial, owner)

            monkeypatch.setattr(NativeBatchSampler, "_commit_pending", hook)
        else:
            def hook(record):
                close_from_inside()
                return record._features, record._targets

            monkeypatch.setattr(loader_module, "_deliver_batch", hook)

        with pytest.raises(RuntimeError):
            next(iter(loader))
        monkeypatch.undo()
        assert len(seen) == 2 and seen[0][0] == phase
        assert seen[1][0] is None and seen[1][2] is True
        assert seen[1][3] == frozenset()
        assert seen[1][1] == (0, 0)
        assert sampler._transaction is None
        assert sampler._active_iterations == set()
        assert settled(live_storages) == baseline
        # The position and the whole sampler are exactly pre-call; only
        # the loader's own ``closed`` differs, which is what close means.
        after = without_sampler_keys(
            world(sampler=sampler, dataset=dataset),
            "next_serial", "next_token")
        assert after["sampler"] == before["sampler"]
        assert after["dataset"] == before["dataset"]
        with pytest.raises(RuntimeError):
            iter(loader)
        # ...and state_dict() still works on the closed loader.
        assert loader.state_dict()["sampler"]["cursor"] == 0
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
def test_a_dataset_close_during_a_transaction_is_permitted(monkeypatch,
                                                           live_storages):
    """§9.5's remaining row: closing the **dataset** touches no loader
    state, so the in-flight transaction completes normally and only the
    *next* attempt refuses."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        seen = []

        def hook(record):
            dataset.close()
            seen.append((dataset.closed, sampler._transaction.status,
                         position(sampler)))
            return record._features, record._targets

        monkeypatch.setattr(loader_module, "_deliver_batch", hook)
        iterator = iter(loader)
        features, targets = next(iterator)
        monkeypatch.undo()
        assert seen == [(True, "committed", (0, 1))]
        # The already-built batch is complete and valid.
        assert features.shape == (3, 2)
        assert targets.shape == (3,)
        assert position(sampler) == (0, 1)
        features.close()
        # The next attempt refuses in Phase 2, consuming nothing.
        before = sampler_view(sampler)
        with pytest.raises(RuntimeError, match=r"closed"):
            next(iterator)
        assert without_sampler_keys({"sampler": sampler_view(sampler)},
                                    "next_serial") == \
            without_sampler_keys({"sampler": before}, "next_serial")
        assert sampler._transaction is None
        assert settled(live_storages) == baseline
        iterator.close()
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


# ===========================================================================
# 9. Serial, token, and exact-match cleanup (§9.4 Phase 5, §15.3)
# ===========================================================================

@needs_backend
def test_serials_increase_and_are_never_reused(monkeypatch, live_storages):
    """Across successful deliveries **and** a failed one: strictly
    increasing, all distinct, and a failed delivery's serial is never
    handed out again."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        serials = []
        original_seam = loader_module._deliver_batch

        def watched(record):
            serials.append(record._txn_serial)
            return original_seam(record)

        monkeypatch.setattr(loader_module, "_deliver_batch", watched)
        for features, _ in loader:
            features.close()
        # One failed delivery in the middle of the next epoch.
        failed = []

        def boom(record):
            failed.append(record._txn_serial)
            raise Boom("injected: delivery seam")

        iterator = iter(loader)
        monkeypatch.setattr(loader_module, "_deliver_batch", boom)
        with pytest.raises(Boom):
            next(iterator)
        monkeypatch.setattr(loader_module, "_deliver_batch", watched)
        for features, _ in iterator:
            features.close()
        monkeypatch.undo()

        assert len(serials) == 6 and len(failed) == 1
        every = serials[:3] + failed + serials[3:]
        assert every == sorted(every), every
        assert len(set(every)) == len(every), every
        # The failed serial sits between the two successful runs and is
        # never handed out again.
        assert serials[2] < failed[0] < serials[3]
        assert sampler._next_serial == every[-1] + 1
        assert settled(live_storages) == baseline
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
def test_only_an_exact_serial_and_token_can_resolve_a_transaction(
        monkeypatch, live_storages):
    """Every cleanup route is exact-match: a stale serial, a foreign
    token, a serial that was never minted, and a wrong status each match
    **nothing**, and the live record is undisturbed."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        iterator = iter(loader)
        features, _ = next(iterator)
        features.close()
        stale_serial = 1
        seen = []

        def probe(record):
            live = sampler._transaction
            seen.append((live.serial, live.owner, live.status))
            # Wrong serial, wrong owner, never-minted serial, and a
            # completed-status mismatch: none of them may match.
            assert sampler._rollback_pending(stale_serial, live.owner) is False
            assert sampler._rollback_pending(live.serial,
                                             live.owner + 1000) is False
            assert sampler._rollback_pending(10 ** 9, live.owner) is False
            assert sampler._rollback_pending(0, live.owner) is False
            assert sampler._complete_pending(stale_serial, live.owner) is False
            assert sampler._complete_pending(live.serial,
                                             live.owner + 1000) is False
            # The record is exactly as it was.
            assert sampler._transaction is live
            assert live.status == "committed"
            assert position(sampler) == (0, 2)
            return record._features, record._targets

        monkeypatch.setattr(loader_module, "_deliver_batch", probe)
        features, _ = next(iterator)
        monkeypatch.undo()
        assert len(seen) == 1
        assert seen[0][0] != stale_serial
        assert position(sampler) == (0, 2)
        features.close()
        iterator.close()
        assert settled(live_storages) == baseline
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
def test_the_rollback_is_idempotent_and_completion_is_terminal(
        monkeypatch, live_storages):
    """Four orderings, all of which must be safe: rollback twice;
    complete then roll back; roll back then complete; and the
    transaction's own ``finally`` arriving after either."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        captured = {}

        def boom(record):
            captured["serial"] = record._txn_serial
            captured["token"] = record._token
            captured["features"] = record._features
            # 1. Roll back from inside; 2. and 3. do nothing at all.
            assert record._rollback(record._txn_serial) is True
            assert position(sampler) == (0, 0)
            assert record._rollback(captured["serial"]) is False
            assert record._rollback(captured["serial"]) is False
            # 4. A completion after a rollback cannot resurrect it, and
            #    cannot advance the position.
            assert sampler._complete_pending(captured["serial"],
                                             captured["token"]) is False
            assert position(sampler) == (0, 0)
            raise Boom("injected: after a manual rollback")

        monkeypatch.setattr(loader_module, "_deliver_batch", boom)
        iterator = iter(loader)
        with pytest.raises(Boom, match="injected"):
            next(iterator)
        monkeypatch.undo()
        # The transaction's own unconditional finally ran afterwards and
        # matched nothing; the tensor was closed exactly once.
        assert captured["features"].closed is True
        assert position(sampler) == (0, 0)
        assert sampler._transaction is None
        assert settled(live_storages) == baseline
        assert sampler._rollback_pending(captured["serial"],
                                         captured["token"]) is False

        # The mirror case: a **completed** transaction can never be rolled
        # back, so no cleanup can reach a delivered batch.
        features, targets = next(iterator)
        delivered_bits = bits(features.to_numpy())
        serial = None

        def watcher(record):
            nonlocal serial
            serial = record._txn_serial
            return record._features, record._targets

        monkeypatch.setattr(loader_module, "_deliver_batch", watcher)
        second_features, _ = next(iterator)
        monkeypatch.undo()
        assert sampler._rollback_pending(serial, iterator._token) is False
        assert sampler._complete_pending(serial, iterator._token) is False
        assert position(sampler) == (0, 2)
        assert features.closed is False
        assert bits(features.to_numpy()) == delivered_bits
        features.close()
        second_features.close()
        iterator.close()
        assert settled(live_storages) == baseline
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
def test_a_rollback_for_one_transaction_cannot_alter_a_later_one(
        monkeypatch, live_storages):
    """A cleanup that arrives late must not reach the transaction that
    replaced it — the exact-match rule is what survives even a raced
    handoff (§16.2)."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        iterator = iter(loader)
        stale = {}

        def capture(record):
            stale["serial"] = record._txn_serial
            stale["token"] = record._token
            return record._features, record._targets

        monkeypatch.setattr(loader_module, "_deliver_batch", capture)
        features, _ = next(iterator)
        features.close()
        monkeypatch.undo()

        checks = []

        def probe(record):
            live = sampler._transaction
            before = position(sampler)
            checks.append((live.serial != stale["serial"],
                           sampler._rollback_pending(stale["serial"],
                                                     stale["token"]),
                           position(sampler) == before,
                           sampler._transaction is live,
                           live.status))
            return record._features, record._targets

        monkeypatch.setattr(loader_module, "_deliver_batch", probe)
        features, _ = next(iterator)
        monkeypatch.undo()
        assert checks == [(True, False, True, True, "committed")]
        assert position(sampler) == (0, 2)
        features.close()
        iterator.close()
        assert settled(live_storages) == baseline
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
def test_iteration_tokens_are_never_reused_either(live_storages):
    """A released token can never be handed out again, so a stale
    iterator can never release another iterator's participation."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        tokens = []
        for _ in range(4):
            iterator = iter(loader)
            tokens.append(iterator._token)
            iterator.close()
        assert tokens == sorted(tokens)
        assert len(set(tokens)) == len(tokens)
        assert sampler._active_iterations == set()
        # A stale iterator's release is a no-op against a live one.
        stale = iter(loader)
        stale_token = stale._token
        stale.close()
        live = iter(loader)
        assert live._token != stale_token
        sampler._end_iteration(stale_token)
        assert live._token in sampler._active_iterations
        live.close()
        assert sampler._active_iterations == set()
    finally:
        loader.close()
        dataset.close()


# ===========================================================================
# 10. Abandonment — every documented position (§9.6, §15.5, §17.3)
# ===========================================================================
#
# Two framings, both from the design, and neither collapsed into the
# other: the **four cleanup positions** a transaction can be abandoned at
# (nothing claimed / claim with no tensor / tensor before publication /
# record before delivery), and the **§9.6 rows** about where an iterator
# is abandoned in its traversal.
#
# Every proof here is an explicit ``close()``. **No test asserts when
# CPython runs a finalizer**, and none needs to: every delivered batch is
# the caller's and every pending one is reachable through close.

ABANDONMENT_POSITIONS = ("fresh", "claim", "constructed", "pending",
                         "committed")


@needs_backend
@pytest.mark.parametrize("stage", ABANDONMENT_POSITIONS)
def test_explicit_close_cleans_up_at_every_abandonment_position(
        monkeypatch, live_storages, tmp_path, stage):
    """The five distinct states an iterator can be closed in, each with a
    non-vacuity control proving that state was really reached, and each
    ending in exactly the same world."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        before = world(loader=loader, sampler=sampler, dataset=dataset,
                       materialize=(0, 1), directory=tmp_path)
        reached = []

        if stage == "fresh":
            iterator = iter(loader)
            reached.append((iterator._txn_serial == 0,
                            sampler._transaction is None))
            iterator.close()
        else:
            if stage == "claim":
                original = NativeTensorDataset.feature_batch

                def hook(self, indices):
                    reached.append((sampler._transaction.status,
                                    loader._iterator._features is None,
                                    settled(live_storages) == baseline))
                    loader._iterator.close()
                    return original(self, indices)

                monkeypatch.setattr(NativeTensorDataset, "feature_batch",
                                    hook)
            elif stage == "constructed":
                original = NativeTensorDataset.target_batch

                def hook(self, indices):
                    reached.append((sampler._transaction.status,
                                    loader._iterator._features is not None,
                                    settled(live_storages) == baseline + 1))
                    loader._iterator.close()
                    return original(self, indices)

                monkeypatch.setattr(NativeTensorDataset, "target_batch", hook)
            elif stage == "pending":
                original = NativeBatchSampler._commit_pending

                def hook(self, serial, owner):
                    reached.append((self._transaction.status,
                                    position(self) == (0, 0),
                                    settled(live_storages) == baseline + 1))
                    loader._iterator.close()
                    return original(self, serial, owner)

                monkeypatch.setattr(NativeBatchSampler, "_commit_pending",
                                    hook)
            else:
                def hook(record):
                    reached.append((sampler._transaction.status,
                                    position(sampler) == (0, 1),
                                    settled(live_storages) == baseline + 1))
                    record.close()
                    return record._features, record._targets

                monkeypatch.setattr(loader_module, "_deliver_batch", hook)

            iterator = iter(loader)
            with pytest.raises(RuntimeError):
                next(iterator)
            monkeypatch.undo()

        # Non-vacuity: the intended state really was the one closed from.
        assert len(reached) == 1, f"the {stage} hook never ran"
        if stage != "fresh":
            expected_status = {"claim": "claim", "constructed": "claim",
                               "pending": "pending",
                               "committed": "committed"}[stage]
            assert reached[0][0] == expected_status, reached
            assert reached[0][1] is True and reached[0][2] is True, reached

        # ...and the world is identical at every one of them.
        after = world(loader=loader, sampler=sampler, dataset=dataset,
                      materialize=(0, 1), directory=tmp_path)
        assert without_sampler_keys(after, "next_serial", "next_token") == \
            without_sampler_keys(before, "next_serial", "next_token")
        assert sampler._transaction is None
        assert sampler._active_iterations == set()
        assert iterator._features is None and iterator._targets is None
        assert iterator._txn_serial == 0
        assert loader._iterator is None
        assert settled(live_storages) == baseline
        # A closed iterator stays a lifecycle error, never an exhaustion.
        with pytest.raises(RuntimeError):
            next(iterator)
        assert iterator.close() is None            # idempotent
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
@pytest.mark.parametrize("when", ["before_first", "between_batches",
                                  "after_supersession", "after_exhaustion",
                                  "after_failed_next"])
def test_an_abandoned_traversal_consumes_only_what_it_delivered(
        monkeypatch, live_storages, when):
    """§9.6's abandonment rows: whatever the iterator's position in its
    traversal, only the batches actually handed to the caller are
    consumed, and the participation is released by the explicit path."""
    loader, sampler, dataset = hardened_pipeline()
    delivered = []
    try:
        baseline = settled(live_storages)
        iterator = iter(loader)
        expected = 0
        if when == "before_first":
            pass
        elif when == "between_batches":
            for _ in range(2):
                features, _ = next(iterator)
                delivered.append(features)
            expected = 2
        elif when == "after_supersession":
            features, _ = next(iterator)
            delivered.append(features)
            expected = 1
            superseding = iter(loader)
            assert iterator._superseded is True
            with pytest.raises(RuntimeError, match=r"supersed"):
                next(iterator)
            superseding.close()
        elif when == "after_exhaustion":
            for features, _ in iterator:
                delivered.append(features)
            expected = 3
            with pytest.raises(StopIteration):
                next(iterator)
        else:
            features, _ = next(iterator)
            delivered.append(features)
            expected = 1

            def boom(record):
                raise Boom("injected: delivery seam")

            monkeypatch.setattr(loader_module, "_deliver_batch", boom)
            with pytest.raises(Boom):
                next(iterator)
            monkeypatch.undo()

        consumed = position(sampler)
        # Exactly the delivered batches, and the canonical boundary when a
        # whole epoch was drained.
        assert consumed == ((1, 0) if expected == 3 else (0, expected))
        # Abandonment: the traversal is dropped, and the explicit close is
        # the proof rather than a finalizer.
        assert iterator.close() is None
        assert iterator.close() is None
        assert sampler._active_iterations == set()
        assert loader._iterator is None
        assert position(sampler) == consumed
        assert sampler._transaction is None
        # Every delivered batch is still the caller's, and still open.
        for features in delivered:
            assert features.closed is False
        assert settled(live_storages) == baseline + len(delivered)
    finally:
        monkeypatch.undo()
        for features in delivered:
            features.close()
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_finalizer_is_a_fallback_and_nothing_depends_on_it(
        live_storages):
    """§15.5: ``__del__`` exists so an abandoned iterator cannot hold a
    participation forever — and that is **all** it is.

    It is invoked explicitly here, never waited for: no assertion in this
    module depends on when CPython collects anything.
    """
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        assert hasattr(loader_module._NativeBatchIterator, "__del__")
        # A closed iterator's finalizer is a no-op.
        closed = iter(loader)
        closed.close()
        assert closed._closed is True
        closed.__del__()
        assert sampler._active_iterations == set()
        # An open one's finalizer releases the participation, exactly as
        # close does — invoked directly, so no timing is involved.
        live = iter(loader)
        assert sampler._active_iterations == {live._token}
        live.__del__()
        assert sampler._active_iterations == set()
        assert live._closed is True
        assert loader._iterator is None
        # It never touches a delivered batch.
        iterator = iter(loader)
        features, _ = next(iterator)
        iterator.__del__()
        assert features.closed is False
        assert settled(live_storages) == baseline + 1
        features.close()
        assert settled(live_storages) == baseline
        # The dataset deliberately has **no** finalizer: it owns two host
        # arrays and no native resource, so one would advertise a lifetime
        # it does not have.
        assert not hasattr(NativeTensorDataset, "__del__")
        assert not hasattr(NativeBatchSampler, "__del__")
        assert not hasattr(NativeDataLoader, "__del__")
    finally:
        loader.close()
        dataset.close()


# ===========================================================================
# 11. Repeated iteration, supersession, and the canonical boundaries
# ===========================================================================

@needs_backend
def test_one_iterator_is_one_epoch_and_the_next_is_the_whole_following_one(
        live_storages):
    """§9.3 + §7.4: an iterator captures the batches remaining in the
    **current** epoch; after the canonical boundary the next iterator runs
    the whole of the following one."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        epoch_zero = sampler.plan(0)
        epoch_one = sampler.plan(1)
        assert epoch_zero != epoch_one

        first = [tuple(t.tolist()) for t in _drain_targets(loader)]
        assert position(sampler) == (1, 0)          # canonical boundary
        second = [tuple(t.tolist()) for t in _drain_targets(loader)]
        assert position(sampler) == (2, 0)
        assert len(first) == len(second) == 3
        assert [tuple(dataset._targets[list(group)].tolist())
                for group in epoch_zero] == first
        assert [tuple(dataset._targets[list(group)].tolist())
                for group in epoch_one] == second
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


def _drain_targets(loader):
    """One iterator's worth of target batches, closing every feature batch
    exactly as a caller must."""
    collected = []
    for features, targets in loader:
        collected.append(targets.copy())
        features.close()
    return collected


@needs_backend
def test_a_restored_mid_epoch_loader_consumes_only_that_epochs_tail(
        live_storages):
    """The behavior an exact resume needs: the first iterator after a
    mid-epoch restoration yields exactly the remaining batches, and the
    next one yields the whole following epoch."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        plan = sampler.plan(0)
        state = loader.state_dict()
        state["sampler"]["cursor"] = 1
        assert loader.load_state_dict(state) is None
        assert sampler.remaining == 2

        tail = _drain_targets(loader)
        assert len(tail) == 2
        assert [tuple(t.tolist()) for t in tail] == [
            tuple(dataset._targets[list(group)].tolist())
            for group in plan[1:]]
        assert position(sampler) == (1, 0)
        assert len(_drain_targets(loader)) == 3
        assert position(sampler) == (2, 0)
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_supersession_consumes_nothing_and_is_refused_mid_transaction(
        monkeypatch, live_storages):
    """§9.2: between transactions a new ``iter()`` supersedes the old one
    and consumes nothing; **during** one it is refused outright, because
    detaching the owner of an undelivered batch would strand both."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        first = iter(loader)
        features, _ = next(first)
        features.close()
        before = position(sampler)
        state_before = loader.state_dict()

        second = iter(loader)
        assert position(sampler) == before          # nothing consumed
        assert loader.state_dict() == state_before
        assert first._superseded is True
        assert second._superseded is False
        assert loader._iterator is second
        # The superseded traversal raises a lifecycle error, not an
        # exhaustion — a ``for`` loop must never swallow this.
        with pytest.raises(RuntimeError, match=r"supersed"):
            next(first)
        assert position(sampler) == before
        # Both participations are live until the superseded one releases.
        assert len(sampler._active_iterations) == 2
        first.close()
        assert sampler._active_iterations == {second._token}
        assert loader._iterator is second           # close did not detach it

        # Mid-transaction, supersession is refused and nothing moves.
        refusals = []

        def probe(record):
            with pytest.raises(RuntimeError):
                iter(loader)
            refusals.append((loader._iterator is second,
                             len(sampler._active_iterations)))
            return record._features, record._targets

        monkeypatch.setattr(loader_module, "_deliver_batch", probe)
        features, _ = next(second)
        monkeypatch.undo()
        assert refusals == [(True, 1)]
        features.close()
        second.close()
        assert settled(live_storages) == baseline
    finally:
        monkeypatch.undo()
        loader.close()
        dataset.close()


@needs_backend
def test_exhaustion_close_and_supersession_stay_three_distinct_states(
        live_storages):
    """Conflating them would let a ``for`` loop silently swallow a
    lifecycle error, so each keeps its own outcome, repeatedly."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        exhausted = iter(loader)
        for features, _ in exhausted:
            features.close()
        for _ in range(3):
            with pytest.raises(StopIteration):
                next(exhausted)
        assert exhausted.close() is None            # idempotent after
        assert exhausted.close() is None
        # ...and closing an exhausted iterator does not turn it into a
        # RuntimeError-raising one retroactively? It does — a close is a
        # close, and that is the documented, distinct outcome.
        with pytest.raises(RuntimeError):
            next(exhausted)

        closed = iter(loader)
        assert closed.close() is None
        for _ in range(3):
            with pytest.raises(RuntimeError):
                next(closed)

        superseded = iter(loader)
        current = iter(loader)
        for _ in range(3):
            with pytest.raises(RuntimeError, match=r"supersed"):
                next(superseded)
        superseded.close()
        current.close()
        assert sampler._active_iterations == set()
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()


# ===========================================================================
# 12. Lifecycle and close ordering (§15.1-§15.6)
# ===========================================================================

@needs_backend
def test_dataset_close_is_idempotent_and_keeps_metadata_readable(
        live_storages):
    """§5.5/§15.3: metadata survives close, materialization refuses, and
    planning and state keep working over the closed dataset."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        baseline = settled(live_storages)
        identity = dataset.identity()
        fingerprint = dataset.fingerprint
        representation = repr(dataset)
        state = loader.state_dict()
        before = sampler_view(sampler)

        assert dataset.close() is None
        assert dataset.close() is None
        assert dataset.closed is True
        # Metadata is all still there, and unchanged.
        assert dataset.identity() == identity
        assert dataset.fingerprint == fingerprint
        assert dataset.samples == 8
        assert len(dataset) == 8
        assert dataset.feature_shape == (2,)
        assert dataset.dtype == "float64"
        assert dataset.device == "cpu"
        assert repr(dataset) != representation      # only ``closed`` moved
        assert "closed=True" in repr(dataset)
        # Materialization refuses, before validating anything at all.
        for indices in ((0, 1), (), (99,), "nonsense", None):
            with pytest.raises(RuntimeError, match=r"closed"):
                dataset.feature_batch(indices)
            with pytest.raises(RuntimeError, match=r"closed"):
                dataset.target_batch(indices)
        # Planning and state are untouched, and no position moved.
        assert sampler_view(sampler) == before
        assert loader.state_dict() == state
        assert sampler.plan() == before["plan"]
        assert sampler.next_batch_indices() == before["next_batch_indices"]
        assert settled(live_storages) == baseline
        # There is no reopen, and none may be invented.
        for forbidden in ("open", "reopen", "restore", "reload"):
            assert not hasattr(dataset, forbidden), forbidden
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_loader_close_is_idempotent_and_touches_nothing_it_does_not_own(
        live_storages):
    """§15.3/§15.6: the loader closes its iterator and marks itself
    closed, and closes **neither** the sampler nor the dataset — and never
    a delivered batch."""
    loader, sampler, dataset = hardened_pipeline()
    delivered = None
    try:
        baseline = settled(live_storages)
        iterator = iter(loader)
        delivered, targets = next(iterator)
        delivered_bits = bits(delivered.to_numpy())
        target_values = targets.tolist()
        state = loader.state_dict()

        assert loader.close() is None
        assert loader.close() is None
        assert loader.closed is True
        assert loader._iterator is None
        assert sampler._active_iterations == set()
        # The iterator it owned is closed; the objects it does not own are
        # untouched.
        with pytest.raises(RuntimeError):
            next(iterator)
        assert dataset.closed is False
        assert not hasattr(sampler, "closed")
        assert loader.sampler is sampler
        assert loader.dataset is dataset
        # iter() refuses; state_dict(), repr, and the properties do not.
        with pytest.raises(RuntimeError, match=r"closed"):
            iter(loader)
        assert loader.state_dict() == state
        assert "closed=True" in repr(loader)
        # The delivered batch is completely untouched, and the host
        # targets were not made writable, emptied, or mutated.
        assert delivered.closed is False
        assert bits(delivered.to_numpy()) == delivered_bits
        assert targets.tolist() == target_values
        assert targets.flags.writeable is False
        assert settled(live_storages) == baseline + 1
    finally:
        if delivered is not None:
            delivered.close()
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_recommended_and_the_reversed_close_order_are_both_defined(
        live_storages):
    """§15.4: loader-then-dataset is recommended and gives nothing up;
    dataset-first is **supported and deterministic** — planning and state
    keep working and only materialization refuses."""
    # Recommended order.
    loader, sampler, dataset = hardened_pipeline()
    baseline = settled(live_storages)
    with loader:
        features, _ = next(iter(loader))
        features.close()
    assert loader.closed is True
    dataset.close()
    assert settled(live_storages) == baseline

    # Reversed order: dataset first, loader still live.
    loader, sampler, dataset = hardened_pipeline()
    try:
        iterator = iter(loader)
        features, _ = next(iterator)
        features.close()
        state = loader.state_dict()
        plan = sampler.plan()
        dataset.close()
        # Planning and state keep working...
        assert sampler.plan() == plan
        assert loader.state_dict() == state
        assert sampler.next_batch_indices() == plan[1]
        assert repr(loader)
        # ...and only materialization refuses, having consumed nothing.
        with pytest.raises(RuntimeError, match=r"closed"):
            next(iterator)
        assert loader.state_dict() == state
        assert sampler._transaction is None
        # A state load over a closed dataset is still legal: it needs only
        # the surviving identity metadata.
        iterator.close()
        assert loader.load_state_dict(state) is None
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_sampler_is_unaffected_by_a_dataset_close_in_every_respect():
    """§15.2/§15.4: the sampler owns nothing releasable, so it has no
    ``close()`` and no closed state to reason about."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        before = sampler_view(sampler)
        dataset.close()
        assert sampler_view(sampler) == before
        assert not hasattr(sampler, "close")
        assert not hasattr(sampler, "closed")
        assert not hasattr(sampler, "__enter__")
        assert not hasattr(sampler, "__exit__")
        # A sampler may even be *constructed* over a closed dataset.
        second = NativeBatchSampler(dataset, batch_size=4)
        assert second.batches_per_epoch == 2
        assert second.next_batch_indices() == (0, 1, 2, 3)
    finally:
        loader.close()
        dataset.close()


@needs_backend
def test_the_context_managers_close_and_propagate(live_storages):
    """Both objects that own something are context managers, and both
    close on the exceptional path."""
    dataset = make_dataset(8, 2)
    baseline = settled(live_storages)
    try:
        sampler = NativeBatchSampler(dataset, batch_size=3)
        with NativeDataLoader(sampler) as loader:
            assert loader.closed is False
            with iter(loader) as iterator:
                features, _ = next(iterator)
                features.close()
            assert sampler._active_iterations == set()
        assert loader.closed is True

        loader = NativeDataLoader(sampler)
        with pytest.raises(Boom, match="propagated"):
            with loader:
                raise Boom("propagated")
        assert loader.closed is True

        loader = NativeDataLoader(sampler)
        with pytest.raises(Boom, match="propagated"):
            with iter(loader) as iterator:
                raise Boom("propagated")
        assert sampler._active_iterations == set()
        with pytest.raises(RuntimeError):
            next(iterator)
        loader.close()

        with make_dataset(4, 2) as scoped:
            assert scoped.closed is False
        assert scoped.closed is True
        assert settled(live_storages) == baseline
    finally:
        dataset.close()


# ===========================================================================
# 13. The checkpoint boundary, adversarially (§13, §14)
# ===========================================================================
#
# J5 proved the ordinary workflow. J7's job is to prove it survives a
# failure: a checkpoint taken **immediately after a failed delivery**
# resumes the same candidate batch, through a real version-3 archive, into
# an entirely fresh object graph.

GRAPH_FEATURES = 4
GRAPH_CLASSES = 3
GRAPH_HIDDEN = 5
GRAPH_SAMPLES = 12
GRAPH_BATCH = 5                     # ceil(12 / 5) == 3 batches per epoch


class HardeningModel(NativeModule):
    """Trainable parameters, persistent buffers, and a **shared**
    generator alias topology, so every restored family is nontrivial and
    none can be recovered from the others."""

    def __init__(self, *, dtype=None, in_seed=1, out_seed=2,
                 shared_seed=101, own_seed=202):
        super().__init__()
        self.linear_in = NativeLinear(GRAPH_FEATURES, GRAPH_HIDDEN,
                                      seed=in_seed, dtype=dtype)
        self.norm = NativeBatchNorm1d(GRAPH_HIDDEN, dtype=dtype)
        self.relu = NativeReLU()
        shared = NativeGenerator(shared_seed)
        self.drop_a = NativeDropout(0.25, generator=shared)
        self.drop_b = NativeDropout(0.25, generator=shared)
        self.drop_c = NativeDropout(0.5, seed=own_seed)
        self.linear_out = NativeLinear(GRAPH_HIDDEN, GRAPH_CLASSES,
                                       seed=out_seed, dtype=dtype)

    def forward(self, x):
        hidden = self.relu(self.norm(self.linear_in(x)))
        hidden = self.drop_c(self.drop_b(self.drop_a(hidden)))
        return self.linear_out(hidden)


class Graph:
    """One complete object graph. A test convenience; no production
    analogue exists or is implied."""

    __slots__ = ("model", "optimizer", "dataset", "sampler", "loader")

    def __init__(self, model, optimizer, dataset, sampler, loader):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.sampler = sampler
        self.loader = loader


def graph_arrays(samples=GRAPH_SAMPLES, offset=0.0):
    values = ((np.arange(samples * GRAPH_FEATURES, dtype=np.float64)
               .reshape(samples, GRAPH_FEATURES) % 7.0) - 3.0) + offset
    targets = np.arange(samples, dtype=np.int64) % GRAPH_CLASSES
    return values, targets


def build_graph(dtype=None, *, in_seed=1, out_seed=2, shared_seed=101,
                own_seed=202, lr=0.01, batch_size=GRAPH_BATCH, shuffle=True,
                seed=7, drop_last=False, samples=GRAPH_SAMPLES, offset=0.0,
                dataset=None):
    if dataset is None:
        values, targets = graph_arrays(samples, offset)
        dataset = NativeTensorDataset(values, targets, dtype=dtype)
    sampler = NativeBatchSampler(dataset, batch_size=batch_size,
                                 shuffle=shuffle, seed=seed,
                                 drop_last=drop_last)
    loader = NativeDataLoader(sampler)
    model = HardeningModel(dtype=dtype, in_seed=in_seed, out_seed=out_seed,
                           shared_seed=shared_seed, own_seed=own_seed)
    optimizer = NativeAdam(model.parameters(), lr=lr)
    return Graph(model, optimizer, dataset, sampler, loader)


def close_graph(graph):
    """Explicit cleanup in the §15.4 order: loader (and so its iterator),
    optimizer moments, dataset snapshots, then every unique parameter and
    buffer. Nothing here relies on garbage collection."""
    graph.loader.close()
    graph.optimizer.close()
    graph.dataset.close()
    seen = set()
    for _, tensor in (list(graph.model.named_parameters())
                      + list(graph.model.named_buffers())):
        if tensor is not None and id(tensor) not in seen:
            seen.add(id(tensor))
            tensor.close()


def train_steps(graph, iterator, steps):
    """Genuine training steps, each moving parameters, running buffers,
    Adam moments and step counters, all three generator streams, and the
    committed loader position. Every delivered batch is closed here,
    because it is **the caller's**."""
    from tensorforge.experimental import NativeCrossEntropyLoss

    loss_fn = NativeCrossEntropyLoss()
    for _ in range(steps):
        features, targets = next(iterator)
        try:
            logits = graph.model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            graph.optimizer.step()
            graph.optimizer.zero_grad()
            loss.close()
            logits.close()
        finally:
            features.close()


def graph_fingerprint(graph):
    """Everything a restoration must reproduce exactly, in raw IEEE-754
    bits — no tolerance anywhere."""
    model, optimizer = graph.model, graph.optimizer
    return {
        "parameters": {name: (bits(tensor.to_numpy()), tensor.version)
                       for name, tensor in model.named_parameters()},
        "buffers": {name: bits(tensor.to_numpy())
                    for name, tensor in model.named_buffers()},
        "generators": {name: generator.state()
                       for name, generator in model.named_generators()},
        "topology": tuple(
            (path, [id(g) for g in model.generators()].index(
                id(getattr(model, path).generator)))
            for path in ("drop_a", "drop_b", "drop_c")),
        "optimizer": json.dumps(optimizer.state_dict(), sort_keys=True,
                                default=repr),
    }


def batch_record(features, targets):
    """One delivered batch as one comparable value, in raw bits."""
    return (features.shape, features.dtype, bits(features.to_numpy()),
            targets.tolist(), targets.dtype.str, bool(targets.flags.writeable))


def manifest_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def training_metadata(step, loader_state):
    """The caller convention J5 recommends. It is a **caller** convention:
    no production constant spells any of these keys."""
    return {"training": {"next_step": step, "data_loader": loader_state}}


@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_checkpoint_taken_after_a_failed_delivery_resumes_it_exactly(
        tmp_path, monkeypatch, live_storages, dtype):
    """**The load-bearing J7 proof.**

    A delivery fails at the seam after the candidate position was
    committed. The rollback restores everything, so a loader state taken
    immediately afterwards is byte-identical to the pre-failure one — and
    a real version-3 archive carrying it restores a wholly fresh graph
    that delivers **exactly the batch the failed call was about to
    deliver**, once, with bit-identical features and targets.
    """
    baseline = settled(live_storages)
    path = str(tmp_path / "after_failed_delivery.npz")

    source = build_graph(dtype=dtype)
    restored = None
    try:
        # 2. A genuinely mid-epoch position: batches are still owed, and
        #    the cursor is neither 0 nor the last step of the epoch.
        iterator = iter(source.loader)
        train_steps(source, iterator, 1)
        iterator.close()
        assert source.sampler.batches_per_epoch == 3
        assert 0 < source.sampler.cursor < source.sampler.batches_per_epoch
        assert source.sampler.remaining == 2

        # 3. Everything the rollback has to restore.
        pre_loader_state = source.loader.state_dict()
        pre_sampler_state = source.sampler.state_dict()
        pre_graph = graph_fingerprint(source)
        candidate = source.sampler.next_batch_indices()
        candidate_after = source.sampler._next_position(source.sampler.epoch,
                                                        source.sampler.cursor)
        oracle_features = source.dataset.feature_batch(candidate)
        try:
            oracle = batch_record(oracle_features,
                                  source.dataset.target_batch(candidate))
        finally:
            oracle_features.close()
        pre_storage = settled(live_storages)

        # 4. The injection, at the seam, after the candidate is committed.
        observed = []
        original_seam = loader_module._deliver_batch
        captured = []

        def exploding_seam(record):
            observed.append({
                "position": position(source.sampler),
                "status": source.sampler._transaction.status,
                "refuses_loader": refuses_state(source.loader),
                "refuses_sampler": refuses_state(source.sampler),
                "storage": settled(live_storages),
            })
            captured.append(record._features)
            raise Boom("injected: delivery seam")

        monkeypatch.setattr(loader_module, "_deliver_batch", exploding_seam)
        iterator = iter(source.loader)
        with pytest.raises(Boom, match="injected"):
            next(iterator)
        monkeypatch.setattr(loader_module, "_deliver_batch", original_seam)

        # Non-vacuity: the seam ran, at the committed phase, with the
        # candidate applied and every snapshot refused.
        assert len(observed) == 1, "the delivery seam never ran"
        assert observed[0]["status"] == "committed"
        assert observed[0]["position"] == candidate_after
        assert observed[0]["position"] != (pre_loader_state["sampler"]["epoch"],
                                           pre_loader_state["sampler"]["cursor"])
        assert observed[0]["refuses_loader"] is True
        assert observed[0]["refuses_sampler"] is True
        assert observed[0]["storage"] == pre_storage + 1

        # 5. The rollback was complete, in every family.
        assert captured[0].closed is True
        assert source.loader.state_dict() == pre_loader_state
        assert source.sampler.state_dict() == pre_sampler_state
        assert source.sampler.next_batch_indices() == candidate
        assert source.sampler._transaction is None
        assert graph_fingerprint(source) == pre_graph
        assert settled(live_storages) == pre_storage
        iterator.close()

        # 6/7. The state taken immediately afterwards is the same state.
        post_loader_state = source.loader.state_dict()
        assert post_loader_state == pre_loader_state

        # 8. A real archive, with the state as ordinary caller metadata.
        save_native_checkpoint(path, source.model,
                               optimizer=source.optimizer,
                               metadata=training_metadata(1,
                                                          post_loader_state))
        manifest = manifest_of(path)
        assert manifest["format"] == "tensorforge.native_checkpoint"
        assert manifest["format_version"] == 3
        assert manifest["metadata"]["training"]["data_loader"] \
            == post_loader_state
        assert "data_loader" not in manifest
        assert "loader" not in manifest

        # 9. Discard the whole saving graph.
        close_graph(source)
        assert settled(live_storages) == baseline

        # 10. An entirely fresh, deliberately *differently configured*
        #     graph — different seeds, learning rate, batch size, shuffle,
        #     and position — so nothing can be inherited by accident.
        restored = build_graph(dtype=dtype, in_seed=91, out_seed=92,
                               shared_seed=93, own_seed=94, lr=0.05,
                               batch_size=GRAPH_BATCH + 2, shuffle=False,
                               seed=31337, drop_last=False)
        assert graph_fingerprint(restored) != pre_graph
        assert restored.sampler.next_batch_indices() != candidate

        # 11/12. Checkpoint first, then the loader state — that order.
        metadata = load_native_checkpoint(path, restored.model,
                                          optimizer=restored.optimizer)
        assert graph_fingerprint(restored) == pre_graph
        assert restored.sampler.next_batch_indices() != candidate
        restored.loader.load_state_dict(
            metadata["training"]["data_loader"])

        # 13. The restored next batch is the exact failed candidate.
        assert restored.sampler.next_batch_indices() == candidate
        assert restored.loader.state_dict() == pre_loader_state
        cursor_before = restored.sampler.cursor

        # 14/15. Deliver it, and compare exact bits.
        features, targets = next(iter(restored.loader))
        try:
            assert batch_record(features, targets) == oracle
        finally:
            features.close()

        # 16. The position advanced exactly once.
        assert restored.sampler.cursor == cursor_before + 1
        assert restored.loader.state_dict()["sampler"]["cursor"] \
            == cursor_before + 1
        assert restored.sampler.epoch == pre_loader_state["sampler"]["epoch"]
    finally:
        # 17/18.
        if restored is not None:
            close_graph(restored)
        monkeypatch.undo()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_failed_checkpoint_save_never_reaches_the_loader(
        tmp_path, live_storages):
    """A checkpoint save that is refused must leave the loader's whole
    private world untouched — and it must never have called a loader
    method to begin with, because no checkpoint code knows a loader
    exists."""
    baseline = settled(live_storages)
    graph = build_graph()
    path = tmp_path / "never_written.npz"
    try:
        iterator = iter(graph.loader)
        train_steps(graph, iterator, 1)
        iterator.close()
        settled_graph = settled(live_storages)
        before_state = graph.loader.state_dict()
        before_sampler = sampler_view(graph.sampler)
        before_graph = graph_fingerprint(graph)
        before_files = globals_view(directory=tmp_path)

        # Non-JSON metadata is refused before the destination moves.
        for bad in ({"array": np.zeros(3)},
                    {"tensor": object()},
                    {1: "int key"},
                    {"nan": float("nan")},
                    {"inf": float("inf")}):
            with pytest.raises((TypeError, ValueError)):
                save_native_checkpoint(str(path), graph.model,
                                       optimizer=graph.optimizer,
                                       metadata=bad)
            assert not path.exists()
        # A bad destination is refused too.
        with pytest.raises((TypeError, ValueError, OSError)):
            save_native_checkpoint(str(tmp_path / "missing" / "x.npz"),
                                   graph.model, optimizer=graph.optimizer)

        # Nothing about the loader, sampler, or graph moved, and no file
        # or temporary was left behind.
        assert graph.loader.state_dict() == before_state
        assert sampler_view(graph.sampler) == before_sampler
        assert graph_fingerprint(graph) == before_graph
        assert globals_view(directory=tmp_path) == before_files
        assert graph.sampler._transaction is None
        assert graph.sampler._active_iterations == set()
        # No batch was allocated by any of it: the graph's own storage is
        # exactly what it was before the refused saves.
        assert settled(live_storages) == settled_graph
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


@needs_backend
def test_a_failed_checkpoint_load_never_reaches_the_loader(
        tmp_path, live_storages, monkeypatch):
    """The load direction: a refused ``load_native_checkpoint`` calls no
    loader method, allocates no batch, and leaves the loader exactly as it
    was — there is no discovery in either direction."""
    baseline = settled(live_storages)
    graph = build_graph()
    path = str(tmp_path / "for_a_failing_load.npz")
    try:
        iterator = iter(graph.loader)
        train_steps(graph, iterator, 1)
        iterator.close()
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                              metadata=training_metadata(
                                  1, graph.loader.state_dict()))
        before_state = graph.loader.state_dict()
        before_sampler = sampler_view(graph.sampler)
        settled_graph = settled(live_storages)

        # Every loader entry point is watched: none of them may be called
        # by the checkpoint, whatever it does. Only calls on **this**
        # loader are recorded — the test's own cleanup of a throwaway
        # graph below is the test's doing, not the checkpoint's.
        touched = []
        watched_loader = graph.loader
        for name in ("state_dict", "load_state_dict", "__iter__", "close"):
            original = getattr(NativeDataLoader, name)

            def watched(self, *args, _name=name, _original=original,
                        **kwargs):
                if self is watched_loader:
                    touched.append(_name)
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(NativeDataLoader, name, watched)

        # An incompatible model: the load is refused, atomically.
        wrong = build_graph(dtype="float32")
        try:
            with pytest.raises((ValueError, TypeError, RuntimeError)):
                load_native_checkpoint(path, wrong.model,
                                       optimizer=wrong.optimizer)
        finally:
            close_graph(wrong)
        # A missing archive, likewise.
        with pytest.raises((OSError, FileNotFoundError, ValueError)):
            load_native_checkpoint(str(tmp_path / "absent.npz"), graph.model,
                                   optimizer=graph.optimizer)
        monkeypatch.undo()

        assert touched == [], touched
        assert graph.loader.state_dict() == before_state
        assert sampler_view(graph.sampler) == before_sampler
        assert graph.sampler._transaction is None
        assert settled(live_storages) == settled_graph
    finally:
        monkeypatch.undo()
        close_graph(graph)
    assert settled(live_storages) == baseline


@needs_backend
def test_a_loader_failure_after_a_successful_checkpoint_rolls_back_nothing(
        tmp_path, live_storages):
    """§13.5's honest non-atomicity boundary, with the whole world
    fingerprinted: the checkpoint half **stays restored** when the loader
    half fails, because there is no cross-object transaction and none may
    be added. The caller's remedy is to discard and repeat both calls, and
    the same archive is still perfectly reusable."""
    baseline = settled(live_storages)
    path = str(tmp_path / "non_atomic.npz")
    source = build_graph()
    incompatible = None
    compatible = None
    try:
        iterator = iter(source.loader)
        train_steps(source, iterator, 1)
        iterator.close()
        saved_state = source.loader.state_dict()
        saved_graph = graph_fingerprint(source)
        save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                               metadata=training_metadata(1, saved_state))
        close_graph(source)

        # A restored graph whose **dataset holds different values**, so
        # the loader half must refuse on the fingerprint.
        incompatible = build_graph(in_seed=51, out_seed=52, offset=0.5)
        before_loader = loader_view(incompatible.loader)
        metadata = load_native_checkpoint(path, incompatible.model,
                                          optimizer=incompatible.optimizer)
        # The checkpoint half succeeded, completely.
        assert graph_fingerprint(incompatible) == saved_graph
        with pytest.raises(ValueError, match=r"content fingerprints differ"):
            incompatible.loader.load_state_dict(
                metadata["training"]["data_loader"])
        # Nothing rolled back: the model, optimizer, and generators are
        # still restored, and the loader is still exactly where it was.
        assert graph_fingerprint(incompatible) == saved_graph
        assert loader_view(incompatible.loader) == before_loader
        assert incompatible.sampler.epoch == 0
        assert incompatible.sampler.cursor == 0

        # The archive is untouched and still reusable by a compatible
        # graph — the caller's documented remedy.
        compatible = build_graph(in_seed=61, out_seed=62)
        metadata = load_native_checkpoint(path, compatible.model,
                                          optimizer=compatible.optimizer)
        assert compatible.loader.load_state_dict(
            metadata["training"]["data_loader"]) is None
        assert compatible.loader.state_dict() == saved_state
        assert graph_fingerprint(compatible) == saved_graph
    finally:
        for graph in (incompatible, compatible):
            if graph is not None:
                close_graph(graph)
    assert settled(live_storages) == baseline


@needs_backend
def test_the_checkpoint_preserves_loader_metadata_without_interpreting_it(
        tmp_path, live_storages):
    """§13.4/§13.6: the archive validates JSON-compatibility and nothing
    else. Malformed loader state survives the round trip **unchanged** and
    is rejected by the *loader*; an absent one gets no default; and the
    returned metadata is the caller's, independent of the archive."""
    baseline = settled(live_storages)
    graph = build_graph()
    try:
        iterator = iter(graph.loader)
        train_steps(graph, iterator, 1)
        iterator.close()
        good = graph.loader.state_dict()

        malformed = [
            {"format": "tensorforge.native_data_loader", "format_version": 1,
             "sampler": {}},
            {"format": "tensorforge.native_data_loader",
             "format_version": 2, "sampler": good["sampler"]},
            {"format": "wrong", "format_version": 1,
             "sampler": good["sampler"]},
            {"format": "tensorforge.native_data_loader", "format_version": 1},
            replace(good, "sampler",
                    replace(good["sampler"], "cursor", 99)),
        ]
        for index, state in enumerate(malformed):
            path = str(tmp_path / f"malformed_{index}.npz")
            save_native_checkpoint(path, graph.model,
                                   optimizer=graph.optimizer,
                                   metadata=training_metadata(1, state))
            # Preserved by the archive, byte for byte...
            assert (manifest_of(path)["metadata"]["training"]["data_loader"]
                    == state)
            metadata = load_native_checkpoint(path, graph.model,
                                              optimizer=graph.optimizer)
            assert metadata["training"]["data_loader"] == state
            # ...and rejected by the loader, not by the checkpoint.
            before = sampler_view(graph.sampler)
            with pytest.raises((TypeError, ValueError)):
                graph.loader.load_state_dict(
                    metadata["training"]["data_loader"])
            assert sampler_view(graph.sampler) == before

        # An absent loader state gets no default invented for it.
        path = str(tmp_path / "no_loader_state.npz")
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata={"training": {"next_step": 3}})
        metadata = load_native_checkpoint(path, graph.model,
                                          optimizer=graph.optimizer)
        assert metadata == {"training": {"next_step": 3}}
        assert "data_loader" not in metadata["training"]
        assert set(manifest_of(path)) >= {"format", "format_version",
                                          "metadata"}
        assert "data_loader" not in manifest_of(path)

        # The returned metadata is the caller's: editing it reaches
        # neither the archive nor a second load.
        path = str(tmp_path / "independent.npz")
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata=training_metadata(4, good))
        first = load_native_checkpoint(path, graph.model,
                                       optimizer=graph.optimizer)
        first["training"]["data_loader"]["sampler"]["cursor"] = 99
        first["training"]["next_step"] = -1
        second = load_native_checkpoint(path, graph.model,
                                        optimizer=graph.optimizer)
        assert second == training_metadata(4, good)
        assert second["training"]["data_loader"] == good
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


@needs_backend
def test_a_wrong_dtype_loader_state_from_an_archive_is_refused(
        tmp_path, live_storages):
    """§19: the dataset identity carries the feature dtype, so a float32
    run's loader state cannot be restored into a float64 pipeline — even
    though the two datasets hold the same numbers."""
    baseline = settled(live_storages)
    source = build_graph(dtype="float32")
    target = None
    try:
        iterator = iter(source.loader)
        train_steps(source, iterator, 1)
        iterator.close()
        path = str(tmp_path / "float32_state.npz")
        state = source.loader.state_dict()
        assert state["sampler"]["dataset"]["feature_dtype"] == "float32"
        save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                               metadata=training_metadata(1, state))
        close_graph(source)
        source = None

        target = build_graph(dtype="float64")
        before = sampler_view(target.sampler)
        with np.load(path, allow_pickle=False) as archive:
            carried = json.loads(archive["manifest"].tobytes().decode("utf-8"))
        assert (carried["metadata"]["training"]["data_loader"]["sampler"]
                ["dataset"]["feature_dtype"]) == "float32"
        with pytest.raises(ValueError, match=r"feature dtype"):
            target.loader.load_state_dict(
                carried["metadata"]["training"]["data_loader"])
        assert sampler_view(target.sampler) == before
    finally:
        for graph in (source, target):
            if graph is not None:
                close_graph(graph)
    assert settled(live_storages) == baseline


# ===========================================================================
# 14. The concurrency contract — documented and unprotected (§16)
# ===========================================================================
#
# **This module starts no thread and asserts no race is safe.** §16 says
# concurrent use is *undefined and unprotected*, and the honest way to
# test that is to prove the two halves of the statement: there is no lock
# or worker anywhere in the production source, and the documentation says
# so in terms a parser can check — with a negative control proving the
# parser can fail.

CONCURRENCY_NAMES = (
    "threading", "thread", "Thread", "Lock", "RLock", "Semaphore",
    "BoundedSemaphore", "Condition", "Event", "Barrier", "queue", "Queue",
    "SimpleQueue", "multiprocessing", "Process", "Pool", "ThreadPool",
    "concurrent", "futures", "Future", "ThreadPoolExecutor",
    "ProcessPoolExecutor", "asyncio", "__aiter__", "__anext__", "acquire",
    "release", "mutex", "atomic", "fork", "spawn", "daemon", "join",
    # the pipeline's own forbidden surface, which would need one of the
    # above to exist
    "prefetch", "num_workers", "workers", "worker", "pin_memory",
    "collate", "collate_fn", "transform",
    # and the process-wide state guard the Phase-J objects deliberately do
    # not join (§16.3)
    "state_transaction", "_native_state_lock",
)


@needs_backend
@pytest.mark.parametrize("module", PIPELINE_MODULES)
def test_no_pipeline_module_contains_a_lock_thread_or_queue(module):
    """§16.1/§16.3, from the source: no lock, no thread, no queue, no
    future, no async primitive, and no acquisition of the process-wide
    state-replacement guard.

    Read from the **AST**, not the text: these modules explain at length
    what they deliberately do not do, so a substring scan would fail on
    prose that documents the prohibition.
    """
    names = code_identifiers(f"src/tensorforge/experimental/{module}")
    offenders = sorted(name for name in CONCURRENCY_NAMES if name in names)
    assert offenders == [], (module, offenders)
    tree = ast.parse((PACKAGE / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.AsyncFunctionDef, ast.Await,
                                     ast.AsyncFor, ast.AsyncWith)), module
    # ...and the module imports nothing from the state-lock module either.
    text = (PACKAGE / module).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+.*_native_state_lock", text,
                         re.M), module


def test_the_concurrency_scanner_can_actually_fail():
    """Negative control for the scan above: it must find each family when
    one is genuinely present, and it must not fire on prose."""
    planted = (
        "import threading\n"
        "from queue import Queue\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        self._lock = threading.Lock()\n"
        "        self._lock.acquire()\n"
        "async def stream():\n"
        "    await go()\n"
    )
    names = set()
    tree = ast.parse(planted)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            for alias in node.names:
                names.add(alias.name)
    for expected in ("threading", "queue", "Queue", "Lock", "acquire",
                     "ThreadPoolExecutor"):
        assert expected in names, expected
    assert any(isinstance(node, ast.AsyncFunctionDef)
               for node in ast.walk(tree))
    assert any(isinstance(node, ast.Await) for node in ast.walk(tree))
    # ...and a docstring that merely *mentions* the words is clean.
    prose = ast.parse('"""No lock, no thread, no queue, no prefetch."""\n')
    prose_names = {node.id for node in ast.walk(prose)
                   if isinstance(node, ast.Name)}
    assert prose_names == set()


def test_the_design_states_the_concurrency_contract_as_unsupported():
    """The documentation half, section-scoped and checked by a parser that
    the control below proves can fail."""
    body = _design_section(16)
    flat = _flat(body).lower()
    for required in ("not thread-safe", "none of them contains a lock",
                     "one thread at a time",
                     "external locking is required", "undefined",
                     "not supported, not protected, not claimed",
                     "guards reentrancy, not concurrency",
                     "join no lock order"):
        assert required.lower() in flat, required
    # ...and it does not promise safety.
    for forbidden in ("thread-safe in practice", "safe for concurrent use",
                      "concurrency is supported", "we take a lock"):
        assert forbidden.lower() not in flat, forbidden
    # §16.3's structural reason, and the two orders it does not join.
    assert "_native_state_lock" in body or "state_transaction" in body
    assert "id()" in body or "id()-sorted" in body


def test_the_concurrency_document_parser_can_actually_fail():
    """Negative control: the checker must reject a §16 that dropped the
    statement, and must reject one that reversed it."""
    required = ("not thread-safe", "none of them contains a lock",
                "external locking is required", "undefined")
    honest = _flat(_design_section(16)).lower()
    assert [term for term in required if term not in honest] == []
    for doctored in (
        honest.replace("not thread-safe", "thread-safe"),
        honest.replace("none of them contains a lock", "each holds a lock"),
        honest.replace("external locking is required",
                       "no external locking is needed"),
    ):
        assert [term for term in required if term not in doctored] != []


def _design_section(number):
    """The body of top-level design section ``number``."""
    text = (REPO_ROOT / "docs" / "native_data_pipeline_design.md").read_text(
        encoding="utf-8")
    marker = f"\n## {number}."
    assert marker in text, number
    body = text.split(marker, 1)[1]
    following = re.search(r"\n## \d+\.", body)
    return body[:following.start()] if following else body


def _flat(text):
    """Whitespace-flattened, emphasis-stripped text, so a claim split
    across lines or wrapped in markdown still reads as one sentence."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text))


def test_this_hardening_module_starts_no_thread_itself():
    """The discipline the contract requires of its own tests: no thread,
    no pool, no async, and therefore no test whose result depends on
    scheduling."""
    names = code_identifiers("tests/test_native_data_hardening.py")
    for forbidden in ("threading", "Thread", "multiprocessing", "asyncio",
                      "ThreadPoolExecutor", "ProcessPoolExecutor",
                      "SimpleQueue", "start_new_thread"):
        assert forbidden not in names, forbidden
    tree = ast.parse((REPO_ROOT / "tests"
                      / "test_native_data_hardening.py").read_text(
                          encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.AsyncFunctionDef, ast.Await)), node


# ===========================================================================
# 15. Stable / native isolation (§18)
# ===========================================================================

@needs_backend
def test_a_stable_tensor_is_never_accepted_anywhere_in_the_pipeline(
        live_storages):
    """§18: a ``tensorforge.Tensor`` is a ``TypeError`` at the same place
    any other non-``ndarray`` is. There is no bridge and no conversion."""
    baseline = settled(live_storages)
    values, targets = host_arrays(6, 2)
    stable_features = tensorforge.Tensor(values)
    stable_targets = tensorforge.Tensor(targets.astype(np.float64))
    with pytest.raises(TypeError):
        NativeTensorDataset(stable_features, targets)
    with pytest.raises(TypeError):
        NativeTensorDataset(values, stable_targets)
    with pytest.raises(TypeError):
        NativeTensorDataset(stable_features, stable_targets)
    dataset = NativeTensorDataset(values, targets)
    try:
        # ...and a stable Tensor is not a sampler's dataset, nor a
        # loader's sampler.
        with pytest.raises(TypeError):
            NativeBatchSampler(stable_features, batch_size=2)
        with pytest.raises(TypeError):
            NativeDataLoader(dataset)
        sampler = NativeBatchSampler(dataset, batch_size=2)
        with pytest.raises(TypeError):
            NativeDataLoader(sampler.dataset)
        # A native tensor is not a dataset input either: the ingress
        # boundary is a NumPy array, explicitly.
        native = NativeTensor.from_array(values)
        try:
            with pytest.raises(TypeError):
                NativeTensorDataset(native, targets)
        finally:
            native.close()
        # ...and a stable Tensor is not a valid index container.
        with pytest.raises(TypeError):
            dataset.feature_batch(tensorforge.Tensor(np.array([0.0, 1.0])))
    finally:
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_two_lines_stay_structurally_separate():
    """§18, from the source and the registries: no import in either
    direction, no routing, no global, and no name crossing over."""
    # The stable package names no experimental module.
    stable = REPO_ROOT / "src" / "tensorforge"
    for path in sorted(stable.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+.*\bexperimental\b",
                             text, re.M), path.name
        assert not re.search(r"^\s*(from|import)\s+.*\bbackends\b", text,
                             re.M), path.name
    for sub in ("nn", "optim"):
        for path in sorted((stable / sub).glob("*.py")):
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"^\s*(from|import)\s+.*\bexperimental\b",
                                 text, re.M), path.name
    # ...and no pipeline module names the stable mini-batch story.
    for module in PIPELINE_MODULES:
        names = code_identifiers(f"src/tensorforge/experimental/{module}")
        for forbidden in ("batches", "train_test_split", "tensorforge.data",
                          "Tensor"):
            assert forbidden not in names, (module, forbidden)
    # The registry row does not move.
    assert cpp.backend_info()["stable_framework_integration"] is False
    # No Phase-J name entered the stable surface.
    for name in ("NativeTensorDataset", "NativeBatchSampler",
                 "NativeDataLoader"):
        assert name not in tensorforge.__all__, name
        assert not hasattr(tensorforge, name), name
    # The stable mini-batch iterator is still there, and still stable-only.
    assert "batches" in tensorforge.__all__
    assert "batches" not in experimental.__all__
    # No module-level default, global, or registry of any kind.
    for module in (dataset_module, sampler_module, loader_module):
        for forbidden in ("default_dataset", "DEFAULT_DATASET",
                          "default_loader", "DEFAULT_LOADER", "current",
                          "CURRENT", "global_loader", "get_loader",
                          "set_loader", "register", "registry"):
            assert not hasattr(module, forbidden), (module.__name__,
                                                    forbidden)


@needs_backend
def test_importing_the_stable_package_still_loads_no_native_library():
    """The isolation Phase J must not break, proved in a **fresh
    interpreter** rather than from an already-populated ``sys.modules``."""
    import subprocess

    script = (
        "import sys\n"
        "import tensorforge\n"
        "loaded = [name for name in sys.modules\n"
        "          if name.startswith('tensorforge.experimental')\n"
        "          or name.startswith('tensorforge.backends')]\n"
        "print(sorted(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=str(REPO_ROOT), check=True)
    assert result.stdout.strip() == "[]", result.stdout


@needs_backend
def test_no_stable_object_is_silently_accepted_and_no_native_one_escapes(
        live_storages):
    """No implicit conversion in either direction, and no input-driven
    routing: there is one implementation and no dispatch."""
    baseline = settled(live_storages)
    loader, sampler, dataset = hardened_pipeline()
    try:
        features, targets = next(iter(loader))
        try:
            # What a batch is: a native tensor and a host int64 array.
            assert isinstance(features, NativeTensor)
            assert not isinstance(features, tensorforge.Tensor)
            assert isinstance(targets, np.ndarray)
            assert not isinstance(targets, NativeTensor)
            assert targets.dtype == np.int64
            # The stable helper does not accept the native half: it fails
            # rather than silently unwrapping a NativeTensor into a
            # stable mini-batch.
            with pytest.raises((TypeError, ValueError, AttributeError,
                                IndexError)):
                list(tensorforge.batches(features, targets, 2))
        finally:
            features.close()
        loader.close()
    finally:
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


# ===========================================================================
# 16. Dtype and target boundaries (§19)
# ===========================================================================

@needs_backend
@pytest.mark.parametrize("dtype", [None, "float64", "float32"])
def test_the_two_supported_widths_and_nothing_else(live_storages, dtype):
    """§19.1/§19.2: float64 and float32, and an omitted ``dtype`` means
    float64 — at the dataset, and therefore at every batch."""
    baseline = settled(live_storages)
    expected = "float64" if dtype is None else dtype
    loader, sampler, dataset = hardened_pipeline(dtype=dtype)
    try:
        assert dataset.dtype == expected
        assert dataset.device == "cpu"
        features, targets = next(iter(loader))
        try:
            assert features.dtype == expected
            assert features.device == "cpu"
            assert features.requires_grad is False
            assert features.grad is None
            host = features.to_numpy()
            assert str(host.dtype) == expected
            assert host.flags["C_CONTIGUOUS"]
        finally:
            features.close()
    finally:
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_host_dtype_never_chooses_the_native_one(live_storages):
    """§19.3, in both directions: a float32 host array with ``dtype``
    omitted gives a **float64** dataset, and a float64 host array with
    ``dtype="float32"`` gives a float32 one."""
    baseline = settled(live_storages)
    values, targets = host_arrays(6, 2)
    narrow = values.astype(np.float32)
    inferred = NativeTensorDataset(narrow, targets)
    asked = NativeTensorDataset(values, targets, dtype="float32")
    try:
        assert narrow.dtype == np.float32
        assert inferred.dtype == "float64"
        assert asked.dtype == "float32"
        for dataset, expected in ((inferred, "float64"), (asked, "float32")):
            batch = dataset.feature_batch((0, 1))
            try:
                assert batch.dtype == expected
            finally:
                batch.close()
        # The two identities differ **because** the dtype does, so their
        # states are not interchangeable.
        assert inferred.identity()["feature_dtype"] == "float64"
        assert asked.identity()["feature_dtype"] == "float32"
        assert inferred.fingerprint != asked.fingerprint
    finally:
        inferred.close()
        asked.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_no_dtype_or_device_surface_exists_beside_the_dataset():
    """§19.6/§19.7 and the §3 rule that a class owning no dtype-bearing
    state must not gain a ``dtype``: the sampler and the loader own none,
    the dataset's is immutable, and no ``device`` argument exists
    anywhere."""
    loader, sampler, dataset = hardened_pipeline()
    try:
        assert "dtype" not in inspect.signature(
            NativeBatchSampler.__init__).parameters
        assert "device" not in inspect.signature(
            NativeBatchSampler.__init__).parameters
        assert list(inspect.signature(
            NativeDataLoader.__init__).parameters) == ["self", "sampler"]
        assert "device" not in inspect.signature(
            NativeTensorDataset.__init__).parameters
        # The dataset's dtype parameter is keyword-only, and its property
        # is read-only.
        parameter = inspect.signature(
            NativeTensorDataset.__init__).parameters["dtype"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
        for attribute, value in (("dtype", "float32"), ("device", "cuda"),
                                 ("samples", 0), ("fingerprint", "x"),
                                 ("closed", False)):
            with pytest.raises(AttributeError):
                setattr(dataset, attribute, value)
        for attribute in ("epoch", "cursor", "batch_size", "seed",
                          "shuffle", "drop_last"):
            with pytest.raises(AttributeError):
                setattr(sampler, attribute, 0)
        for attribute in ("closed", "sampler", "dataset"):
            with pytest.raises(AttributeError):
                setattr(loader, attribute, None)
        # No casting, promotion, or movement API of any kind.
        for owner in (NativeTensorDataset, NativeBatchSampler,
                      NativeDataLoader):
            for forbidden in ("astype", "to", "float", "double", "half",
                              "cuda", "cpu", "map_location", "type",
                              "promote", "cast"):
                assert not hasattr(owner, forbidden), (owner.__name__,
                                                       forbidden)
        # Unsupported widths are refused at the one place a width is
        # chosen, by value rather than silently normalized.
        values, targets = host_arrays(4, 2)
        for bad in ("float16", "bfloat16", "float128", "int64", "bool",
                    "complex64", "FLOAT64", "double", ""):
            with pytest.raises(ValueError):
                NativeTensorDataset(values, targets, dtype=bad)
        for bad in (64, 3.5, np.float64, ["float64"], np.dtype("float64")):
            with pytest.raises(TypeError):
                NativeTensorDataset(values, targets, dtype=bad)
    finally:
        loader.close()
        dataset.close()


@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_targets_stay_read_only_owning_host_int64_at_every_width(
        live_storages, dtype):
    """§19.4/§19.5 and §10.2: targets are copied host ``int64`` arrays at
    every feature width, read-only, independently owning, and **never**
    native tensors. No integer tensor dtype exists."""
    baseline = settled(live_storages)
    loader, sampler, dataset = hardened_pipeline(dtype=dtype)
    try:
        iterator = iter(loader)
        first_features, first = next(iterator)
        second_features, second = next(iterator)
        try:
            for targets in (first, second):
                assert isinstance(targets, np.ndarray)
                assert targets.dtype == np.int64
                assert targets.ndim == 1
                assert targets.flags["C_CONTIGUOUS"]
                assert targets.flags.writeable is False
                assert targets.flags.owndata or targets.base is not None
                assert not np.shares_memory(targets, dataset._targets)
                with pytest.raises(ValueError):
                    targets[0] = 0
                with pytest.raises(ValueError):
                    targets.fill(0)
            assert not np.shares_memory(first, second)
            # The batch-index sequence is the **only** cross-dtype claim,
            # and it carries no dtype at all.
            assert first.tolist() == dataset._targets[
                list(sampler.plan()[0])].tolist()
        finally:
            first_features.close()
            second_features.close()
        iterator.close()
        # No native integer tensor exists, is needed, or is implied.
        assert "int64" not in cpp.SUPPORTED_DTYPES
        with pytest.raises((ValueError, TypeError)):
            NativeTensor.from_array(np.arange(4), dtype="int64")
    finally:
        loader.close()
        dataset.close()
    assert settled(live_storages) == baseline


# ===========================================================================
# 17. Globals, registries, and the filesystem (§12.7)
# ===========================================================================

@needs_backend
def test_an_accepted_operation_moves_no_global_or_registry(live_storages,
                                                           tmp_path):
    """The *accepted* half of §12.7's last clause: ordinary, successful
    use — a whole epoch, a state round trip, a restoration — touches no
    global RNG, no registry, no environment variable, no working
    directory, and no file.

    Deliberately not a claim of complete process purity: it names the
    globals the design says the phase does not use.
    """
    baseline = settled(live_storages)
    loader, sampler, dataset = hardened_pipeline()
    try:
        before = globals_view(directory=tmp_path)
        for features, targets in loader:
            features.close()
        state = loader.state_dict()
        assert loader.load_state_dict(state) is None
        assert sampler.load_state_dict(state["sampler"]) is None
        assert sampler.plan() and sampler.epoch_permutation()
        assert repr(loader) and repr(sampler) and repr(dataset)
        for features, targets in loader:
            features.close()
        assert globals_view(directory=tmp_path) == before
        assert settled(live_storages) == baseline
    finally:
        loader.close()
        dataset.close()
    assert globals_view(directory=tmp_path) == before
    assert settled(live_storages) == baseline


@needs_backend
def test_the_pipeline_consults_no_random_source_of_its_own():
    """§8/§4.5: the order is a pure function of ``(seed, epoch, length)``,
    so no module here reads Python's ``random``, NumPy's global RNG, a
    clock, the environment, or a ``NativeGenerator``."""
    for module in PIPELINE_MODULES:
        names = code_identifiers(f"src/tensorforge/experimental/{module}")
        for forbidden in ("random", "secrets", "time", "monotonic",
                          "perf_counter", "getenv", "environ", "os",
                          "NativeGenerator", "_reserve_call", "uuid",
                          "getpid", "urandom"):
            assert forbidden not in names, (module, forbidden)
    # And the behavioral half: seeding the two global RNGs differently
    # cannot change a plan.
    loader, sampler, dataset = hardened_pipeline()
    try:
        random.seed(1)
        np.random.seed(1)
        first = (sampler.plan(0), sampler.plan(5),
                 sampler.epoch_permutation(9))
        random.seed(999_999)
        np.random.seed(424_242)
        second = (sampler.plan(0), sampler.plan(5),
                  sampler.epoch_permutation(9))
        assert first == second
    finally:
        loader.close()
        dataset.close()


# ===========================================================================
# 18. J7's own non-goals — what this milestone must not have added
# ===========================================================================

def test_j7_added_no_public_name_module_example_or_benchmark():
    """J7 is **evidence**. Its whole diff is this module, narrow status
    edits to existing tests, and documentation."""
    assert len(experimental.__all__) == 25
    assert len(set(experimental.__all__)) == 25
    for invented in ("NativeDataHardening", "NativeBatchIterator",
                     "NativeFaultInjector", "NativeTransactionInspector",
                     "live_storage_count", "native_live_storages",
                     "arm_batch_failure", "deliver_batch"):
        assert not hasattr(experimental, invented), invented
        assert invented not in experimental.__all__, invented
    # No new production module under the package.
    landed = {"native_dataset.py", "native_sampler.py",
              "native_data_loader.py", "_native_permutation.py"}
    present = {path.name for path in PACKAGE.glob("*.py")
               if "data" in path.name or "sampler" in path.name
               or "permutation" in path.name}
    assert present == landed, sorted(present)
    for absent in ("native_data_hardening.py", "native_data_benchmark.py",
                   "native_fault_injection.py", "native_data_workers.py"):
        assert not (PACKAGE / absent).exists(), absent
    # The benchmark and its contract module are **J8's**, not J7's: they
    # are named here rather than merely counted, so this check keeps
    # stating which milestone contributed which artifact. J9's closure
    # module has not started.
    assert (REPO_ROOT / "benchmarks"
            / "benchmark_native_data_pipeline.py").exists()
    assert (REPO_ROOT / "tests" / "test_native_data_benchmark.py").exists()
    assert not (REPO_ROOT / "tests"
                / "test_native_phase_j_closure.py").exists()
    examples = [path.name for path in (REPO_ROOT / "examples").glob("*.py")
                if path.name != "__init__.py"]
    benchmarks = [path.name for path in (REPO_ROOT / "benchmarks").glob("*.py")
                  if path.name != "__init__.py"]
    # 16 examples since J6; 8 benchmarks when J7 landed, and 9 since J8
    # added exactly one. J7's own delta to both is still zero.
    assert len(examples) == 16, sorted(examples)
    assert len(benchmarks) == 9, sorted(benchmarks)
    assert "benchmark_native_data_pipeline.py" in benchmarks


def test_j7_moved_no_capability_schema_or_version():
    """Every locked row, re-read from the live registries rather than from
    a copy."""
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    info = cpp.backend_info()
    assert info["dtype"] == "float64"
    assert info["device"] == "cpu"
    assert info["stable_framework_integration"] is False
    assert checkpoint_module._FORMAT == "tensorforge.native_checkpoint"
    assert checkpoint_module._FORMAT_VERSION == 3
    assert checkpoint_module._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert optimizer_state.FORMAT_VERSION == 1
    assert loader_module._FORMAT == "tensorforge.native_data_loader"
    assert loader_module._FORMAT_VERSION == 1
    assert loader_module._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert loader_module._STATE_FIELDS == ("format", "format_version",
                                           "sampler")
    assert sampler_module._FORMAT == "tensorforge.native_sampler"
    assert sampler_module._FORMAT_VERSION == 1
    assert sampler_module._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert sampler_module._IDENTITY_DTYPES == ("float64", "float32")
    # No version 4 anywhere, and no second accepted loader/sampler version.
    for module in (loader_module, sampler_module):
        assert 2 not in module._SUPPORTED_FORMAT_VERSIONS
    assert 4 not in checkpoint_module._SUPPORTED_FORMAT_VERSIONS


def test_j7_added_no_public_hook_inspector_or_counter():
    """The seams this module uses are the ones that already existed, and
    every one of them is private and unexported."""
    for name in ("_deliver_batch", "_NativeBatchIterator", "_FORMAT",
                 "_FORMAT_VERSION", "_STATE_FIELDS"):
        assert hasattr(loader_module, name), name
        assert name not in experimental.__all__, name
        assert not hasattr(experimental, name), name
    for name in ("_claim_batch", "_publish_pending", "_commit_pending",
                 "_rollback_pending", "_complete_pending", "_assign_state",
                 "_validate_state", "_begin_iteration", "_end_iteration"):
        assert hasattr(NativeBatchSampler, name), name
        assert not name.lstrip("_") in dir(NativeBatchSampler), name
    # No public counterpart to any of them arrived.
    for owner in (NativeTensorDataset, NativeBatchSampler, NativeDataLoader):
        for forbidden in ("transaction", "pending", "claim", "serial",
                          "live_storage", "storage_count", "on_deliver",
                          "delivery_hook", "set_hook", "inject_failure",
                          "arm_failure", "fail_next", "debug"):
            assert not hasattr(owner, forbidden), (owner.__name__, forbidden)
    # The backend's one documented fault-injection arm is the one that
    # already existed, and J7 added no second.
    assert hasattr(cpp, "_arm_alloc_failure")
    assert hasattr(cpp, "fault_injection_available")
    for invented in ("_arm_batch_failure", "_arm_gather_failure",
                     "_arm_transfer_failure", "arm_alloc_failure"):
        assert not hasattr(cpp, invented), invented


def test_j7_left_the_native_artifacts_untouched():
    """No C++, no CMake, no CTest, and no ABI change: the counts are read
    from the real files."""
    sources = sorted((REPO_ROOT / "cpp" / "tests").glob("test_*.cpp"))
    assert len(sources) == 24, [path.name for path in sources]
    registered = re.findall(
        r"add_test\s*\(\s*NAME\s+(\w+)",
        (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8"))
    assert len(registered) == 24 == len(set(registered)), registered
    exported = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        exported.update(re.findall(
            r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(", text, re.S))
    assert len(exported) == 54, sorted(exported)
    # Negative control: the export scanner really does find symbols, so
    # "54" is a measurement rather than a dead regex.
    assert "tf_storage_create" in exported or any(
        name.startswith("tf_storage") for name in exported), sorted(exported)
    # No data-pipeline symbol appeared; the phase plans none.
    forbidden = re.compile(r"^tf_(dataset|sampler|loader|batch|shuffle|"
                           r"permut|gather)", re.I)
    for name in sorted(exported):
        assert not forbidden.search(name), name
    # No Phase-J vocabulary reached the native side at all.
    for path in list((REPO_ROOT / "cpp" / "src").rglob("*.cpp")) + \
            list((REPO_ROOT / "cpp" / "include").rglob("*.h")):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("dataset", "sampler", "dataloader", "data_loader",
                          "minibatch", "mini_batch", "tf_gather"):
            assert forbidden not in text.lower(), (path.name, forbidden)


def test_j7_declares_no_dependency_and_measures_no_time():
    """No dependency, no benchmark, and **nothing timed**: read from this
    module's own AST, so the claim is about what actually shipped."""
    tree = ast.parse((REPO_ROOT / "tests"
                      / "test_native_data_hardening.py").read_text(
                          encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("torch", "tensorflow", "jax", "sklearn", "pandas",
                      "matplotlib", "timeit", "time", "datetime",
                      "threading", "multiprocessing", "asyncio",
                      "concurrent"):
        assert forbidden not in imported, forbidden
    # Everything it does import is the standard library, NumPy, pytest, or
    # TensorForge itself — no dependency was added for J7.
    allowed = {"ast", "collections", "gc", "inspect", "json", "os",
               "pathlib", "random", "re", "subprocess", "sys", "types",
               "numpy", "pytest", "tensorforge"}
    assert imported <= allowed, sorted(imported - allowed)
    names = code_identifiers("tests/test_native_data_hardening.py")
    for forbidden in ("perf_counter", "process_time", "monotonic",
                      "benchmark", "elapsed"):
        assert forbidden not in names, forbidden
    # Negative control for the import scanner.
    planted = ast.parse("import torch\nfrom timeit import default_timer\n")
    planted_names = set()
    for node in ast.walk(planted):
        if isinstance(node, ast.Import):
            planted_names |= {alias.name.split(".")[0]
                              for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            planted_names.add(node.module.split(".")[0])
    assert {"torch", "timeit"} <= planted_names
