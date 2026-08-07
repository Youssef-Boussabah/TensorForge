"""Native checkpoint metadata integration (Phase J, milestone J5;
docs/native_data_pipeline_design.md §2.6, §3.4, §3.5, §6, §7.3-§7.8, §9.3,
§9.4, §9.5, §10, §11, §12.4, §12.5, §12.7, §13.1-§13.7, §14.1-§14.5, §15,
§16, §17, §18, §19, §20, §23 J5).

J5 adds **no production code at all**. Everything it proves already exists:
``save_native_checkpoint`` / ``load_native_checkpoint`` carry a validated,
recursively JSON-compatible ``metadata`` mapping, and ``NativeDataLoader``
has ``state_dict()`` / ``load_state_dict(state)``. What was never proved is
that those two halves compose into the one supported workflow — so this
module is the proof, and the checkpoint module is provably unchanged.

The workflow, and the whole of J5's public surface::

    loader_state = loader.state_dict()             # 1. snapshot, no iteration
    save_native_checkpoint(                        # 2. ordinary metadata
        path, model, optimizer=optimizer,
        metadata={"training": {"next_step": step + 1,
                               "data_loader": loader_state}})
    ...
    metadata = load_native_checkpoint(             # 3. checkpoint first
        path, fresh_model, optimizer=fresh_optimizer)
    fresh_loader.load_state_dict(                  # 4. loader second
        metadata["training"]["data_loader"])
    next_step = metadata["training"]["next_step"]

What this module proves:

* **§13.1 the format does not move** — version **3**, accepted
  ``(1, 2, 3)``, the same six manifest root keys, the same array
  inventory. No loader field, no loader array, no permutation, no dataset
  payload. Loader state exists **only** below ``metadata``.
* **§13.3 the keys are the caller's, not a schema** — ``"training"``,
  ``"data_loader"``, and ``"next_step"`` are conventions this repository
  speaks consistently and **no runtime code knows**. Alternate nesting,
  alternate names, and two loaders' states side by side all round-trip
  unchanged, because the checkpoint sees ordinary metadata.
* **§13.5 the ordering, and the honest atomicity boundary** — the loader
  snapshot precedes the save with no iteration in between; the checkpoint
  load precedes the loader load; a failed checkpoint load leaves the
  loader untouched; and a loader load that fails **after** a successful
  checkpoint load rolls **nothing** back, because there is no cross-object
  transaction and none is claimed.
* **§13.7 the delivery boundaries** — a save after a *failed* delivery
  resumes the **same candidate** batch, a save after a *successful* one
  resumes the **following** batch, and a save at an epoch boundary resumes
  the canonical ``(epoch + 1, 0)`` next epoch. Checkpoint metadata cannot
  capture a skipped-but-undelivered position.
* **§14 the exact restoration** — entirely fresh model, optimizer,
  generators, dataset, sampler, and loader, every one deliberately built
  *wrong* first, restored from a real version-3 ``.npz`` and compared with
  **no tolerance anywhere**: raw IEEE-754 bits through ``uint32``/
  ``uint64`` views, exact ``int64`` targets, exact generator alias
  topology, exact indices, and the exact remaining sequence.
* **§13.6 the non-coupling** — asserted by AST source inspection in both
  directions and by driving a real save and load with the loader's two
  state methods patched to record any call: neither fires.

**Not proved here, because it does not exist:** the J6 training example,
the J7 hardening matrix, and the J8 benchmark. J5 is a checkpoint
*integration* proof; it adds no example, no public ``train``, and no
benchmark, and §18 below asserts their absence.

No test here asserts an exact error message, a dict ordering, a timing, a
GC event, or a speed.

Selector: python -m pytest -q tests/test_native_data_checkpoint.py
"""

import ast
import gc
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

import tensorforge.experimental as experimental
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeBatchSampler,
    NativeCrossEntropyLoss,
    NativeDataLoader,
    NativeDropout,
    NativeGenerator,
    NativeLinear,
    NativeModule,
    NativeReLU,
    NativeTensorDataset,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint as checkpoint_module
from tensorforge.experimental import native_data_loader as loader_module

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "src" / "tensorforge" / "experimental"

CHECKPOINT_SOURCE = "src/tensorforge/experimental/native_checkpoint.py"
PIPELINE_SOURCES = (
    "src/tensorforge/experimental/native_dataset.py",
    "src/tensorforge/experimental/native_sampler.py",
    "src/tensorforge/experimental/native_data_loader.py",
    "src/tensorforge/experimental/_native_permutation.py",
)

CHECKPOINT_FORMAT = "tensorforge.native_checkpoint"
CHECKPOINT_VERSION = 3
CHECKPOINT_VERSIONS = (1, 2, 3)
MANIFEST_ROOT_KEYS = {"format", "format_version", "model", "optimizer",
                      "generators", "metadata"}
LOADER_FORMAT = "tensorforge.native_data_loader"
SAMPLER_FORMAT = "tensorforge.native_sampler"

# The model's geometry. Small on purpose: every proof here is about exact
# equality and ordering, never about size.
FEATURES = 4
HIDDEN = 6
CLASSES = 3
SAMPLES = 12
BATCH = 3                      # -> 4 batches per epoch at drop_last=False
# The exit gate uses a size that does **not** divide the sample count, so
# ``drop_last`` genuinely changes the epoch length (3 batches vs 2) and the
# parametrization is not two spellings of one case.
RESUME_BATCH = 5

needs_backend = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


# ===========================================================================
# Fixtures and helpers
# ===========================================================================

@pytest.fixture(autouse=True)
def _disarm_allocation_faults():
    """No injected allocation failure survives a test, whatever it did."""
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every ``NativeStorage`` currently open — the project's
    deterministic instrumentation for native-allocation lifetime. There is
    no public counter, and J5 adds none."""
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


def settled(live_storages):
    """The live-storage count after a collection. Collection *settles* the
    count; it is never the proof that anything was released — every test
    here closes what it owns explicitly first."""
    gc.collect()
    return len(live_storages)


def bit_view(values):
    """Raw IEEE-754 bits of a host array, as unsigned integers of the
    matching width. **Never a tolerance**: float32 through ``uint32``,
    float64 through ``uint64``, each compared only against itself."""
    if values.dtype == np.float32:
        return values.view(np.uint32).tolist()
    assert values.dtype == np.float64, values.dtype
    return values.view(np.uint64).tolist()


def host_arrays(samples=SAMPLES, features=FEATURES, classes=CLASSES):
    """Deterministic source values. Every row is identifiable, so a batch
    can be checked against the indices that produced it."""
    values = (np.arange(samples * features, dtype=np.float64)
              .reshape(samples, features) % 7.0) - 3.0
    targets = np.arange(samples, dtype=np.int64) % classes
    return values, targets


def make_dataset(dtype=None, samples=SAMPLES, offset=0.0):
    """A dataset built from freshly constructed host arrays. Two calls with
    the same arguments give two **different objects** with the same four
    §6.4 identity fields — which is exactly what a restored graph needs."""
    values, targets = host_arrays(samples)
    return NativeTensorDataset(values + offset, targets, dtype=dtype)


class ResumeModel(NativeModule):
    """Trainable parameters, persistent buffers, and a **shared** generator
    alias topology — the smallest graph in which every restored family is
    nontrivial and none of them can be recovered from the others.

    ``drop_a`` and ``drop_b`` sit on **one** ``NativeGenerator``; ``drop_c``
    owns its own. That sharing is not recoverable from the states alone, so
    it is the topology J5's restoration proof has to preserve.
    """

    def __init__(self, *, dtype=None, in_seed=1, out_seed=2,
                 shared_seed=101, own_seed=202, hidden=HIDDEN):
        super().__init__()
        self.linear_in = NativeLinear(FEATURES, hidden, seed=in_seed,
                                      dtype=dtype)
        self.norm = NativeBatchNorm1d(hidden, dtype=dtype)
        self.relu = NativeReLU()
        shared = NativeGenerator(shared_seed)
        self.drop_a = NativeDropout(0.25, generator=shared)
        self.drop_b = NativeDropout(0.25, generator=shared)
        self.drop_c = NativeDropout(0.5, seed=own_seed)
        self.linear_out = NativeLinear(hidden, CLASSES, seed=out_seed,
                                       dtype=dtype)

    def forward(self, x):
        hidden = self.relu(self.norm(self.linear_in(x)))
        hidden = self.drop_c(self.drop_b(self.drop_a(hidden)))
        return self.linear_out(hidden)


class Graph:
    """One complete object graph: model, optimizer, dataset, sampler, and
    loader. Held together only so a test can build two of them and close
    both explicitly — it is a test convenience and no production analogue
    exists or is implied."""

    __slots__ = ("model", "optimizer", "dataset", "sampler", "loader")

    def __init__(self, model, optimizer, dataset, sampler, loader):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.sampler = sampler
        self.loader = loader


def build_graph(dtype=None, *, in_seed=1, out_seed=2, shared_seed=101,
                own_seed=202, lr=0.01, batch_size=BATCH, shuffle=True,
                seed=7, drop_last=False, samples=SAMPLES, dataset=None,
                hidden=HIDDEN):
    dataset = make_dataset(dtype, samples) if dataset is None else dataset
    sampler = NativeBatchSampler(dataset, batch_size=batch_size,
                                 shuffle=shuffle, seed=seed,
                                 drop_last=drop_last)
    loader = NativeDataLoader(sampler)
    model = ResumeModel(dtype=dtype, in_seed=in_seed, out_seed=out_seed,
                        shared_seed=shared_seed, own_seed=own_seed,
                        hidden=hidden)
    optimizer = NativeAdam(model.parameters(), lr=lr)
    return Graph(model, optimizer, dataset, sampler, loader)


def close_graph(graph):
    """Explicit cleanup, in the §15.4 order: the loader (and so its
    iterator) first, then the optimizer's moments, then the dataset's host
    snapshots, then every unique parameter and persistent buffer. Nothing
    here relies on garbage collection."""
    graph.loader.close()
    graph.optimizer.close()
    graph.dataset.close()
    seen = set()
    for _, tensor in list(graph.model.named_parameters()) + \
            list(graph.model.named_buffers()):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            tensor.close()


def train_steps(graph, iterator, steps):
    """``steps`` genuine training steps taken from ``iterator``.

    Every step moves parameters, the batch-norm running buffers, the Adam
    moments and step counters, all three registered generator streams, and
    the committed loader position — so a save taken afterwards is
    nontrivial in every family at once. Every delivered feature batch is
    closed by this loop, because it is **the caller's** (§10.5).
    """
    loss_fn = NativeCrossEntropyLoss()
    losses = []
    for _ in range(steps):
        features, targets = next(iterator)
        try:
            logits = graph.model(features)
            loss = loss_fn(logits, targets)
            losses.append(loss.to_numpy().copy())
            loss.backward()
            graph.optimizer.step()
            graph.optimizer.zero_grad()
            loss.close()
            logits.close()
        finally:
            features.close()
    return losses


def generator_topology(model):
    """The alias topology as one comparable value: which dropout modules
    share a generator, by position in the model's deduplicated generator
    order. ``drop_a`` and ``drop_b`` must land on the same index and
    ``drop_c`` on a different one — a fact no per-generator state carries.
    """
    order = {id(generator): index
             for index, generator in enumerate(model.generators())}
    return tuple((path, order[id(getattr(model, path).generator)])
                 for path in ("drop_a", "drop_b", "drop_c"))


def graph_fingerprint(graph):
    """Everything a restoration must reproduce **exactly**, as one
    comparable value. Raw IEEE-754 bit patterns throughout; no tolerance,
    no ``allclose``, no rounding, and nothing process-local."""
    model, optimizer = graph.model, graph.optimizer
    snapshot = model.state_dict()
    try:
        keys = list(snapshot)
        tensors = {
            name: (tensor.dtype, tuple(tensor.shape), tensor.device,
                   tuple(bit_view(tensor.to_numpy())))
            for name, tensor in snapshot.items()
        }
    finally:
        for tensor in snapshot.values():
            tensor.close()
    optimizer_state = optimizer.state_dict()
    try:
        moments = {
            label: tuple(
                (entry.dtype, tuple(entry.shape), entry.device,
                 tuple(bit_view(entry.to_numpy())))
                for entry in optimizer_state[label]
            )
            for label in ("m", "v")
        }
        parameters = tuple(
            (tuple(entry["shape"]), entry["dtype"], entry["device"])
            for entry in optimizer_state["parameters"]
        )
        state_format_version = optimizer_state["format_version"]
        optimizer_name = optimizer_state["optimizer"]
    finally:
        for label in ("m", "v"):
            for entry in optimizer_state[label]:
                entry.close()
    return {
        "keys": keys,
        "tensors": tensors,
        "generators": model.generator_state_dict(),
        "generator_names": [name for name, _ in model.named_generators()],
        "generator_topology": generator_topology(model),
        "optimizer_type": type(optimizer).__name__,
        "optimizer_name": optimizer_name,
        "state_format_version": state_format_version,
        "lr": optimizer.lr,
        "betas": tuple(optimizer.betas),
        "eps": optimizer.eps,
        "step_counts": tuple(optimizer.step_counts),
        "parameters": parameters,
        "moments": moments,
    }


def loader_fingerprint(loader):
    """Every publicly observable fact about a loader, its sampler, and its
    dataset — plus the private bookkeeping a rejected operation must leave
    untouched. Used before and after everything that must change nothing.
    """
    sampler = loader.sampler
    dataset = loader.dataset
    return (
        id(loader), id(sampler), id(dataset),
        loader.closed, dataset.closed,
        sampler.seed, sampler.shuffle, sampler.batch_size, sampler.drop_last,
        sampler.epoch, sampler.cursor,
        sampler.batches_per_epoch, sampler.remaining,
        sampler.next_batch_indices(),
        sampler.epoch_permutation(),
        sampler.plan(),
        json.dumps(loader.state_dict(), sort_keys=True),
        json.dumps(sampler.state_dict(), sort_keys=True),
        loader._iterator is None,
        sampler._transaction is None,
        frozenset(sampler._active_iterations),
    )


def training_metadata(next_step, loader_state):
    """The **recommended** caller convention of §13.2 — and nothing more.
    No production constant spells any of these three names."""
    return {"training": {"next_step": next_step,
                         "data_loader": loader_state}}


def drain(loader):
    """Every batch of one iteration, closing each feature tensor exactly as
    a caller must."""
    for features, targets in loader:
        yield features, targets
        features.close()


def batch_record(features, targets):
    """One delivered batch reduced to exactly what §14.3 compares."""
    host = features.to_numpy()
    return {
        "dtype": features.dtype,
        "shape": tuple(features.shape),
        "device": features.device,
        "contiguous": features.contiguous,
        "bits": tuple(bit_view(host)),
        "target_dtype": targets.dtype,
        "target_shape": targets.shape,
        "targets": targets.tolist(),
        "target_flags": (bool(targets.flags["C_CONTIGUOUS"]),
                         bool(targets.flags["OWNDATA"]),
                         bool(targets.flags["WRITEABLE"])),
    }


def manifest_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def array_names_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return sorted(archive.files)


def code_identifiers(relative):
    """Every identifier a module's **executable code** names.

    A source-text scan would be wrong here: these modules explain at length
    what they deliberately do *not* do, so a prose mention of
    ``NativeDataLoader`` inside ``native_checkpoint``'s docstring would
    fail a substring check that is supposed to be about behavior. Reading
    the AST asks the question that was meant.
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
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


def record_loader_state_calls(monkeypatch):
    """Patch the loader's two state methods to record every call, and
    return the log. **No production hook exists or is added**: this patches
    the class from the test, exactly as J4's non-coupling proof does."""
    calls = []
    original_state_dict = NativeDataLoader.state_dict
    original_load = NativeDataLoader.load_state_dict

    def spy_state_dict(self):
        calls.append("state_dict")
        return original_state_dict(self)

    def spy_load(self, state):
        calls.append("load_state_dict")
        return original_load(self, state)

    monkeypatch.setattr(NativeDataLoader, "state_dict", spy_state_dict)
    monkeypatch.setattr(NativeDataLoader, "load_state_dict", spy_load)
    return calls


def refuses_state_dict(loader):
    """Whether ``loader.state_dict()`` refuses right now (§9.5)."""
    try:
        loader.state_dict()
    except RuntimeError:
        return True
    return False


# ===========================================================================
# 0. Negative controls for the scanners this module relies on
# ===========================================================================

def test_the_identifier_scanner_can_actually_find_something():
    """A source scan that finds nothing must be shown able to find
    something, or every absence below is a dead regex."""
    checkpoint = code_identifiers(CHECKPOINT_SOURCE)
    assert "save_native_checkpoint" in checkpoint
    assert "load_native_checkpoint" in checkpoint
    assert "_validated_metadata" in checkpoint
    loader = code_identifiers(PIPELINE_SOURCES[2])
    assert "NativeDataLoader" in loader
    assert "state_dict" in loader
    assert "load_state_dict" in loader


def test_the_bit_view_helper_really_distinguishes_values():
    """The comparison every exact proof below runs through. It must
    separate values that differ in one bit, and it must not use a
    tolerance."""
    left = np.array([1.0, 2.0], dtype=np.float64)
    right = np.array([1.0, np.nextafter(2.0, 3.0)], dtype=np.float64)
    assert bit_view(left) != bit_view(right)
    assert bit_view(left) == bit_view(left.copy())
    narrow = np.array([1.0, 2.0], dtype=np.float32)
    assert bit_view(narrow) == bit_view(narrow.copy())
    assert bit_view(np.array([-0.0])) != bit_view(np.array([0.0]))


# ===========================================================================
# 1. The supported caller workflow — J5's exit gate
# ===========================================================================

def _archive_resume_proof(tmp_path, live_storages, dtype, shuffle, drop_last,
                          completed_steps):
    """Save a real version-3 archive mid-epoch, restore into an entirely
    fresh object graph, and prove the continuation is exact.

    Returns ``(expected_indices, tail_length, next_position)`` so a caller
    can assert the interruption really was mid-epoch and non-vacuous.
    """
    baseline = settled(live_storages)
    path = str(tmp_path / f"resume_{dtype}_{shuffle}_{drop_last}.npz")

    # --- the saving graph -------------------------------------------------
    source = build_graph(dtype, batch_size=RESUME_BATCH, shuffle=shuffle,
                         drop_last=drop_last, lr=0.01)
    batches_per_epoch = source.sampler.batches_per_epoch
    assert batches_per_epoch == (2 if drop_last else 3)
    # Non-vacuous by construction: the interruption is genuinely mid-epoch,
    # exactly as I9's SPLIT_STEP discipline requires.
    assert completed_steps % batches_per_epoch != 0
    assert 0 < completed_steps < batches_per_epoch

    iterator = iter(source.loader)
    train_steps(source, iterator, completed_steps)
    iterator.close()

    # --- 1. the snapshot, then the save. No iteration in between. --------
    loader_state = source.loader.state_dict()
    next_step = completed_steps
    expected_indices = source.sampler.next_batch_indices()
    saved_fingerprint = graph_fingerprint(source)
    source_identity = source.dataset.identity()

    save_native_checkpoint(
        path, source.model, optimizer=source.optimizer,
        metadata=training_metadata(next_step, loader_state),
    )
    assert os.path.isfile(path)
    # The save moved nothing: the loader still describes the same next
    # batch, and the model/optimizer/generator state is untouched.
    assert source.loader.state_dict() == loader_state
    assert source.sampler.next_batch_indices() == expected_indices
    assert graph_fingerprint(source) == saved_fingerprint

    # --- the oracle: the source's own remaining tail, index by index -----
    remaining = source.sampler.remaining
    tail = tail_of(source, remaining)
    boundary_state = source.loader.state_dict()
    next_position = (boundary_state["sampler"]["epoch"],
                     boundary_state["sampler"]["cursor"])
    following_plan = source.sampler.plan()
    following = tail_of(source, source.sampler.remaining)
    close_graph(source)

    # --- the restored graph: entirely fresh, deliberately built wrong ----
    restored = build_graph(dtype, in_seed=91, out_seed=92, shared_seed=93,
                           own_seed=94, lr=0.05, batch_size=4,
                           shuffle=not shuffle, seed=1234, drop_last=False)
    # Separately constructed from logically identical source values: equal
    # samples, feature shape, feature dtype, and fingerprint — a different
    # object with the same §6.4 identity.
    assert restored.dataset.identity() == source_identity
    assert restored.dataset.feature_shape == (FEATURES,)
    assert restored.dataset.dtype == (dtype or "float64")
    # Give it a genuinely different position too, through the authoritative
    # public state-loading route rather than by poking private fields.
    warm = restored.loader.state_dict()
    warm["sampler"]["epoch"], warm["sampler"]["cursor"] = 5, 1
    restored.loader.load_state_dict(warm)

    fresh_fingerprint = graph_fingerprint(restored)
    assert fresh_fingerprint != saved_fingerprint
    assert restored.loader.state_dict() != loader_state
    assert restored.sampler.next_batch_indices() != expected_indices

    generator_ids = [id(g) for g in restored.model.generators()]
    parameter_ids = [id(p) for _, p in restored.model.named_parameters()]
    buffer_ids = [id(b) for _, b in restored.model.named_buffers()]
    loader_before = restored.loader.state_dict()

    # --- 2. checkpoint load first ----------------------------------------
    metadata = load_native_checkpoint(path, restored.model,
                                      optimizer=restored.optimizer)
    assert graph_fingerprint(restored) == saved_fingerprint
    # No generator, parameter, or buffer object was constructed by the load.
    assert [id(g) for g in restored.model.generators()] == generator_ids
    assert [id(p) for _, p in restored.model.named_parameters()] \
        == parameter_ids
    assert [id(b) for _, b in restored.model.named_buffers()] == buffer_ids
    # The loader has not been touched yet — that is the caller's next line.
    assert restored.loader.state_dict() == loader_before
    assert restored.loader.state_dict() != loader_state

    # --- 3. inspect the metadata, 4. load the loader ----------------------
    assert metadata["training"]["next_step"] == next_step
    assert type(metadata["training"]["next_step"]) is int
    assert metadata["training"]["data_loader"] == loader_state
    assert restored.loader.load_state_dict(
        metadata["training"]["data_loader"]) is None

    # --- the exact continuation ------------------------------------------
    assert restored.loader.state_dict() == loader_state
    assert restored.sampler.next_batch_indices() == expected_indices
    assert restored.sampler.remaining == remaining
    # Identity is preserved absolutely: the loader kept its own sampler and
    # its own dataset, and adopted only the six configuration values.
    assert restored.loader.sampler is restored.sampler
    assert restored.loader.dataset is restored.dataset
    assert restored.sampler.batch_size == RESUME_BATCH
    assert restored.sampler.shuffle is shuffle
    assert restored.sampler.drop_last is drop_last
    assert restored.sampler.seed == 7

    assert tail_of(restored, remaining) == tail
    assert restored.loader.state_dict() == boundary_state
    assert restored.sampler.plan() == following_plan
    assert tail_of(restored, restored.sampler.remaining) == following

    close_graph(restored)
    assert settled(live_storages) == baseline
    return expected_indices, len(tail), next_position


def tail_of(graph, count):
    """``count`` batches from one fresh iteration, each recorded as
    ``(planned indices, delivered batch)`` — the exact pair §14.3's first
    two rows compare. Every feature tensor is closed here, because it is
    the caller's."""
    records = []
    iterator = iter(graph.loader)
    for _ in range(count):
        indices = graph.sampler.next_batch_indices()
        features, targets = next(iterator)
        try:
            records.append((indices, batch_record(features, targets)))
        finally:
            features.close()
    with pytest.raises(StopIteration):
        next(iterator)
    iterator.close()
    return records


@needs_backend
@pytest.mark.parametrize("dtype", [None, "float64", "float32"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_a_real_archive_restores_the_exact_remaining_sequence(
        tmp_path, live_storages, dtype, shuffle, drop_last):
    """**The J5 exit gate.** A genuine mid-epoch interruption, carried
    through a real version-3 ``.npz`` as ordinary caller metadata and
    restored into an entirely fresh model, optimizer, generator,
    dataset, sampler, and loader, reproduces the exact next batch, the
    exact remaining sequence, the canonical epoch boundary, and the whole
    following epoch — with no tolerance anywhere."""
    indices, tail_length, next_position = _archive_resume_proof(
        tmp_path, live_storages, dtype, shuffle, drop_last, completed_steps=1)
    assert len(indices) >= 1
    assert tail_length >= 1
    assert next_position == (1, 0)


@needs_backend
def test_the_archive_resume_proof_is_not_vacuous(tmp_path, live_storages):
    """The negative control: **omitting** the loader restoration must make
    the continuation differ, so the exit gate above cannot pass by
    accident."""
    baseline = settled(live_storages)
    path = str(tmp_path / "vacuity.npz")

    source = build_graph(batch_size=RESUME_BATCH, shuffle=True)
    iterator = iter(source.loader)
    train_steps(source, iterator, 1)
    iterator.close()
    loader_state = source.loader.state_dict()
    expected = source.sampler.next_batch_indices()
    expected_tail = tuple(source.sampler.plan()[source.sampler.cursor:])
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(1, loader_state))
    close_graph(source)

    restored = build_graph(in_seed=91, out_seed=92, batch_size=4,
                           shuffle=False, seed=1234)
    metadata = load_native_checkpoint(path, restored.model,
                                      optimizer=restored.optimizer)
    # The checkpoint load restored the model — and touched the loader not
    # at all, which is exactly why the loader must be loaded explicitly.
    assert metadata["training"]["data_loader"] == loader_state
    # Without the caller's second line, every observable continuation
    # differs: the next batch, the whole remaining index sequence, and the
    # state itself.
    assert restored.sampler.next_batch_indices() != expected
    assert tuple(restored.sampler.plan()[restored.sampler.cursor:]) \
        != expected_tail
    assert restored.loader.state_dict() != loader_state
    # ...and applying it makes all three agree, so the control really is
    # controlling for the restoration and nothing else.
    restored.loader.load_state_dict(metadata["training"]["data_loader"])
    assert restored.sampler.next_batch_indices() == expected
    assert tuple(restored.sampler.plan()[restored.sampler.cursor:]) \
        == expected_tail
    assert restored.loader.state_dict() == loader_state
    close_graph(restored)
    assert settled(live_storages) == baseline


# ===========================================================================
# 2. Fresh objects, and the identity separation §14.2 requires
# ===========================================================================

@needs_backend
def test_no_object_from_the_saving_graph_survives_into_the_restored_one(
        tmp_path, live_storages):
    baseline = settled(live_storages)
    path = str(tmp_path / "fresh.npz")

    source = build_graph(shuffle=True)
    iterator = iter(source.loader)
    train_steps(source, iterator, 1)
    iterator.close()
    loader_state = source.loader.state_dict()
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(1, loader_state))

    source_ids = {
        "model": id(source.model),
        "optimizer": id(source.optimizer),
        "dataset": id(source.dataset),
        "sampler": id(source.sampler),
        "loader": id(source.loader),
        "parameters": {id(p) for _, p in source.model.named_parameters()},
        "buffers": {id(b) for _, b in source.model.named_buffers()},
        "generators": {id(g) for g in source.model.generators()},
    }
    fingerprint = graph_fingerprint(source)
    close_graph(source)

    restored = build_graph(in_seed=91, out_seed=92, shared_seed=93,
                           own_seed=94, lr=0.05, batch_size=BATCH + 2,
                           shuffle=False, seed=4242)
    assert id(restored.model) != source_ids["model"]
    assert id(restored.optimizer) != source_ids["optimizer"]
    assert id(restored.dataset) != source_ids["dataset"]
    assert id(restored.sampler) != source_ids["sampler"]
    assert id(restored.loader) != source_ids["loader"]
    assert not ({id(p) for _, p in restored.model.named_parameters()}
                & source_ids["parameters"])
    assert not ({id(b) for _, b in restored.model.named_buffers()}
                & source_ids["buffers"])
    assert not ({id(g) for g in restored.model.generators()}
                & source_ids["generators"])
    # Equal content, different object — the §6 compatibility question.
    assert restored.dataset.identity() == {
        "samples": SAMPLES, "feature_shape": [FEATURES],
        "feature_dtype": "float64",
        "fingerprint": restored.dataset.fingerprint,
    }

    metadata = load_native_checkpoint(path, restored.model,
                                      optimizer=restored.optimizer)
    restored.loader.load_state_dict(metadata["training"]["data_loader"])
    assert graph_fingerprint(restored) == fingerprint
    # ...and the restored graph still owns its own objects afterwards.
    assert restored.loader.sampler is restored.sampler
    assert restored.sampler.dataset is restored.dataset
    assert restored.loader.dataset is restored.dataset
    close_graph(restored)
    assert settled(live_storages) == baseline


@needs_backend
def test_the_generator_alias_topology_survives_and_is_nontrivial(
        tmp_path, live_storages):
    """Two dropouts on one generator and one on its own: a topology no
    per-generator state carries, so it can only be preserved by restoring
    in place into the live objects."""
    baseline = settled(live_storages)
    path = str(tmp_path / "topology.npz")

    source = build_graph()
    assert source.model.drop_a.generator is source.model.drop_b.generator
    assert source.model.drop_a.generator is not source.model.drop_c.generator
    assert len(list(source.model.generators())) == 2
    iterator = iter(source.loader)
    train_steps(source, iterator, 2)
    iterator.close()
    states = source.model.generator_state_dict()
    names = [name for name, _ in source.model.named_generators()]
    topology = generator_topology(source.model)
    # Nontrivial: the shared stream really advanced, and the two streams
    # disagree, so a restoration that swapped them would be visible.
    assert all(state["calls"] > 0 for state in states.values())
    assert len({state["calls"] for state in states.values()}) > 1
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(
                               2, source.loader.state_dict()))
    close_graph(source)

    restored = build_graph(shared_seed=93, own_seed=94)
    assert restored.model.generator_state_dict() != states
    before_ids = [id(g) for g in restored.model.generators()]
    load_native_checkpoint(path, restored.model, optimizer=restored.optimizer)
    assert restored.model.generator_state_dict() == states
    assert [name for name, _ in restored.model.named_generators()] == names
    assert generator_topology(restored.model) == topology
    assert restored.model.drop_a.generator is restored.model.drop_b.generator
    assert (restored.model.drop_a.generator
            is not restored.model.drop_c.generator)
    # Restored **in place**: no generator object was constructed by the load.
    assert [id(g) for g in restored.model.generators()] == before_ids
    close_graph(restored)
    assert settled(live_storages) == baseline


# ===========================================================================
# 3. Model, optimizer, and generator restoration, exactly
# ===========================================================================

@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_every_restored_family_matches_bit_for_bit(tmp_path, live_storages,
                                                   dtype):
    """§14.3, row by row, at each dtype **compared only against itself**."""
    baseline = settled(live_storages)
    path = str(tmp_path / f"families_{dtype}.npz")

    source = build_graph(dtype, lr=0.01)
    iterator = iter(source.loader)
    train_steps(source, iterator, 3)
    iterator.close()
    fingerprint = graph_fingerprint(source)
    # Nontrivial in every family before the save.
    assert all(count > 0 for count in fingerprint["step_counts"])
    assert any(any(bits) for *_, bits in fingerprint["tensors"].values())
    assert fingerprint["moments"]["m"] and fingerprint["moments"]["v"]
    assert "norm.running_mean" in fingerprint["keys"]
    assert "norm.running_var" in fingerprint["keys"]
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(
                               3, source.loader.state_dict()))
    close_graph(source)

    restored = build_graph(dtype, in_seed=91, out_seed=92, shared_seed=93,
                           own_seed=94, lr=0.05)
    iterator = iter(restored.loader)
    train_steps(restored, iterator, 1)
    iterator.close()
    stale = graph_fingerprint(restored)
    assert stale != fingerprint
    assert stale["lr"] != fingerprint["lr"]
    assert stale["step_counts"] != fingerprint["step_counts"]

    load_native_checkpoint(path, restored.model, optimizer=restored.optimizer)
    current = graph_fingerprint(restored)
    # Compared field by field, so a failure names the family that moved.
    assert current["keys"] == fingerprint["keys"]
    assert current["tensors"] == fingerprint["tensors"]
    assert current["generators"] == fingerprint["generators"]
    assert current["generator_names"] == fingerprint["generator_names"]
    assert current["generator_topology"] == fingerprint["generator_topology"]
    assert current["optimizer_type"] == fingerprint["optimizer_type"] \
        == "NativeAdam"
    assert current["state_format_version"] == 1
    assert current["lr"] == fingerprint["lr"]
    assert current["betas"] == fingerprint["betas"]
    assert current["eps"] == fingerprint["eps"]
    assert current["step_counts"] == fingerprint["step_counts"]
    assert current["parameters"] == fingerprint["parameters"]
    assert current["moments"] == fingerprint["moments"]
    # Every dtype and device travelled unchanged; nothing was cast.
    for name, (entry_dtype, _, device, _) in current["tensors"].items():
        assert entry_dtype in ("float64", "float32"), name
        assert device == "cpu", name
    close_graph(restored)
    assert settled(live_storages) == baseline


# ===========================================================================
# 4. The exact next batch (§14.3 row 1), at both dtypes
# ===========================================================================

@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_next_batch_after_restoration_is_exact(tmp_path, live_storages,
                                                   dtype):
    baseline = settled(live_storages)
    path = str(tmp_path / f"next_{dtype}.npz")

    source = build_graph(dtype, shuffle=True)
    iterator = iter(source.loader)
    train_steps(source, iterator, 2)
    iterator.close()
    loader_state = source.loader.state_dict()
    expected_indices = source.sampler.next_batch_indices()
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(2, loader_state))
    # The oracle, produced from the source's own dataset by index.
    oracle_features = source.dataset.feature_batch(expected_indices)
    try:
        oracle = batch_record(oracle_features,
                              source.dataset.target_batch(expected_indices))
    finally:
        oracle_features.close()
    close_graph(source)

    restored = build_graph(dtype, in_seed=91, out_seed=92, batch_size=BATCH + 1,
                           shuffle=False, seed=999, drop_last=True)
    metadata = load_native_checkpoint(path, restored.model,
                                      optimizer=restored.optimizer)
    restored.loader.load_state_dict(metadata["training"]["data_loader"])

    # Before delivering anything: the planned indices already agree.
    assert restored.sampler.next_batch_indices() == expected_indices

    features, targets = next(iter(restored.loader))
    try:
        record = batch_record(features, targets)
        assert record == oracle
        assert record["dtype"] == dtype
        assert record["device"] == "cpu"
        assert record["contiguous"] is True
        assert record["target_dtype"] == np.int64
        # C-contiguous, owning, and read-only.
        assert record["target_flags"] == (True, True, False)
        assert features.to_numpy().dtype == (
            np.float32 if dtype == "float32" else np.float64)
        # Fresh owning native storage, not a view of anything.
        assert features.closed is False
    finally:
        features.close()
    close_graph(restored)
    assert settled(live_storages) == baseline


@needs_backend
def test_batch_indices_are_identical_across_dtypes(tmp_path):
    """§14.4: the permutation is a pure function of ``(seed, epoch,
    samples)`` and carries no dtype at all, so two equivalent datasets at
    different widths plan the **same** indices — while their states stay
    non-interchangeable in both directions."""
    wide = build_graph("float64", shuffle=True)
    narrow = build_graph("float32", shuffle=True)
    try:
        assert wide.sampler.plan() == narrow.sampler.plan()
        assert (wide.sampler.next_batch_indices()
                == narrow.sampler.next_batch_indices())
        assert wide.sampler.epoch_permutation() \
            == narrow.sampler.epoch_permutation()
        # ...and the two loader states name different data.
        with pytest.raises(ValueError):
            wide.loader.load_state_dict(narrow.loader.state_dict())
        with pytest.raises(ValueError):
            narrow.loader.load_state_dict(wide.loader.state_dict())
    finally:
        close_graph(wide)
        close_graph(narrow)


@needs_backend
def test_a_cross_dtype_loader_state_from_an_archive_is_rejected(
        tmp_path, live_storages):
    """The same mismatch, arriving through a real archive: the checkpoint
    preserves the state faithfully and the **loader** is the authority that
    rejects it against the live dataset."""
    baseline = settled(live_storages)
    path = str(tmp_path / "crossdtype.npz")

    # One float64 model, saved beside a **float32** dataset's loader state.
    # The dtypes that differ are the datasets'; the model's are identical,
    # so the archive itself loads cleanly and only the loader objects.
    source = build_graph(dataset=make_dataset("float32"))
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(
                               0, source.loader.state_dict()))
    saved_state = source.loader.state_dict()
    assert saved_state["sampler"]["dataset"]["feature_dtype"] == "float32"
    close_graph(source)

    restored = build_graph(dataset=make_dataset("float64"))
    metadata = load_native_checkpoint(path, restored.model,
                                      optimizer=restored.optimizer)
    # Preserved verbatim by the archive...
    assert metadata["training"]["data_loader"] == saved_state
    before = loader_fingerprint(restored.loader)
    # ...and rejected by the loader, transactionally.
    with pytest.raises(ValueError):
        restored.loader.load_state_dict(metadata["training"]["data_loader"])
    assert loader_fingerprint(restored.loader) == before
    close_graph(restored)
    assert settled(live_storages) == baseline


# ===========================================================================
# 5. `next_step` — caller metadata, and no off-by-one
# ===========================================================================

@needs_backend
def test_next_step_names_the_first_step_not_yet_executed(tmp_path,
                                                         live_storages):
    baseline = settled(live_storages)
    path = str(tmp_path / "next_step.npz")
    total_steps = 4

    source = build_graph(shuffle=True)
    iterator = iter(source.loader)
    executed = []
    for step in range(total_steps):
        if step == 2:
            break
        train_steps(source, iterator, 1)
        executed.append(step)
    iterator.close()
    completed = executed[-1]
    next_step = completed + 1
    assert executed == [0, 1]
    assert next_step == 2
    loader_state = source.loader.state_dict()
    # The loader agrees by construction: two deliveries, cursor at two.
    assert loader_state["sampler"]["cursor"] == next_step
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(next_step, loader_state))
    close_graph(source)

    restored = build_graph(in_seed=91, shuffle=False, seed=3)
    metadata = load_native_checkpoint(path, restored.model,
                                      optimizer=restored.optimizer)
    value = metadata["training"]["next_step"]
    assert type(value) is int
    assert not isinstance(value, bool)
    assert value == 2
    assert list(range(value, total_steps)) == [2, 3]
    restored.loader.load_state_dict(metadata["training"]["data_loader"])
    assert restored.sampler.cursor == value
    close_graph(restored)
    assert settled(live_storages) == baseline


@needs_backend
def test_a_bool_is_never_accepted_as_a_step_by_the_loader_state():
    """``next_step`` is the caller's, and metadata would carry a ``bool``
    happily — but the loader's own schema refuses one everywhere it
    matters, so a caller who conflated the two is told."""
    graph = build_graph()
    try:
        state = graph.loader.state_dict()
        assert type(state["format_version"]) is int
        assert type(state["sampler"]["cursor"]) is int
        broken = graph.loader.state_dict()
        broken["sampler"]["cursor"] = True
        with pytest.raises(TypeError):
            graph.loader.load_state_dict(broken)
        broken = graph.loader.state_dict()
        broken["format_version"] = True
        with pytest.raises(TypeError):
            graph.loader.load_state_dict(broken)
    finally:
        close_graph(graph)


# ===========================================================================
# 6. Metadata behavior — preserved, never interpreted (§13.3, §13.4)
# ===========================================================================

@needs_backend
def test_the_recommended_metadata_shape_round_trips_unchanged(tmp_path):
    graph = build_graph()
    path = str(tmp_path / "recommended.npz")
    try:
        state = graph.loader.state_dict()
        payload = training_metadata(7, state)
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata=payload)
        metadata = load_native_checkpoint(path, graph.model,
                                          optimizer=graph.optimizer)
        assert metadata == payload
        assert metadata["training"]["data_loader"] == state
        assert metadata["training"]["data_loader"]["format"] == LOADER_FORMAT
        assert (metadata["training"]["data_loader"]["sampler"]["format"]
                == SAMPLER_FORMAT)
    finally:
        close_graph(graph)


@needs_backend
def test_alternate_metadata_keys_and_nesting_survive_unchanged(tmp_path):
    """§13.3: the runtime does not know ``"training"`` or
    ``"data_loader"``, so a caller's own dialect must round-trip exactly."""
    graph = build_graph()
    path = str(tmp_path / "alternate.npz")
    try:
        payload = {
            "resume": {
                "loader_A": graph.loader.state_dict(),
                "step_to_run": 7,
            },
            "notes": ["anything", {"nested": True}, None, 1.5],
        }
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata=payload)
        metadata = load_native_checkpoint(path, graph.model,
                                          optimizer=graph.optimizer)
        assert metadata == payload
        assert graph.loader.load_state_dict(
            metadata["resume"]["loader_A"]) is None
        assert metadata["resume"]["step_to_run"] == 7
    finally:
        close_graph(graph)


@needs_backend
def test_two_loader_states_live_side_by_side_without_interpretation(tmp_path):
    """Two distinct loaders' states under caller-selected keys in one
    metadata tree. The checkpoint preserves both and understands neither;
    **no production multi-loader feature is added**."""
    first = build_graph(shuffle=True, seed=11)
    second = build_graph(shuffle=False, seed=22, batch_size=4)
    path = str(tmp_path / "two_loaders.npz")
    try:
        state_one = first.loader.state_dict()
        state_two = second.loader.state_dict()
        assert state_one != state_two
        payload = {"pipelines": {"train": state_one, "eval": state_two}}
        save_native_checkpoint(path, first.model, optimizer=first.optimizer,
                               metadata=payload)
        metadata = load_native_checkpoint(path, first.model,
                                          optimizer=first.optimizer)
        assert metadata["pipelines"]["train"] == state_one
        assert metadata["pipelines"]["eval"] == state_two
        # Each is still a valid loader state for its own dataset.
        assert first.loader.load_state_dict(
            metadata["pipelines"]["train"]) is None
        assert second.loader.load_state_dict(
            metadata["pipelines"]["eval"]) is None
    finally:
        close_graph(first)
        close_graph(second)


@needs_backend
def test_empty_metadata_keeps_its_existing_behavior(tmp_path):
    graph = build_graph()
    path = str(tmp_path / "empty.npz")
    try:
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer)
        assert load_native_checkpoint(path, graph.model,
                                      optimizer=graph.optimizer) == {}
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata={})
        assert load_native_checkpoint(path, graph.model,
                                      optimizer=graph.optimizer) == {}
        assert manifest_of(path)["metadata"] == {}
    finally:
        close_graph(graph)


@needs_backend
def test_missing_loader_metadata_gets_no_default(tmp_path, live_storages):
    """§13.4: absent is absent. The checkpoint invents no loader state, and
    the caller's own ``.get`` chain returns ``None`` — which
    ``load_state_dict`` refuses, having changed nothing."""
    baseline = settled(live_storages)
    graph = build_graph()
    path = str(tmp_path / "missing.npz")
    try:
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata={"training": {"next_step": 3}})
        metadata = load_native_checkpoint(path, graph.model,
                                          optimizer=graph.optimizer)
        assert metadata == {"training": {"next_step": 3}}
        assert metadata.get("training", {}).get("data_loader") is None
        assert metadata.get("nothing", {}).get("data_loader") is None
        before = loader_fingerprint(graph.loader)
        with pytest.raises(TypeError):
            graph.loader.load_state_dict(
                metadata.get("training", {}).get("data_loader"))
        assert loader_fingerprint(graph.loader) == before
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


@needs_backend
@pytest.mark.parametrize("malformed", [
    {},
    {"format": LOADER_FORMAT, "format_version": 1},
    {"format": SAMPLER_FORMAT, "format_version": 1, "sampler": {}},
    {"format": LOADER_FORMAT, "format_version": 2, "sampler": {}},
    {"format": LOADER_FORMAT, "format_version": 1, "sampler": {}},
    {"format": LOADER_FORMAT, "format_version": 1, "sampler": "not a dict"},
    {"format": LOADER_FORMAT, "format_version": 1, "sampler": {}, "extra": 1},
    [1, 2, 3],
    "a loader state",
    17,
])
def test_malformed_loader_metadata_is_preserved_then_rejected(
        tmp_path, live_storages, malformed):
    """The load-bearing distinction: **the checkpoint accepts ordinary
    JSON-compatible metadata**, because it does not interpret loader state;
    **the loader validates the loader schema**, transactionally, when the
    caller explicitly hands it over."""
    baseline = settled(live_storages)
    graph = build_graph()
    path = str(tmp_path / "malformed.npz")
    try:
        save_native_checkpoint(
            path, graph.model, optimizer=graph.optimizer,
            metadata={"training": {"data_loader": malformed}})
        metadata = load_native_checkpoint(path, graph.model,
                                          optimizer=graph.optimizer)
        # Preserved exactly — the archive did not judge it.
        assert metadata["training"]["data_loader"] == malformed
        before = loader_fingerprint(graph.loader)
        with pytest.raises((TypeError, ValueError)):
            graph.loader.load_state_dict(metadata["training"]["data_loader"])
        # §12.7: a rejection changed nothing at all.
        assert loader_fingerprint(graph.loader) == before
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


@needs_backend
def test_a_wrong_dataset_loader_state_is_rejected_naming_identity(
        tmp_path, live_storages):
    baseline = settled(live_storages)
    path = str(tmp_path / "wrong_dataset.npz")

    source = build_graph()
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(
                               0, source.loader.state_dict()))
    close_graph(source)

    # A dataset with the same shape and dtype but different **content**, so
    # only the fingerprint differs — §12.4 step 8's last comparison.
    other = build_graph(dataset=make_dataset(offset=1.0))
    try:
        metadata = load_native_checkpoint(path, other.model,
                                          optimizer=other.optimizer)
        state = metadata["training"]["data_loader"]
        assert state["sampler"]["dataset"]["samples"] == SAMPLES
        assert (state["sampler"]["dataset"]["fingerprint"]
                != other.dataset.fingerprint)
        before = loader_fingerprint(other.loader)
        with pytest.raises(ValueError) as error:
            other.loader.load_state_dict(state)
        assert "fingerprint" in str(error.value)
        assert loader_fingerprint(other.loader) == before
    finally:
        close_graph(other)
    assert settled(live_storages) == baseline


@needs_backend
@pytest.mark.parametrize("bad", [
    {"loader": object()},
    {1: "a non-string key"},
    {"loss": float("nan")},
    {"loss": float("inf")},
    {"loss": float("-inf")},
    {"values": [1.0, float("nan")]},
    {"array": np.arange(3)},
    {"scalar": np.float64(1.5)},
])
def test_non_json_metadata_is_refused_before_the_destination_moves(
        tmp_path, live_storages, bad):
    """§13.4's fourth row, and the existing metadata validation that
    provides it. J5 **does not weaken it**: the save fails, an existing
    destination stays byte-identical, no temporary file survives, and every
    live object — including the loader snapshot the caller already took —
    is untouched and still valid."""
    baseline = settled(live_storages)
    graph = build_graph()
    directory = tmp_path / "guarded"
    directory.mkdir()
    path = str(directory / "existing.npz")
    try:
        loader_state = graph.loader.state_dict()
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata=training_metadata(1, loader_state))
        original_bytes = Path(path).read_bytes()
        fingerprint = graph_fingerprint(graph)
        loader_before = loader_fingerprint(graph.loader)

        with pytest.raises((TypeError, ValueError)):
            save_native_checkpoint(
                path, graph.model, optimizer=graph.optimizer,
                metadata={"training": {"data_loader": loader_state,
                                       **bad}})

        assert Path(path).read_bytes() == original_bytes
        assert sorted(p.name for p in directory.iterdir()) == ["existing.npz"]
        assert graph_fingerprint(graph) == fingerprint
        assert loader_fingerprint(graph.loader) == loader_before
        # The snapshot itself was never the problem, and still loads.
        assert graph.loader.load_state_dict(loader_state) is None
        # ...and the untouched archive is still readable.
        assert load_native_checkpoint(
            path, graph.model,
            optimizer=graph.optimizer)["training"]["data_loader"] \
            == loader_state
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


@needs_backend
def test_returned_metadata_is_independent_and_caller_owned(tmp_path):
    """A fresh plain-Python dict at every load: mutating it reaches no live
    checkpoint object, no archive, and no loader."""
    graph = build_graph()
    path = str(tmp_path / "independent.npz")
    try:
        state = graph.loader.state_dict()
        payload = training_metadata(5, state)
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata=payload)
        first = load_native_checkpoint(path, graph.model,
                                       optimizer=graph.optimizer)
        assert type(first) is dict
        assert type(first["training"]) is dict
        assert type(first["training"]["data_loader"]) is dict
        first["training"]["next_step"] = 999
        first["training"]["data_loader"]["sampler"]["cursor"] = 99
        first["injected"] = "nonsense"

        second = load_native_checkpoint(path, graph.model,
                                        optimizer=graph.optimizer)
        assert second == payload
        assert second is not first
        assert second["training"] is not first["training"]
        assert "injected" not in second
        # The caller's own pre-save dict was not captured by reference
        # either: the archive is a copy, so editing it changes nothing.
        payload["training"]["next_step"] = -1
        third = load_native_checkpoint(path, graph.model,
                                       optimizer=graph.optimizer)
        assert third["training"]["next_step"] == 5
    finally:
        close_graph(graph)


# ===========================================================================
# 7. Save ordering (§13.5)
# ===========================================================================

@needs_backend
def test_the_loader_snapshot_precedes_the_save_and_nothing_iterates_between(
        tmp_path, monkeypatch, live_storages):
    """The supported order, proved as an ordered event log rather than
    claimed: ``state_dict`` fires, then the save, and no batch is delivered
    in between — so the archive describes the exact next batch."""
    baseline = settled(live_storages)
    path = str(tmp_path / "order.npz")
    graph = build_graph(shuffle=True)
    try:
        iterator = iter(graph.loader)
        train_steps(graph, iterator, 1)
        iterator.close()

        events = []
        original_state_dict = NativeDataLoader.state_dict
        original_save = checkpoint_module.save_native_checkpoint
        original_seam = loader_module._deliver_batch

        def spy_state_dict(self):
            events.append("snapshot")
            return original_state_dict(self)

        def spy_save(*args, **kwargs):
            events.append("save")
            return original_save(*args, **kwargs)

        def spy_seam(record):
            events.append("deliver")
            return original_seam(record)

        monkeypatch.setattr(NativeDataLoader, "state_dict", spy_state_dict)
        monkeypatch.setattr(loader_module, "_deliver_batch", spy_seam)

        # 1. snapshot, 2. save. Nothing between them.
        loader_state = graph.loader.state_dict()
        expected = graph.sampler.next_batch_indices()
        spy_save(path, graph.model, optimizer=graph.optimizer,
                 metadata=training_metadata(1, loader_state))
        assert events == ["snapshot", "save"]

        # The saved state really does describe the exact next batch: the
        # very next delivery uses exactly those indices.
        delivered_indices = graph.sampler.next_batch_indices()
        features, _ = next(iter(graph.loader))
        features.close()
        assert events == ["snapshot", "save", "deliver"]
        assert delivered_indices == expected
        assert manifest_of(path)["metadata"]["training"]["data_loader"] \
            == loader_state
        # ...and the state the archive holds is the pre-delivery one.
        assert loader_state["sampler"]["cursor"] == 1
        assert graph.loader.state_dict()["sampler"]["cursor"] == 2
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


@needs_backend
def test_the_saved_state_describes_the_batch_the_run_had_not_taken(
        tmp_path, live_storages):
    """The non-vacuity half of the ordering rule: if the loader *were*
    iterated between the snapshot and the save, the archive would describe
    a position already left — so the two are compared explicitly."""
    baseline = settled(live_storages)
    path_correct = str(tmp_path / "correct.npz")
    path_late = str(tmp_path / "late.npz")

    graph = build_graph(shuffle=True)
    try:
        iterator = iter(graph.loader)
        train_steps(graph, iterator, 1)
        iterator.close()

        early_state = graph.loader.state_dict()
        expected = graph.sampler.next_batch_indices()
        save_native_checkpoint(path_correct, graph.model,
                               optimizer=graph.optimizer,
                               metadata=training_metadata(1, early_state))

        # The mistake the rule forbids, performed deliberately so the two
        # archives can be compared.
        features, _ = next(iter(graph.loader))
        features.close()
        late_state = graph.loader.state_dict()
        save_native_checkpoint(path_late, graph.model,
                               optimizer=graph.optimizer,
                               metadata=training_metadata(1, late_state))

        assert early_state != late_state
        assert (manifest_of(path_correct)["metadata"]["training"]
                ["data_loader"]["sampler"]["cursor"]
                != manifest_of(path_late)["metadata"]["training"]
                ["data_loader"]["sampler"]["cursor"])
        # The correctly ordered archive names the batch that had not run.
        assert early_state["sampler"]["cursor"] == 1
        assert expected == graph.sampler.plan()[1]
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


# ===========================================================================
# 8. Restore ordering (§13.5)
# ===========================================================================

@needs_backend
def test_a_failed_checkpoint_load_never_reaches_the_loader(
        tmp_path, monkeypatch, live_storages):
    """Restore order, proved from the failure side: the checkpoint load is
    first, so when it fails the loader has not been touched — not its
    state, not its plan, not its iterator slot, and not one byte of native
    storage."""
    baseline = settled(live_storages)
    path = str(tmp_path / "mismatch.npz")

    source = build_graph()
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(
                               1, source.loader.state_dict()))
    close_graph(source)

    # A model whose parameter shapes cannot match the archive: the load
    # fails in Phase 1, before anything is staged.
    mismatched = build_graph(hidden=HIDDEN + 2, batch_size=BATCH,
                             shuffle=True, seed=5, lr=0.02)
    try:
        # The complete "before" picture is taken *before* the spies exist,
        # so reading it cannot itself be mistaken for a checkpoint call.
        before = loader_fingerprint(mismatched.loader)
        storage_before = settled(live_storages)
        calls = record_loader_state_calls(monkeypatch)

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            load_native_checkpoint(path, mismatched.model,
                                   optimizer=mismatched.optimizer)

        assert calls == [], "a failed checkpoint load called a loader method"
        assert settled(live_storages) == storage_before
        # ...and only now, having proved the load called nothing, does the
        # test read the loader again.
        assert loader_fingerprint(mismatched.loader) == before
        # The archive is unchanged and still reusable.
        assert manifest_of(path)["format_version"] == CHECKPOINT_VERSION
    finally:
        close_graph(mismatched)
    assert settled(live_storages) == baseline


@needs_backend
def test_the_loader_is_loaded_only_by_an_explicit_caller_line(
        tmp_path, monkeypatch, live_storages):
    """The runtime half of §13.6: a real save and a real load call neither
    loader state method, and the caller's own two lines call exactly one
    each, in order."""
    baseline = settled(live_storages)
    path = str(tmp_path / "explicit.npz")
    graph = build_graph()
    try:
        calls = record_loader_state_calls(monkeypatch)

        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata={"training": {"next_step": 0}})
        assert calls == [], "save_native_checkpoint called a loader method"

        metadata = load_native_checkpoint(path, graph.model,
                                          optimizer=graph.optimizer)
        assert calls == [], "load_native_checkpoint called a loader method"
        assert metadata == {"training": {"next_step": 0}}

        # Now the caller's own lines, and only now.
        state = graph.loader.state_dict()
        assert calls == ["state_dict"]
        graph.loader.load_state_dict(state)
        assert calls == ["state_dict", "load_state_dict"]
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


# ===========================================================================
# 9. The atomicity boundary — and the honest absence of a cross-object one
# ===========================================================================

@needs_backend
def test_a_loader_load_failing_after_a_successful_checkpoint_load_rolls_back_nothing(
        tmp_path, live_storages):
    """§13.5's explicit non-atomicity, executable.

    The checkpoint load is atomic over the model, the optimizer, and every
    registered generator. The loader load is atomic over the loader and its
    sampler. **There is no transaction spanning the two**, and this test
    exists to prove the repository does not quietly grow one: the first
    call succeeds, the second fails, and the first is **not** undone.
    """
    baseline = settled(live_storages)
    path = str(tmp_path / "no_cross_object.npz")

    source = build_graph(shuffle=True)
    iterator = iter(source.loader)
    train_steps(source, iterator, 2)
    iterator.close()
    loader_state = source.loader.state_dict()
    fingerprint = graph_fingerprint(source)
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(2, loader_state))
    close_graph(source)

    # 2-3. A fresh **compatible** model and optimizer, but a fresh
    #      **incompatible** dataset, sampler, and loader.
    wrong_dataset = make_dataset(offset=2.5)
    incompatible = build_graph(in_seed=91, out_seed=92, shared_seed=93,
                               own_seed=94, lr=0.09, batch_size=BATCH + 1,
                               shuffle=False, seed=808,
                               dataset=wrong_dataset)
    try:
        # 4. capture the fresh loader's state.
        loader_before = loader_fingerprint(incompatible.loader)
        stale = graph_fingerprint(incompatible)
        assert stale != fingerprint

        # 5-6. the checkpoint load succeeds and restores all three families.
        metadata = load_native_checkpoint(path, incompatible.model,
                                          optimizer=incompatible.optimizer)
        assert graph_fingerprint(incompatible) == fingerprint

        # 7-8. the loader load is rejected on dataset identity.
        with pytest.raises(ValueError):
            incompatible.loader.load_state_dict(
                metadata["training"]["data_loader"])

        # 9. the loader is exactly as it was...
        assert loader_fingerprint(incompatible.loader) == loader_before
        # 10-11. ...and the model, optimizer, and generators stay restored.
        #        Nothing rolled the first call back, and nothing claims it
        #        would.
        assert graph_fingerprint(incompatible) == fingerprint
        # The archive is untouched and reusable.
        assert manifest_of(path)["format_version"] == CHECKPOINT_VERSION
    finally:
        close_graph(incompatible)

    # 12-13. The documented recovery: discard everything, rebuild a wholly
    #        fresh compatible graph, and repeat the two calls from the same
    #        archive. Nothing partial survived, because nothing was written.
    recovered = build_graph(in_seed=71, out_seed=72, lr=0.03,
                            batch_size=BATCH + 2, shuffle=False, seed=606)
    try:
        metadata = load_native_checkpoint(path, recovered.model,
                                          optimizer=recovered.optimizer)
        assert graph_fingerprint(recovered) == fingerprint
        assert recovered.loader.load_state_dict(
            metadata["training"]["data_loader"]) is None
        assert recovered.loader.state_dict() == loader_state
    finally:
        close_graph(recovered)
    assert settled(live_storages) == baseline


@needs_backend
def test_one_batch_handoff_is_atomic_and_a_snapshot_cannot_see_inside_it(
        monkeypatch, live_storages):
    """The third atomicity row (§9.4/§9.5): ``__next__`` is atomic over one
    handoff, and while it is open ``state_dict()`` **refuses** rather than
    reporting a committed cursor that skipped an undelivered batch."""
    baseline = settled(live_storages)
    graph = build_graph(shuffle=True)
    try:
        observed = []
        original_seam = loader_module._deliver_batch

        def probing_seam(record):
            observed.append({
                "epoch": graph.sampler._epoch,
                "cursor": graph.sampler._cursor,
                "loader_refused": refuses_state_dict(graph.loader),
                "sampler_refused": refuses_state_dict(graph.sampler),
                "transaction": graph.sampler._transaction is not None,
            })
            return original_seam(record)

        monkeypatch.setattr(loader_module, "_deliver_batch", probing_seam)
        before = graph.loader.state_dict()
        features, _ = next(iter(graph.loader))
        features.close()

        assert len(observed) == 1, "the probe never ran"
        seen = observed[0]
        # Non-vacuity: the candidate position really had been applied when
        # the snapshot was refused — the raw fields had already moved.
        assert seen["transaction"] is True
        assert (seen["epoch"], seen["cursor"]) != (
            before["sampler"]["epoch"], before["sampler"]["cursor"])
        assert seen["loader_refused"] is True
        assert seen["sampler_refused"] is True
        # Outside the transaction it answers again, and correctly.
        assert graph.loader.state_dict()["sampler"]["cursor"] == 1
    finally:
        close_graph(graph)
    assert settled(live_storages) == baseline


# ===========================================================================
# 10. Delivery boundaries (§13.7) — the three rows, each through an archive
# ===========================================================================

@needs_backend
def test_a_failed_delivery_then_a_checkpoint_resumes_the_same_candidate(
        tmp_path, monkeypatch, live_storages):
    """**The load-bearing J5 test.**

    A delivery that fails at the ``_deliver_batch`` seam consumes nothing:
    the §9.4 Phase-5 rollback restores the exact pre-delivery position, so
    a checkpoint taken immediately afterwards records the **same candidate
    batch** the failed call was about to deliver — and a fresh graph
    restored from it delivers exactly that batch, once.
    """
    baseline = settled(live_storages)
    path = str(tmp_path / "failed_delivery.npz")

    source = build_graph(shuffle=True)
    iterator = iter(source.loader)
    train_steps(source, iterator, 1)
    iterator.close()

    # --- everything the rollback must restore, captured first.
    pre_state = source.loader.state_dict()
    pre_sampler_state = source.sampler.state_dict()
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

    # --- inject a real failure at the seam, after the candidate has been
    #     applied. The observation list is the non-vacuity proof.
    observed = []
    original_seam = loader_module._deliver_batch

    def exploding_seam(record):
        observed.append({
            "epoch": source.sampler._epoch,
            "cursor": source.sampler._cursor,
            "refused": refuses_state_dict(source.loader),
            "transaction": source.sampler._transaction is not None,
        })
        raise RuntimeError("injected delivery failure")

    monkeypatch.setattr(loader_module, "_deliver_batch", exploding_seam)
    iterator = iter(source.loader)
    with pytest.raises(RuntimeError, match="injected delivery failure"):
        next(iterator)
    # The seam goes back to normal for the retry half of the proof — by
    # restoring it explicitly, never by undoing every patch this test's
    # fixtures also installed.
    monkeypatch.setattr(loader_module, "_deliver_batch", original_seam)

    # --- the seam really executed at the committed/pending phase.
    assert len(observed) == 1, "the delivery seam never ran"
    assert observed[0]["transaction"] is True
    assert (observed[0]["epoch"], observed[0]["cursor"]) == candidate_after
    assert (observed[0]["epoch"], observed[0]["cursor"]) != (
        pre_state["sampler"]["epoch"], pre_state["sampler"]["cursor"])
    assert observed[0]["refused"] is True

    # --- the rollback was complete.
    assert source.sampler.epoch == pre_state["sampler"]["epoch"]
    assert source.sampler.cursor == pre_state["sampler"]["cursor"]
    assert source.loader.state_dict() == pre_state
    assert source.sampler.state_dict() == pre_sampler_state
    assert source.sampler.next_batch_indices() == candidate
    assert settled(live_storages) == pre_storage
    assert source.sampler._transaction is None
    iterator.close()

    # --- the checkpoint therefore records the same candidate.
    post_state = source.loader.state_dict()
    assert post_state == pre_state
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(1, post_state))
    assert (manifest_of(path)["metadata"]["training"]["data_loader"]
            == post_state)
    close_graph(source)

    # --- a wholly fresh graph resumes from exactly that batch.
    restored = build_graph(in_seed=91, out_seed=92, shared_seed=93,
                           own_seed=94, lr=0.05, batch_size=BATCH + 2,
                           shuffle=False, seed=31337, drop_last=False)
    try:
        metadata = load_native_checkpoint(path, restored.model,
                                          optimizer=restored.optimizer)
        assert restored.sampler.next_batch_indices() != candidate
        restored.loader.load_state_dict(metadata["training"]["data_loader"])
        assert restored.sampler.next_batch_indices() == candidate
        cursor_before = restored.sampler.cursor

        features, targets = next(iter(restored.loader))
        try:
            assert batch_record(features, targets) == oracle
        finally:
            features.close()
        # It advanced **exactly once**.
        assert restored.sampler.cursor == cursor_before + 1
        assert restored.loader.state_dict()["sampler"]["cursor"] \
            == cursor_before + 1
    finally:
        close_graph(restored)
    assert settled(live_storages) == baseline


@needs_backend
def test_a_successful_delivery_then_a_checkpoint_resumes_the_following_batch(
        tmp_path, live_storages):
    """The committed direction: the position moved exactly once, when
    ownership transferred, so the archive names the **next** batch and the
    delivered one is never replayed."""
    baseline = settled(live_storages)
    path = str(tmp_path / "delivered.npz")

    source = build_graph(shuffle=True)
    plan = source.sampler.plan()
    iterator = iter(source.loader)
    delivered_indices = source.sampler.next_batch_indices()
    features, targets = next(iterator)
    try:
        delivered = batch_record(features, targets)
    finally:
        features.close()
    iterator.close()
    assert delivered_indices == plan[0]

    following_indices = source.sampler.next_batch_indices()
    assert following_indices == plan[1]
    assert following_indices != delivered_indices
    oracle_features = source.dataset.feature_batch(following_indices)
    try:
        following = batch_record(
            oracle_features, source.dataset.target_batch(following_indices))
    finally:
        oracle_features.close()

    loader_state = source.loader.state_dict()
    assert loader_state["sampler"]["cursor"] == 1
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(1, loader_state))
    close_graph(source)

    restored = build_graph(in_seed=91, batch_size=BATCH + 2, shuffle=False,
                           seed=5150)
    try:
        metadata = load_native_checkpoint(path, restored.model,
                                          optimizer=restored.optimizer)
        restored.loader.load_state_dict(metadata["training"]["data_loader"])
        assert restored.sampler.next_batch_indices() == following_indices
        assert restored.sampler.next_batch_indices() != delivered_indices

        features, targets = next(iter(restored.loader))
        try:
            record = batch_record(features, targets)
        finally:
            features.close()
        assert record == following
        assert record != delivered, "the delivered batch was replayed"
    finally:
        close_graph(restored)
    assert settled(live_storages) == baseline


@needs_backend
def test_an_epoch_boundary_save_resumes_the_first_batch_of_the_next_epoch(
        tmp_path, live_storages):
    """§7.4's canonical boundary, carried through an archive. The state
    reads ``(epoch + 1, 0)`` and nothing else: no terminal marker, no
    end-of-epoch flag, and no special resume rule."""
    baseline = settled(live_storages)
    path = str(tmp_path / "boundary.npz")

    source = build_graph(shuffle=True)
    per_epoch = source.sampler.batches_per_epoch
    assert per_epoch == 4
    iterator = iter(source.loader)
    for _ in range(per_epoch):
        features, _ = next(iterator)
        features.close()
    iterator.close()

    loader_state = source.loader.state_dict()
    # The canonical form, and the only representation of it.
    assert (loader_state["sampler"]["epoch"],
            loader_state["sampler"]["cursor"]) == (1, 0)
    assert set(loader_state) == {"format", "format_version", "sampler"}
    assert source.sampler.remaining == per_epoch
    next_epoch_indices = source.sampler.next_batch_indices()
    assert next_epoch_indices == source.sampler.plan(1)[0]
    oracle_features = source.dataset.feature_batch(next_epoch_indices)
    try:
        oracle = batch_record(
            oracle_features, source.dataset.target_batch(next_epoch_indices))
    finally:
        oracle_features.close()

    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(per_epoch, loader_state))
    close_graph(source)

    restored = build_graph(in_seed=91, batch_size=BATCH + 1, shuffle=False,
                           seed=2718, drop_last=True)
    try:
        metadata = load_native_checkpoint(path, restored.model,
                                          optimizer=restored.optimizer)
        restored.loader.load_state_dict(metadata["training"]["data_loader"])
        assert (restored.sampler.epoch, restored.sampler.cursor) == (1, 0)
        assert restored.sampler.next_batch_indices() == next_epoch_indices
        assert restored.sampler.remaining == per_epoch
        features, targets = next(iter(restored.loader))
        try:
            assert batch_record(features, targets) == oracle
        finally:
            features.close()
        # ...and the metadata's next_step needed no boundary special case.
        assert metadata["training"]["next_step"] == per_epoch
    finally:
        close_graph(restored)
    assert settled(live_storages) == baseline


# ===========================================================================
# 11. The archive itself — shape, and the absence of a loader anywhere in it
# ===========================================================================

@needs_backend
def test_the_archive_format_and_root_schema_did_not_move(tmp_path):
    graph = build_graph()
    path = str(tmp_path / "shape.npz")
    try:
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata=training_metadata(
                                   1, graph.loader.state_dict()))
        manifest = manifest_of(path)
        assert manifest["format"] == CHECKPOINT_FORMAT
        assert manifest["format_version"] == CHECKPOINT_VERSION
        assert set(manifest) == MANIFEST_ROOT_KEYS
        assert checkpoint_module._FORMAT == CHECKPOINT_FORMAT
        assert checkpoint_module._FORMAT_VERSION == CHECKPOINT_VERSION
        assert (checkpoint_module._SUPPORTED_FORMAT_VERSIONS
                == CHECKPOINT_VERSIONS)
        assert checkpoint_module._MANIFEST_KEYS == MANIFEST_ROOT_KEYS
    finally:
        close_graph(graph)


@needs_backend
def test_loader_state_exists_only_below_metadata(tmp_path):
    """No root field, no loader array, no permutation, no dataset payload.
    The only numeric arrays are the model's parameters/buffers and the
    optimizer's moments, exactly as before J5."""
    graph = build_graph()
    path = str(tmp_path / "inventory.npz")
    try:
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata=training_metadata(
                                   1, graph.loader.state_dict()))
        manifest = manifest_of(path)
        names = array_names_of(path)

        # 1. every array is the manifest, a model entry, or a moment.
        assert "manifest" in names
        for name in names:
            assert (name == "manifest"
                    or name.startswith("model::")
                    or name.startswith("optimizer::m::")
                    or name.startswith("optimizer::v::")), name
        # 2. nothing in the archive is named for a Phase-J concept.
        for name in names:
            lowered = name.lower()
            for forbidden in ("loader", "sampler", "dataset", "permut",
                              "batch", "shuffle", "cursor", "epoch",
                              "target", "feature"):
                assert forbidden not in lowered, (name, forbidden)
        # 3. the model arrays account for exactly the model's own state.
        model_arrays = [n for n in names if n.startswith("model::")]
        assert len(model_arrays) == len(manifest["model"]["keys"])

        # 4. the loader's format tag appears **only** under metadata.
        for section in ("model", "optimizer", "generators"):
            blob = json.dumps(manifest[section])
            assert LOADER_FORMAT not in blob, section
            assert SAMPLER_FORMAT not in blob, section
        assert LOADER_FORMAT in json.dumps(manifest["metadata"])
        for key in MANIFEST_ROOT_KEYS - {"metadata"}:
            assert "data_loader" not in json.dumps(manifest[key]), key

        # 5. the metadata is JSON, not pickle, and the load path proves it
        #    by reading the archive with pickle disabled.
        with np.load(path, allow_pickle=False) as archive:
            assert archive["manifest"].dtype == np.uint8
            for name in names:
                assert archive[name].dtype != np.object_
        assert load_native_checkpoint(
            path, graph.model,
            optimizer=graph.optimizer)["training"]["data_loader"] \
            == graph.loader.state_dict()
    finally:
        close_graph(graph)


@needs_backend
def test_the_archive_grows_only_by_the_metadata_the_caller_supplied(tmp_path):
    """The capture set did not grow: saving **with** and **without** loader
    state produces the same array inventory and the same manifest apart
    from the caller's own ``metadata`` value."""
    graph = build_graph()
    without = str(tmp_path / "without.npz")
    with_state = str(tmp_path / "with.npz")
    try:
        save_native_checkpoint(without, graph.model,
                               optimizer=graph.optimizer)
        save_native_checkpoint(with_state, graph.model,
                               optimizer=graph.optimizer,
                               metadata=training_metadata(
                                   1, graph.loader.state_dict()))
        assert array_names_of(without) == array_names_of(with_state)
        bare, carrying = manifest_of(without), manifest_of(with_state)
        assert set(bare) == set(carrying) == MANIFEST_ROOT_KEYS
        for key in MANIFEST_ROOT_KEYS - {"metadata"}:
            assert bare[key] == carrying[key], key
        assert bare["metadata"] == {}
        assert carrying["metadata"]["training"]["data_loader"]["format"] \
            == LOADER_FORMAT
    finally:
        close_graph(graph)


# ===========================================================================
# 12. Non-coupling (§13.6), in both directions
# ===========================================================================

@pytest.mark.parametrize("relative", PIPELINE_SOURCES)
def test_no_pipeline_module_references_checkpoint_runtime(relative):
    names = code_identifiers(relative)
    for forbidden in ("native_checkpoint", "save_native_checkpoint",
                      "load_native_checkpoint", "_validated_metadata",
                      "savez", "npz"):
        assert forbidden not in names, (relative, forbidden)


def test_the_checkpoint_module_references_no_pipeline_object():
    names = code_identifiers(CHECKPOINT_SOURCE)
    for forbidden in ("NativeTensorDataset", "NativeBatchSampler",
                      "NativeDataLoader", "native_dataset", "native_sampler",
                      "native_data_loader", "_native_permutation"):
        assert forbidden not in names, forbidden


def test_the_checkpoint_module_has_no_loader_discovery_of_any_kind():
    """Absence asserted as *executable* names, not as prose: a registry, a
    traversal, a registration, or an automatic call would all show up
    here."""
    names = code_identifiers(CHECKPOINT_SOURCE)
    for forbidden in ("named_loaders", "loaders", "data_loaders",
                      "register_loader", "loader_state_dict",
                      "load_loader_state_dict", "named_datasets", "datasets",
                      "named_samplers", "samplers", "discover_loaders",
                      "loader_registry", "map_location"):
        assert forbidden not in names, forbidden
    # ...and the module names no caller convention either.
    source = (REPO_ROOT / CHECKPOINT_SOURCE).read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value,
                                                                 str)}
    for convention in ("data_loader", "training", "next_step",
                       LOADER_FORMAT, SAMPLER_FORMAT):
        assert convention not in literals, convention


def test_no_module_registers_or_traverses_loaders():
    """The other half of "no discovery": nothing on the module system knows
    a loader exists, so there is nothing for a checkpoint to find."""
    from tensorforge.experimental import NativeModule

    for forbidden in ("register_loader", "loaders", "named_loaders",
                      "data_loaders", "loader_state_dict",
                      "load_loader_state_dict", "register_dataset",
                      "datasets", "named_datasets"):
        assert not hasattr(NativeModule, forbidden), forbidden
    for forbidden in ("save", "load", "checkpoint", "save_checkpoint",
                      "load_checkpoint", "to_metadata", "from_metadata"):
        assert not hasattr(NativeDataLoader, forbidden), forbidden


def test_the_checkpoint_entry_points_take_no_loader_argument():
    """No loader convenience argument appeared on either function, and no
    loader convenience method appeared on the loader."""
    assert list(inspect.signature(save_native_checkpoint).parameters) == [
        "path", "model", "optimizer", "metadata"]
    assert list(inspect.signature(load_native_checkpoint).parameters) == [
        "path", "model", "optimizer"]


# ===========================================================================
# 13. Cleanup, and the live-storage baseline (§14.5, §15)
# ===========================================================================

@needs_backend
def test_the_whole_workflow_returns_native_storage_to_baseline(
        tmp_path, live_storages):
    """Every delivered feature batch closed explicitly, every temporary
    closed at its last use, both graphs closed, and the live-storage count
    exactly back where it started."""
    baseline = settled(live_storages)
    path = str(tmp_path / "cleanup.npz")

    source = build_graph(shuffle=True)
    iterator = iter(source.loader)
    train_steps(source, iterator, 2)
    iterator.close()
    state = source.loader.state_dict()
    save_native_checkpoint(path, source.model, optimizer=source.optimizer,
                           metadata=training_metadata(2, state))
    close_graph(source)

    restored = build_graph(in_seed=91, shuffle=False, seed=17)
    metadata = load_native_checkpoint(path, restored.model,
                                      optimizer=restored.optimizer)
    restored.loader.load_state_dict(metadata["training"]["data_loader"])
    for features, _ in drain(restored.loader):
        assert features.closed is False
    close_graph(restored)
    assert settled(live_storages) == baseline


@needs_backend
def test_the_saved_archive_is_a_real_file_and_leaves_no_temporary(tmp_path):
    directory = tmp_path / "archive"
    directory.mkdir()
    path = str(directory / "real.npz")
    graph = build_graph()
    try:
        save_native_checkpoint(path, graph.model, optimizer=graph.optimizer,
                               metadata=training_metadata(
                                   1, graph.loader.state_dict()))
        assert sorted(p.name for p in directory.iterdir()) == ["real.npz"]
        assert os.path.getsize(path) > 0
    finally:
        close_graph(graph)


# ===========================================================================
# 14. J5's non-goals, asserted as absence
# ===========================================================================

def test_no_checkpoint_version_four_and_no_new_root_field():
    assert checkpoint_module._FORMAT_VERSION == 3
    assert checkpoint_module._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert 4 not in checkpoint_module._SUPPORTED_FORMAT_VERSIONS
    assert checkpoint_module._MANIFEST_KEYS == MANIFEST_ROOT_KEYS


def test_j5_added_no_public_export():
    """The export delta is zero, exactly as J4's was."""
    assert len(experimental.__all__) == 25
    assert len(set(experimental.__all__)) == 25
    for forbidden in ("NativeCheckpointMetadata", "save_loader_state",
                      "load_loader_state", "save_training_checkpoint",
                      "resume_native_training", "NativeTrainingState",
                      "native_loader_metadata"):
        assert forbidden not in experimental.__all__, forbidden
        assert not hasattr(experimental, forbidden), forbidden


def test_j5_added_no_production_module():
    """No new production integration module appeared beside the four
    Phase-J ones."""
    present = {path.name for path in PACKAGE.glob("*.py")}
    for forbidden in ("native_data_checkpoint.py", "native_training.py",
                      "native_resume.py", "native_loader_checkpoint.py",
                      "_native_loader_registry.py"):
        assert forbidden not in present, forbidden
    for expected in ("native_dataset.py", "native_sampler.py",
                     "native_data_loader.py", "_native_permutation.py",
                     "native_checkpoint.py"):
        assert expected in present, expected


def test_j5_added_no_example_and_no_benchmark():
    """J5 itself shipped neither: it is an integration proof whose whole
    diff is this module plus documentation.

    The example that exists in the tree today is **J6's**
    (``native_minibatch_training.py``), and it is named here rather than
    merely counted so this check keeps stating *which* artifact each
    milestone contributed. J5's own delta is still zero, and the benchmark
    that exists in the tree today is **J8's**
    (``benchmark_native_data_pipeline.py``) — shipping either under a J5
    heading would be the over-claim this repository's guardrails exist to
    prevent."""
    examples = sorted(path.name
                      for path in (REPO_ROOT / "examples").glob("*.py")
                      if path.name != "__init__.py")
    benchmarks = sorted(path.name
                        for path in (REPO_ROOT / "benchmarks").glob("*.py")
                        if path.name != "__init__.py")
    # 15 at J5, 16 since J6 — the one example J6 added — and 17 since
    # **K6** added the one integer-indexing example. Both are named rather
    # than merely counted, so an unrecorded example still fails.
    assert len(examples) == 17, examples
    assert "native_minibatch_training.py" in examples
    assert "native_integer_indexing.py" in examples
    # 8 at J5, 9 since **J8** — the one benchmark J8 added — and 10 since
    # **K8** added the one integer characterization harness. Both are
    # named, so an unrecorded benchmark still fails.
    assert len(benchmarks) == 10, benchmarks
    assert "benchmark_native_data_pipeline.py" in benchmarks
    assert "benchmark_native_integer.py" in benchmarks
    # J6 landed its own proof module, J7 its hardening matrix, J8 its
    # benchmark contract module, and J9 the closure guardrails — all four
    # test-only, and each in a file of its own. J5's own artifact delta to
    # both inventories stays zero, which is what the counts above assert.
    for shipped in ("test_native_minibatch_training.py",
                    "test_native_data_hardening.py",
                    "test_native_data_benchmark.py",
                    "test_native_phase_j_closure.py"):
        assert (REPO_ROOT / "tests" / shipped).exists(), shipped


def test_j5_touched_no_cpp_or_build_surface():
    """Asserted against the tree: the capability boundary Phase J never
    moves, and the build surface J5 has no reason to touch."""
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    info = cpp.backend_info()
    assert info["dtype"] == "float64"
    assert info["stable_framework_integration"] is False
    from tensorforge.experimental import native_optimizer_state

    assert native_optimizer_state.FORMAT_VERSION == 1
