"""Loader state and exact mid-epoch resume (Phase J, milestone J4;
docs/native_data_pipeline_design.md §3.5, §6, §7, §9.3, §9.4, §9.5, §11.1,
§11.2, §11.3, §11.4, §11.5, §12.4, §12.5, §12.7, §15.3, §16, §17.4, §17.5,
§18, §19, §20).

J4 adds exactly two methods to an existing class — ``NativeDataLoader
.state_dict()`` and ``NativeDataLoader.load_state_dict(state)`` — and no
public name, no export, no C ABI symbol, no checkpoint field, and no
checkpoint version. What this module proves:

* **§11.3 the schema** — a compact tagged wrapper with **exactly three**
  root keys around **exactly** the sampler's unchanged §11.2 state; fresh
  containers at every level and at every call; JSON round-trippable; and
  accepted unchanged by the checkpoint's own existing metadata validator,
  which is what lets a *caller* carry it through the version-3 metadata
  channel without the archive growing a field.
* **§11.4/§11.5 what state may not carry, and may not repair** — no
  permutation, no payload, no NumPy object, no process-local value; and
  no cast, coercion, default, or clamp on the way back in.
* **§12.5 the load, in exact order** — closed guard, transaction guard,
  active-iteration guard, wrapper, the **delegated** nested sampler
  validation, then a commit that cannot fail. Precedence is probed with
  deliberately malformed arguments, so "the guard ran first" is evidence
  rather than a claim.
* **§12.7/§17.5 the transactional rejection** — every rejected load is
  compared against a complete before/after fingerprint of the observable
  world, down to the permutation cache's behavior, the iterator slot, and
  the native live-storage count.
* **The J4 exit gate** — a restored loader over a **separately
  constructed** dataset, sampler, and loader reproduces the exact
  remaining batch sequence of an interrupted epoch: identical indices,
  identical raw IEEE-754 feature bits, identical targets, the same
  canonical next-epoch position, and the same following whole epoch.
  **No tolerance is used anywhere.**

**Not tested here, because it belongs to another module:** the
caller-managed checkpoint-metadata workflow, which landed at J5 and is
proved end to end in ``tests/test_native_data_checkpoint.py``. What §12
below still asserts about it is only the *production* non-coupling — that
this module imports no checkpoint code and no checkpoint code knows a
loader exists — which J5 did not change and could not.

**Not tested here, because it does not exist:** automatic loader
discovery, a training example, and a benchmark. Those are J6 onward, and
their absence *is* asserted, in §12 below.

No test here asserts an exact error message, a dict ordering, a timing, a
GC event, or a speed.

Selector: python -m pytest -q tests/test_native_loader_state.py
"""

import ast
import gc
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest

import tensorforge
import tensorforge.experimental as experimental
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeBatchSampler, NativeDataLoader, NativeTensor, NativeTensorDataset,
)
from tensorforge.experimental import native_checkpoint
from tensorforge.experimental import native_data_loader as loader_module
from tensorforge.experimental import native_sampler as sampler_module

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "src" / "tensorforge" / "experimental"
LOADER_SOURCE = "src/tensorforge/experimental/native_data_loader.py"
SAMPLER_SOURCE = "src/tensorforge/experimental/native_sampler.py"
DATASET_SOURCE = "src/tensorforge/experimental/native_dataset.py"
CHECKPOINT_SOURCE = "src/tensorforge/experimental/native_checkpoint.py"

LOADER_FORMAT = "tensorforge.native_data_loader"
SAMPLER_FORMAT = "tensorforge.native_sampler"
LOADER_ROOT_KEYS = {"format", "format_version", "sampler"}
SAMPLER_ROOT_KEYS = {"format", "format_version", "dataset", "seed", "shuffle",
                     "batch_size", "drop_last", "epoch", "cursor"}
DATASET_KEYS = {"samples", "feature_shape", "feature_dtype", "fingerprint"}

# Everything that materializes a batch needs the built library. The schema,
# the validation ordering, the rejection fingerprints, the guards, and the
# absence checks are pure Python and stay provable without a compiler.
needs_backend = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

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
    no public counter, and J4 adds none."""
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


def make_dataset(samples=8, width=2, dtype=None, rank=2, offset=0.0):
    """A dataset whose values identify their row exactly, so a batch can be
    checked against the indices that produced it."""
    if rank == 1:
        features = np.arange(samples, dtype=np.float64) * 10.0 + offset
    elif rank == 2:
        features = (np.arange(samples * width, dtype=np.float64)
                    .reshape(samples, width)) + offset
    else:
        features = (np.arange(samples * width * 3, dtype=np.float64)
                    .reshape(samples, width, 3)) + offset
    targets = np.arange(samples, dtype=np.int64) % 3
    return NativeTensorDataset(features, targets, dtype=dtype)


def make_loader(samples=8, dataset=None, **kwargs):
    """A loader over a fresh dataset and sampler. Returns
    ``(loader, sampler, dataset)`` so a test can reach every level."""
    kwargs.setdefault("batch_size", 3)
    dataset = make_dataset(samples) if dataset is None else dataset
    sampler = NativeBatchSampler(dataset, **kwargs)
    return NativeDataLoader(sampler), sampler, dataset


def bit_view(values):
    """Raw IEEE-754 bits of a host array, as unsigned integers of the
    matching width. **Never a tolerance**: float32 through ``uint32``,
    float64 through ``uint64``, each compared only against itself."""
    if values.dtype == np.float32:
        return values.view(np.uint32).tolist()
    assert values.dtype == np.float64, values.dtype
    return values.view(np.uint64).tolist()


def world(loader):
    """Every publicly observable fact about a loader, its sampler, and its
    dataset, as one comparable value.

    Used before and after every operation that must change nothing. It
    reaches the permutation cache only through its *observable behavior*
    (the planning results), because the cache is not state (§7.8), and it
    reaches the transaction and the iterator slot through the private
    bookkeeping, because "no record was left behind" is exactly what a
    rejected load must guarantee.
    """
    sampler = loader.sampler
    dataset = loader.dataset
    return (
        id(loader), id(sampler), id(dataset),
        loader.closed, dataset.closed,
        sampler.batch_size, sampler.shuffle, sampler.seed, sampler.drop_last,
        sampler.epoch, sampler.cursor,
        sampler.batches_per_epoch, sampler.remaining,
        sampler.next_batch_indices(),
        sampler.epoch_permutation(),
        sampler.plan(),
        json.dumps(sampler.state_dict(), sort_keys=True),
        json.dumps(loader.state_dict(), sort_keys=True),
        loader._iterator is None,
        sampler._transaction is None,
        frozenset(sampler._active_iterations),
    )


def reentrant_probe(monkeypatch, loader, probe, phase="pending"):
    """Run ``probe()`` from *inside* a live batch transaction, at the
    requested phase, and return what it collected.

    The probe runs on the calling thread, so a reentrant arrival is real
    rather than simulated — the established Phase-G/J3 technique, using
    the private ``_deliver_batch`` seam and no public hook.
    """
    collected = []

    if phase == "pending":
        def at_seam(record):
            collected.append(probe())
            return record._features, record._targets

        monkeypatch.setattr(loader_module, "_deliver_batch", at_seam)
    else:
        original = NativeTensorDataset.feature_batch

        def at_claim(self, indices):
            collected.append(probe())
            return original(self, indices)

        monkeypatch.setattr(NativeTensorDataset, "feature_batch", at_claim)
    features, _ = next(iter(loader))
    features.close()
    assert len(collected) == 1, "the reentrant probe never ran"
    return collected[0]


def code_identifiers(relative):
    """Every identifier a module's **executable code** names.

    A source-text scan would be wrong here: these modules explain at
    length what they deliberately do *not* do, so a prose mention of
    ``native_checkpoint`` would fail a substring check that is supposed to
    be about behavior. Reading the AST asks the question that was meant.
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


def imported_pairs(relative):
    """``{(module, name)}`` for every import in a module."""
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    pairs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pairs.update(("", alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            pairs.update((node.module or "", alias.name)
                         for alias in node.names)
    return pairs


def test_the_identifier_scanner_can_actually_find_something():
    """Negative control for every source scan below."""
    names = code_identifiers(LOADER_SOURCE)
    assert "NativeDataLoader" in names
    assert "state_dict" in names
    assert "load_state_dict" in names
    assert "_FORMAT" in names
    assert "native_checkpoint" not in names
    assert imported_pairs(LOADER_SOURCE), "the import scanner found nothing"


# ===========================================================================
# 1. The public method surface — two methods, and no new name
# ===========================================================================

def test_the_loader_gained_exactly_state_dict_and_load_state_dict():
    assert hasattr(NativeDataLoader, "state_dict")
    assert hasattr(NativeDataLoader, "load_state_dict")
    assert callable(NativeDataLoader.state_dict)
    assert callable(NativeDataLoader.load_state_dict)
    # The exact signatures. ``state_dict`` takes nothing — no ``strict``,
    # no ``prefix``, no ``keep_vars``, no destination — and
    # ``load_state_dict`` takes exactly the state.
    assert list(inspect.signature(
        NativeDataLoader.state_dict).parameters) == ["self"]
    assert list(inspect.signature(
        NativeDataLoader.load_state_dict).parameters) == ["self", "state"]


@pytest.mark.parametrize("alias", [
    "state", "load_state", "save", "restore", "save_state", "dump",
    "to_dict", "from_dict", "serialize", "deserialize", "checkpoint",
    "metadata", "__len__", "__next__", "reset", "advance", "step",
    "set_cursor", "set_epoch", "seek", "epoch", "cursor", "seed",
    "batch_size", "shuffle", "drop_last", "validate", "validate_state",
    "_validate_state_dict",
])
def test_no_alias_or_public_mutator_arrived_beside_them(alias):
    """J4 adds two methods and nothing else: no second spelling of either,
    no public cursor or epoch setter, no reset, no public validator, and
    no checkpoint or metadata convenience."""
    assert not hasattr(NativeDataLoader, alias), alias


def test_the_public_loader_surface_is_exactly_what_the_contract_names():
    """§3.5 after J4: three properties, iteration, the two state methods,
    close, the context manager, and the repr — and nothing else public."""
    public = {name for name in dir(NativeDataLoader)
              if not name.startswith("_")}
    assert public == {"sampler", "dataset", "closed", "close",
                      "state_dict", "load_state_dict"}


def test_j4_added_no_public_experimental_name():
    """The J4 invariant over the live inventory: ``__all__`` is **still**
    25 names, exactly J3's set. The export delta is zero."""
    post_j3 = {
        "NativeTensor", "NativeGenerator", "NativeParameter",
        "NativeParameterRegistry", "NativeModule", "NativeLinear",
        "NativeReLU", "NativeFlatten", "NativeConv2d", "NativeMaxPool2d",
        "NativeSequential", "NativeLayerNorm", "NativeBatchNorm1d",
        "NativeBatchNorm2d", "NativeDropout", "NativeMSELoss",
        "NativeCrossEntropyLoss", "native_accuracy", "NativeSGD",
        "NativeAdam", "save_native_checkpoint", "load_native_checkpoint",
        "NativeTensorDataset", "NativeBatchSampler", "NativeDataLoader",
    }
    assert len(post_j3) == 25
    live = set(experimental.__all__)
    assert len(experimental.__all__) == len(live), "duplicate export"
    assert live == post_j3
    assert len(experimental.__all__) == 25


def test_the_loader_state_constants_stay_private():
    """§11.3's tag and version are module constants, not a registry."""
    assert loader_module._FORMAT == LOADER_FORMAT
    assert loader_module._FORMAT_VERSION == 1
    assert loader_module._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert set(loader_module._STATE_FIELDS) == LOADER_ROOT_KEYS
    assert len(loader_module._STATE_FIELDS) == 3
    # There is exactly one supported version, no alias tag, no migration
    # path, and no placeholder for a second one.
    assert 2 not in loader_module._SUPPORTED_FORMAT_VERSIONS
    for name in ("_FORMAT", "_FORMAT_VERSION", "_SUPPORTED_FORMAT_VERSIONS",
                 "_STATE_FIELDS", "_require_exact_int", "_require_exact_keys"):
        assert name not in experimental.__all__, name
        assert not hasattr(experimental, name), name
        assert not hasattr(tensorforge, name), name
    for name in ("NativeLoaderState", "LoaderState", "NativeDataLoaderState",
                 "loader_state", "load_loader_state"):
        assert not hasattr(experimental, name), name
        assert name not in experimental.__all__, name


def test_no_new_public_class_was_defined_anywhere_under_src():
    """A tagged wrapper is a plain dict, not a type. No state class."""
    invented = re.compile(
        r"^\s*class \w*(LoaderState|StateWrapper|DataLoaderState)\b", re.M)
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not invented.search(text), path.name
    # Negative control: the scanner really does find such a class.
    assert invented.search("class NativeLoaderState:\n    pass\n")


def test_the_sampler_schema_and_version_did_not_move():
    """§11.2 is unchanged by the wrapper landing around it."""
    assert sampler_module._FORMAT == SAMPLER_FORMAT
    assert sampler_module._FORMAT_VERSION == 1
    assert sampler_module._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert set(sampler_module._STATE_FIELDS) == SAMPLER_ROOT_KEYS
    assert set(sampler_module._DATASET_FIELDS) == DATASET_KEYS


def test_the_loader_module_imports_only_what_the_contract_allows():
    """One module, three names — the class it iterates and the two
    schema-shaped rules it shares rather than restates. No ctypes, no
    backends package, no NumPy, no json, no checkpoint, no generator, no
    threading, no queue."""
    assert imported_pairs(LOADER_SOURCE) == {
        ("native_sampler", "NativeBatchSampler"),
        ("native_sampler", "_require_exact_int"),
        ("native_sampler", "_require_exact_keys"),
    }
    names = code_identifiers(LOADER_SOURCE)
    for forbidden in ("json", "numpy", "np", "ctypes", "backends", "threading",
                      "Lock", "queue", "asyncio", "multiprocessing", "random",
                      "os", "time", "pickle", "hashlib"):
        assert forbidden not in names, forbidden


# ===========================================================================
# 2. The exact schema (§11.3)
# ===========================================================================

def test_the_loader_state_is_a_three_key_tagged_wrapper():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=20240612)
    state = loader.state_dict()
    assert type(state) is dict
    assert set(state) == LOADER_ROOT_KEYS
    assert len(state) == 3
    assert state["format"] == LOADER_FORMAT
    assert type(state["format"]) is str
    assert state["format_version"] == 1
    assert type(state["format_version"]) is int
    assert type(state["format_version"]) is not bool
    assert type(state["sampler"]) is dict
    dataset.close()


def test_the_nested_object_is_exactly_the_sampler_state():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7, drop_last=True)
    state = loader.state_dict()
    assert state["sampler"] == sampler.state_dict()
    inner = state["sampler"]
    assert set(inner) == SAMPLER_ROOT_KEYS
    assert inner["format"] == SAMPLER_FORMAT
    assert inner["format_version"] == 1
    assert set(inner["dataset"]) == DATASET_KEYS
    assert inner["dataset"]["samples"] == 8
    assert inner["dataset"]["feature_shape"] == [2]
    assert type(inner["dataset"]["feature_shape"]) is list
    assert inner["dataset"]["feature_dtype"] == "float64"
    assert re.fullmatch(r"[0-9a-f]{64}", inner["dataset"]["fingerprint"])
    assert inner["seed"] == 7
    assert inner["shuffle"] is True
    assert inner["batch_size"] == 3
    assert inner["drop_last"] is True
    assert inner["epoch"] == 0
    assert inner["cursor"] == 0
    dataset.close()


def test_no_sampler_field_is_duplicated_at_the_loader_root():
    """The loader owns no epoch, cursor, seed, shuffle, batch size, or
    drop-last of its own, so a second copy could disagree with the first."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    state = loader.state_dict()
    for owned_by_the_sampler in ("dataset", "seed", "shuffle", "batch_size",
                                 "drop_last", "epoch", "cursor",
                                 "batches_per_epoch", "remaining"):
        assert owned_by_the_sampler not in state, owned_by_the_sampler
    dataset.close()


def test_every_container_is_fresh_at_every_call():
    """§11.3: nothing mutable is shared with the loader, the sampler, the
    dataset, the permutation cache, or a previous result."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    sampler.epoch_permutation()               # populate the cache
    first = loader.state_dict()
    second = loader.state_dict()
    assert first == second
    assert first is not second
    assert first["sampler"] is not second["sampler"]
    assert first["sampler"]["dataset"] is not second["sampler"]["dataset"]
    assert (first["sampler"]["dataset"]["feature_shape"]
            is not second["sampler"]["dataset"]["feature_shape"])
    # It shares nothing with the sampler's own snapshot either.
    inner = sampler.state_dict()
    assert first["sampler"] is not inner
    assert first["sampler"]["dataset"] is not inner["dataset"]
    # ...and editing what a caller was given reaches nothing.
    first["sampler"]["cursor"] = 99
    first["sampler"]["dataset"]["samples"] = 99
    first["sampler"]["dataset"]["feature_shape"].append(99)
    assert loader.state_dict() == second
    assert sampler.cursor == 0
    assert dataset.samples == 8
    assert dataset.feature_shape == (2,)
    dataset.close()


def test_the_state_survives_a_json_round_trip_with_semantic_equality():
    for kwargs in ({"batch_size": 3, "shuffle": True, "seed": 2**64 - 1},
                   {"batch_size": 4, "shuffle": False, "seed": 0,
                    "drop_last": True}):
        loader, sampler, dataset = make_loader(8, **kwargs)
        state = loader.state_dict()
        restored = json.loads(json.dumps(state))
        assert restored == state
        assert restored is not state
        inner = restored["sampler"]
        # Booleans stay booleans and the full unsigned 64-bit seed survives.
        assert inner["shuffle"] is kwargs["shuffle"]
        assert inner["seed"] == kwargs["seed"]
        assert type(inner["dataset"]["feature_shape"]) is list
        # ...and a round-tripped state loads without normalization.
        assert loader.load_state_dict(restored) is None
        assert loader.state_dict() == state
        dataset.close()


def test_the_state_is_accepted_unchanged_by_the_checkpoint_metadata_validator():
    """Compatibility evidence only: the state is plain JSON-compatible
    metadata that the **existing** validator already accepts. J5 owns the
    archive workflow, and no checkpoint code is changed here."""
    loader, sampler, dataset = make_loader(800, batch_size=16, shuffle=True,
                                           seed=20240612)
    state = loader.state_dict()
    state["sampler"]["epoch"] = 3
    state["sampler"]["cursor"] = 27
    loader.load_state_dict(state)
    carried = {"training": {"next_step": 12, "data_loader": state}}
    normalized = native_checkpoint._validated_metadata(
        carried, "metadata", frozenset())
    assert normalized == carried
    inner = normalized["training"]["data_loader"]["sampler"]
    assert inner["shuffle"] is True and inner["drop_last"] is False
    assert inner["epoch"] == 3 and inner["cursor"] == 27
    # ...and the validated copy is still a valid loader state.
    assert loader.load_state_dict(
        normalized["training"]["data_loader"]) is None
    assert loader.state_dict() == state
    dataset.close()


def test_the_state_carries_no_payload_permutation_or_process_local_value():
    """§11.4: no dataset content, no NumPy object, no bytes, no id, no
    address, no callable — and **no permutation**, which is derivable."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    state = loader.state_dict()
    order = sampler.epoch_permutation()

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert type(key) is str, key
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert type(value) in (int, bool, str, float), type(value)
            assert not isinstance(value, np.generic)
            assert not isinstance(value, (NativeTensor, np.ndarray, bytes))
            assert not callable(value)

    walk(state)
    text = json.dumps(state)
    assert str(list(order)) not in text
    assert str(order) not in text
    for forbidden in ("permutation", "cache", "0x", "object at", "ndarray",
                      "NativeTensor", "serial", "transaction", "token",
                      "iterator", "generator"):
        assert forbidden not in text, forbidden
    for process_local in (id(loader), id(sampler), id(dataset)):
        assert str(process_local) not in text
    dataset.close()


def test_the_state_size_does_not_grow_with_the_number_of_samples():
    """§11.4: nothing in the state is proportional to the dataset. The two
    encodings differ only by the digits of ``samples`` itself."""
    small, _, small_dataset = make_loader(8, batch_size=2, shuffle=True)
    large, _, large_dataset = make_loader(4000, batch_size=2, shuffle=True)
    small_text = json.dumps(small.state_dict(), sort_keys=True)
    large_text = json.dumps(large.state_dict(), sort_keys=True)
    assert abs(len(large_text) - len(small_text)) <= 8, (
        len(small_text), len(large_text))
    small_dataset.close()
    large_dataset.close()


def test_the_state_reports_both_dtypes_without_inference():
    for dtype, expected in ((None, "float64"), ("float64", "float64"),
                            ("float32", "float32")):
        dataset = make_dataset(8, dtype=dtype)
        loader = NativeDataLoader(NativeBatchSampler(dataset, batch_size=3))
        assert (loader.state_dict()["sampler"]["dataset"]["feature_dtype"]
                == expected)
        dataset.close()
    # ...and a float32 NumPy input with no dtype still says float64.
    features = np.arange(16, dtype=np.float32).reshape(8, 2)
    targets = np.arange(8, dtype=np.int64) % 3
    dataset = NativeTensorDataset(features, targets)
    loader = NativeDataLoader(NativeBatchSampler(dataset, batch_size=3))
    assert (loader.state_dict()["sampler"]["dataset"]["feature_dtype"]
            == "float64")
    dataset.close()


def test_a_scalar_sample_dataset_emits_an_empty_feature_shape():
    dataset = make_dataset(6, rank=1)
    loader = NativeDataLoader(NativeBatchSampler(dataset, batch_size=2))
    shape = loader.state_dict()["sampler"]["dataset"]["feature_shape"]
    assert shape == []
    assert type(shape) is list
    assert loader.load_state_dict(loader.state_dict()) is None
    dataset.close()


# ===========================================================================
# 3. State purity, and where it is allowed (§9.5, §12.5, §15.3)
# ===========================================================================

def test_state_dict_is_pure():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    sampler.epoch_permutation()
    before = world(loader)
    results = [loader.state_dict() for _ in range(4)]
    assert world(loader) == before
    assert (sampler.epoch, sampler.cursor) == (0, 0)
    for result in results[1:]:
        assert result == results[0]
        assert result is not results[0]
    dataset.close()


@needs_backend
def test_state_dict_allocates_nothing_native(live_storages):
    baseline = settled(live_storages)
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True)
    for _ in range(5):
        loader.state_dict()
    assert settled(live_storages) == baseline
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


def test_state_dict_works_with_a_closed_dataset():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    before = loader.state_dict()
    dataset.close()
    assert loader.state_dict() == before
    assert loader.load_state_dict(before) is None
    assert loader.state_dict() == before


def test_state_dict_works_after_the_loader_is_closed():
    """§15.3: a caller may close a loader and then record where it
    stopped. Reading a position is not a resource operation, and it must
    not reopen anything or create an iterator."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    loader.close()
    assert loader.closed is True
    state = loader.state_dict()
    assert set(state) == LOADER_ROOT_KEYS
    assert state["sampler"] == sampler.state_dict()
    assert loader.closed is True
    assert loader._iterator is None
    assert loader.state_dict() == state
    # ...and there is no reopen method.
    for absent in ("open", "reopen", "reset"):
        assert not hasattr(loader, absent), absent
    dataset.close()


@needs_backend
def test_state_dict_after_close_allocates_nothing_and_reopens_nothing(
        live_storages):
    baseline = settled(live_storages)
    loader, sampler, dataset = make_loader(8, batch_size=3)
    features, _ = next(iter(loader))
    features.close()
    loader.close()
    dataset.close()
    for _ in range(3):
        loader.state_dict()
    assert loader.closed is True
    assert settled(live_storages) == baseline


@needs_backend
def test_state_dict_between_batches_describes_the_exact_next_batch():
    """§9.5: allowed mid-iteration whenever no transaction is in flight,
    and it describes the position after the last **successfully
    delivered** batch — which is always the next one."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    plan = sampler.plan()
    iterator = iter(loader)
    for index in range(len(plan)):
        state = loader.state_dict()
        assert state["sampler"] == sampler.state_dict()
        assert state["sampler"]["cursor"] == index
        captured = sampler.next_batch_indices()
        assert captured == plan[index]
        countdown = iterator._to_yield
        active = frozenset(sampler._active_iterations)
        # Taking the snapshot changes no countdown and no participation.
        loader.state_dict()
        assert iterator._to_yield == countdown
        assert frozenset(sampler._active_iterations) == active
        features, targets = next(iterator)
        assert len(targets) == len(captured)
        features.close()
    assert loader.state_dict()["sampler"]["epoch"] == 1
    assert loader.state_dict()["sampler"]["cursor"] == 0
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
def test_state_dict_is_allowed_after_exhaustion_and_after_supersession():
    loader, sampler, dataset = make_loader(6, batch_size=3)
    iterator = iter(loader)
    for features, _ in iterator:
        features.close()
    exhausted = loader.state_dict()
    assert exhausted["sampler"]["epoch"] == 1
    assert exhausted["sampler"]["cursor"] == 0
    first = iter(loader)
    second = iter(loader)                       # supersedes ``first``
    assert first._superseded is True
    assert loader.state_dict() == exhausted
    first.close()
    second.close()
    assert loader.state_dict() == exhausted
    loader.close()
    dataset.close()


@needs_backend
@pytest.mark.parametrize("phase", ["claim", "pending"])
def test_state_dict_is_refused_while_a_transaction_is_in_flight(monkeypatch,
                                                                phase):
    """§9.5: inside the commit-before-delivery window there is no honest
    answer, so the operation refuses rather than captures one."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)

    def probe():
        assert sampler._transaction is not None
        with pytest.raises(RuntimeError):
            loader.state_dict()
        with pytest.raises(RuntimeError):
            sampler.state_dict()
        return sampler._transaction.status

    status = reentrant_probe(monkeypatch, loader, probe, phase=phase)
    assert status == ("claim" if phase == "claim" else "committed")
    loader.close()
    dataset.close()


@needs_backend
def test_no_snapshot_can_observe_a_skipped_but_undelivered_position(
        monkeypatch):
    """The reason the refusal exists: at the pending phase the candidate
    position **has** been applied, so an answer would either expose a
    cursor that skipped an undelivered batch or contradict the object's
    own fields."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    observed = []

    def probe():
        # The raw fields already show the candidate...
        observed.append((sampler.epoch, sampler.cursor))
        # ...and yet no snapshot may report it.
        with pytest.raises(RuntimeError):
            loader.state_dict()
        return None

    reentrant_probe(monkeypatch, loader, probe, phase="pending")
    assert observed == [(0, 1)], observed
    # After the delivery completed, the same position is reportable —
    # because now it is true.
    assert loader.state_dict()["sampler"]["cursor"] == 1
    loader.close()
    dataset.close()


# ===========================================================================
# 4. Load validation and its exact ordering (§12.5)
# ===========================================================================

@pytest.mark.parametrize("malformed", [None, [], (), "state", 7,
                                       {"bad": "state"}, object()])
def test_a_closed_loader_refuses_before_it_inspects_the_state(malformed):
    """§12.5 step 1: the closed guard runs **first**, so even a state that
    could never parse produces the lifecycle error — and the loader is
    never silently reopened."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    before = loader.state_dict()
    loader.close()
    with pytest.raises(RuntimeError):
        loader.load_state_dict(malformed)
    assert loader.closed is True
    assert loader.state_dict() == before
    assert (sampler.epoch, sampler.cursor) == (0, 0)
    assert loader._iterator is None
    dataset.close()


@needs_backend
def test_a_closed_loader_refuses_even_its_own_valid_state(live_storages):
    baseline = settled(live_storages)
    loader, sampler, dataset = make_loader(8, batch_size=3)
    features, _ = next(iter(loader))
    features.close()
    state = loader.state_dict()
    loader.close()
    with pytest.raises(RuntimeError):
        loader.load_state_dict(state)
    assert loader.closed is True
    assert loader.state_dict() == state
    assert settled(live_storages) == baseline
    dataset.close()


@needs_backend
@pytest.mark.parametrize("phase", ["claim", "pending"])
@pytest.mark.parametrize("malformed", [None, [], {"bad": "state"}])
def test_a_live_transaction_refuses_before_it_inspects_the_state(
        monkeypatch, phase, malformed):
    """§12.5 step 2: the transaction guard is second, and it comes before
    the iteration guard because it is the more specific and more dangerous
    condition. Probed with malformed input, so the precedence is proved."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)

    def probe():
        record = sampler._transaction
        serial, status = record.serial, record.status
        with pytest.raises(RuntimeError) as excinfo:
            loader.load_state_dict(malformed)
        # An iterator is necessarily active here too, so the two guards
        # are told apart by *which* one answered rather than by type: the
        # transaction is the more specific condition and must win. Only
        # the distinguishing word is asserted; no message is a contract.
        assert "transaction" in str(excinfo.value)
        # Nothing was disturbed: the record, its status, the batch it owns,
        # the iterator slot, and the participation are all untouched.
        assert sampler._transaction is record
        assert (record.serial, record.status) == (serial, status)
        assert loader._iterator is not None
        assert sampler._active_iterations
        return (sampler.epoch, sampler.cursor)

    reentrant_probe(monkeypatch, loader, probe, phase=phase)
    assert (sampler.epoch, sampler.cursor) == (0, 1)
    # ...and with the transaction gone but an iterator still active, the
    # *other* guard answers, which is what makes the ordering observable.
    iterator = iter(loader)
    with pytest.raises(RuntimeError) as iterating:
        loader.load_state_dict(malformed)
    assert "iterator" in str(iterating.value)
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
@pytest.mark.parametrize("malformed", [None, [], {"bad": "state"}, 7])
def test_an_active_iteration_refuses_before_it_inspects_the_state(malformed):
    """§12.5 step 3, and §9.3's reason: an iterator captured its epoch's
    remaining batch count when it was created, so replacing the position
    underneath it would strand that countdown."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    iterator = iter(loader)
    before = world(loader)
    with pytest.raises(RuntimeError):
        loader.load_state_dict(malformed)
    assert world(loader) == before
    # ...and it holds before the iterator's first batch too, not only
    # between batches.
    features, _ = next(iterator)
    features.close()
    with pytest.raises(RuntimeError):
        loader.load_state_dict(malformed)
    assert (sampler.epoch, sampler.cursor) == (0, 1)
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
def test_a_valid_state_is_refused_while_any_iterator_is_active():
    """Not only malformed input: a perfectly valid state is refused too,
    for the lifecycle reason rather than a schema one."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    state = loader.state_dict()
    iterator = iter(loader)
    with pytest.raises(RuntimeError):
        loader.load_state_dict(state)
    iterator.close()
    assert loader.load_state_dict(state) is None
    loader.close()
    dataset.close()


@needs_backend
@pytest.mark.parametrize("release", ["exhaust", "close", "loader-close"])
def test_loading_becomes_legal_once_participation_is_released(release):
    loader, sampler, dataset = make_loader(6, batch_size=3, shuffle=True,
                                           seed=7)
    source = loader.state_dict()
    source["sampler"]["epoch"] = 4
    iterator = iter(loader)
    with pytest.raises(RuntimeError):
        loader.load_state_dict(source)
    if release == "exhaust":
        for features, _ in iterator:
            features.close()
        assert sampler._active_iterations == set()
        assert loader.load_state_dict(source) is None
    elif release == "close":
        iterator.close()
        assert sampler._active_iterations == set()
        assert loader.load_state_dict(source) is None
    else:
        loader.close()
        assert sampler._active_iterations == set()
        # A closed loader still refuses — for the closed reason now.
        with pytest.raises(RuntimeError):
            loader.load_state_dict(source)
    dataset.close()


@needs_backend
def test_no_state_operation_touches_a_delivered_batch(live_storages):
    """§9.4 Phase 6/§15.6: after the seam returns, the batch is the
    caller's and no loader path retains or can reach it. A snapshot, a
    rejected load, a successful load, and a close must all leave a
    delivered batch open and byte-identical."""
    baseline = settled(live_storages)
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    iterator = iter(loader)
    features, targets = next(iterator)
    iterator.close()
    values = features.to_numpy().copy()
    held = targets.copy()
    live_after_delivery = settled(live_storages)
    assert live_after_delivery == baseline + 1

    loader.state_dict()
    bad = loader.state_dict()
    bad["sampler"]["cursor"] = 99
    with pytest.raises(ValueError):
        loader.load_state_dict(bad)
    good = loader.state_dict()
    good["sampler"]["cursor"] = 0
    assert loader.load_state_dict(good) is None
    loader.close()
    loader.state_dict()

    # The delivered batch is untouched and still open.
    assert bit_view(features.to_numpy()) == bit_view(values)
    assert targets.tolist() == held.tolist()
    assert settled(live_storages) == live_after_delivery
    features.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_superseded_iterator_blocks_the_load_until_it_releases():
    """§9.5: "any iterator participation" includes a superseded iterator
    that has not yet closed, exhausted, or been finalized."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    state = loader.state_dict()
    first = iter(loader)
    second = iter(loader)
    assert first._superseded is True
    assert len(sampler._active_iterations) == 2
    with pytest.raises(RuntimeError):
        loader.load_state_dict(state)
    second.close()
    assert len(sampler._active_iterations) == 1
    with pytest.raises(RuntimeError):
        loader.load_state_dict(state)
    first.close()
    assert sampler._active_iterations == set()
    assert loader.load_state_dict(state) is None
    loader.close()
    dataset.close()


@pytest.mark.parametrize("wrong", [
    None, [], (), "state", 7, 3.5, True, object(), {"format"},
])
def test_the_state_container_must_be_a_dict(wrong):
    loader, sampler, dataset = make_loader(8)
    with pytest.raises(TypeError):
        loader.load_state_dict(wrong)
    dataset.close()


def test_a_mapping_that_is_not_a_dict_is_rejected():
    """Exact-type discipline, matching §11.5's no-normalization rule: an
    ``OrderedDict``, a ``dict`` subclass, and a mapping proxy are refused
    rather than converted."""
    import collections
    import types

    class Subclass(dict):
        pass

    loader, sampler, dataset = make_loader(8)
    state = loader.state_dict()
    for wrong in (collections.OrderedDict(state), Subclass(state),
                  types.MappingProxyType(state), json.dumps(state)):
        with pytest.raises(TypeError):
            loader.load_state_dict(wrong)
    dataset.close()


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda s: s.pop("format"), id="missing-format"),
    pytest.param(lambda s: s.pop("format_version"), id="missing-version"),
    pytest.param(lambda s: s.pop("sampler"), id="missing-sampler"),
    pytest.param(lambda s: s.update(extra=1), id="unexpected-key"),
    pytest.param(lambda s: s.update(strict=False), id="no-strict-flag"),
    pytest.param(lambda s: s.update(permutation=[0, 1]), id="no-permutation"),
    pytest.param(lambda s: s.update(epoch=1), id="no-root-epoch"),
    pytest.param(lambda s: s.update(cursor=0), id="no-root-cursor"),
    pytest.param(lambda s: s.update(seed=7), id="no-root-seed"),
    pytest.param(lambda s: s.update(batch_size=3), id="no-root-batch-size"),
    pytest.param(lambda s: s.update(dataset={}), id="no-root-dataset"),
    pytest.param(lambda s: s.update(loader={}), id="no-nested-loader"),
])
def test_the_root_key_set_is_exact_in_both_directions(mutate):
    loader, sampler, dataset = make_loader(8)
    state = loader.state_dict()
    mutate(state)
    with pytest.raises(ValueError):
        loader.load_state_dict(state)
    dataset.close()


def test_the_key_set_message_names_missing_and_unexpected_deterministically():
    """Not a message contract: only that both halves are reported and that
    the report is sorted, so a caller sees a deterministic answer."""
    loader, sampler, dataset = make_loader(8)
    state = loader.state_dict()
    state.pop("sampler")
    state["zeta"] = 1
    state["alpha"] = 2
    with pytest.raises(ValueError) as excinfo:
        loader.load_state_dict(state)
    message = str(excinfo.value)
    assert "sampler" in message
    assert message.index("'alpha'") < message.index("'zeta'")
    dataset.close()


@pytest.mark.parametrize("value,expected", [
    (7, TypeError), (None, TypeError), (b"x", TypeError), (1.0, TypeError),
    ([LOADER_FORMAT], TypeError),
    (SAMPLER_FORMAT, ValueError),
    ("tensorforge.native_checkpoint", ValueError),
    ("tensorforge.native_dataset", ValueError),
    ("", ValueError),
    ("TENSORFORGE.NATIVE_DATA_LOADER", ValueError),
    ("Tensorforge.Native_Data_Loader", ValueError),
    ("tensorforge.native_data_loader ", ValueError),
])
def test_the_loader_format_tag_is_validated_by_type_then_by_value(value,
                                                                  expected):
    loader, sampler, dataset = make_loader(8)
    state = loader.state_dict()
    state["format"] = value
    with pytest.raises(expected):
        loader.load_state_dict(state)
    dataset.close()


def test_a_str_subclass_is_not_the_format_tag():
    class Tag(str):
        pass

    loader, sampler, dataset = make_loader(8)
    state = loader.state_dict()
    state["format"] = Tag(LOADER_FORMAT)
    with pytest.raises(TypeError):
        loader.load_state_dict(state)
    dataset.close()


@pytest.mark.parametrize("value,expected", [
    (True, TypeError), (False, TypeError), (1.0, TypeError), ("1", TypeError),
    (np.int64(1), TypeError), (np.uint64(1), TypeError), (None, TypeError),
    ([1], TypeError),
    (0, ValueError), (2, ValueError), (-1, ValueError), (4, ValueError),
    (2**64, ValueError),
])
def test_the_loader_format_version_is_validated_by_type_then_by_value(
        value, expected):
    loader, sampler, dataset = make_loader(8)
    state = loader.state_dict()
    state["format_version"] = value
    with pytest.raises(expected):
        loader.load_state_dict(state)
    dataset.close()


@pytest.mark.parametrize("value", [
    None, [], "x", 7, ((1, 2),), True,
])
def test_the_nested_sampler_object_must_be_a_dict(value):
    loader, sampler, dataset = make_loader(8)
    state = loader.state_dict()
    state["sampler"] = value
    with pytest.raises(TypeError):
        loader.load_state_dict(state)
    dataset.close()


def test_a_sampler_object_is_not_a_sampler_state():
    """A live object where a plain dict is required is a ``TypeError``,
    not something to introspect."""
    loader, sampler, dataset = make_loader(8)
    for wrong in (sampler, loader, dataset, json.dumps(sampler.state_dict())):
        state = loader.state_dict()
        state["sampler"] = wrong
        with pytest.raises(TypeError):
            loader.load_state_dict(state)
    dataset.close()


@pytest.mark.parametrize("mutate,expected", [
    pytest.param(lambda s: s["sampler"].pop("cursor"), ValueError,
                 id="nested-missing-key"),
    pytest.param(lambda s: s["sampler"].update(extra=1), ValueError,
                 id="nested-extra-key"),
    pytest.param(lambda s: s["sampler"].__setitem__("format", 7), TypeError,
                 id="nested-format-type"),
    pytest.param(lambda s: s["sampler"].__setitem__("format", LOADER_FORMAT),
                 ValueError, id="nested-format-value"),
    pytest.param(lambda s: s["sampler"].__setitem__("format_version", True),
                 TypeError, id="nested-version-type"),
    pytest.param(lambda s: s["sampler"].__setitem__("format_version", 2),
                 ValueError, id="nested-version-value"),
    pytest.param(lambda s: s["sampler"].__setitem__("dataset", None),
                 TypeError, id="nested-dataset-type"),
    pytest.param(lambda s: s["sampler"]["dataset"].pop("fingerprint"),
                 ValueError, id="nested-dataset-keys"),
    pytest.param(lambda s: s["sampler"]["dataset"].__setitem__("samples", "8"),
                 TypeError, id="nested-dataset-field-type"),
    pytest.param(lambda s: s["sampler"]["dataset"].__setitem__("samples", 0),
                 ValueError, id="nested-dataset-field-range"),
    pytest.param(
        lambda s: s["sampler"]["dataset"].__setitem__("feature_shape", [0]),
        ValueError, id="nested-shape-range"),
    pytest.param(
        lambda s: s["sampler"]["dataset"].__setitem__("feature_shape", [True]),
        TypeError, id="nested-shape-element-type"),
    pytest.param(
        lambda s: s["sampler"]["dataset"].__setitem__("feature_dtype",
                                                      "float16"),
        ValueError, id="nested-dtype-value"),
    pytest.param(
        lambda s: s["sampler"]["dataset"].__setitem__("fingerprint",
                                                      "A" * 64),
        ValueError, id="nested-uppercase-fingerprint"),
    pytest.param(
        lambda s: s["sampler"]["dataset"].__setitem__("fingerprint",
                                                      "a" * 63),
        ValueError, id="nested-short-fingerprint"),
    pytest.param(lambda s: s["sampler"].__setitem__("seed", -1), ValueError,
                 id="nested-seed-range"),
    pytest.param(lambda s: s["sampler"].__setitem__("seed", 2**64), ValueError,
                 id="nested-seed-above-uint64"),
    pytest.param(lambda s: s["sampler"].__setitem__("seed", True), TypeError,
                 id="nested-seed-type"),
    pytest.param(lambda s: s["sampler"].__setitem__("shuffle", 1), TypeError,
                 id="nested-shuffle-type"),
    pytest.param(lambda s: s["sampler"].__setitem__("batch_size", 0),
                 ValueError, id="nested-batch-size-range"),
    pytest.param(lambda s: s["sampler"].__setitem__("drop_last", 0), TypeError,
                 id="nested-drop-last-type"),
    pytest.param(lambda s: s["sampler"].__setitem__("epoch", -1), ValueError,
                 id="nested-epoch-range"),
    pytest.param(lambda s: s["sampler"].__setitem__("epoch", 2**64),
                 ValueError, id="nested-epoch-above-uint64"),
    pytest.param(lambda s: s["sampler"].__setitem__("cursor", -1), ValueError,
                 id="nested-cursor-negative"),
    pytest.param(lambda s: s["sampler"].__setitem__("cursor", 3), ValueError,
                 id="nested-cursor-equals-count"),
    pytest.param(lambda s: s["sampler"].__setitem__("cursor", 99), ValueError,
                 id="nested-cursor-above-count"),
    pytest.param(lambda s: [s["sampler"].__setitem__("batch_size", 99),
                            s["sampler"].__setitem__("drop_last", True)],
                 ValueError, id="nested-zero-batch"),
])
def test_every_nested_sampler_rule_still_applies_through_the_wrapper(
        mutate, expected):
    """The nested validation is **delegated**, not restated — so every
    J2/J3 rule reaches a state that arrived through the loader."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    state = loader.state_dict()
    mutate(state)
    with pytest.raises(expected):
        loader.load_state_dict(state)
    dataset.close()


def test_the_nested_validation_is_delegated_rather_than_duplicated():
    """One authority: ``NativeBatchSampler._validate_state``. The loader
    calls it, and does not re-implement a single one of its rules."""
    source = (REPO_ROOT / LOADER_SOURCE).read_text(encoding="utf-8")
    tree = ast.parse(source)
    loaded = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "load_state_dict")
    called = {node.func.attr for node in ast.walk(loaded)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}
    assert "_validate_state" in called
    assert "_assign_state" in called
    assert "_require_no_transaction" in called
    assert "_require_no_active_iteration" in called
    # The public sampler loader is deliberately **not** called: it would
    # mutate before the wrapper's own transaction completed.
    assert "load_state_dict" not in called
    names = code_identifiers(LOADER_SOURCE)
    # None of the nested schema's vocabulary is restated here. (The loader
    # *reads* ``batches_per_epoch`` for its repr, which is a report rather
    # than a validation rule, so that name is deliberately not listed.)
    for owned_by_the_sampler in ("_DATASET_FIELDS", "_IDENTITY_DTYPES",
                                 "_FINGERPRINT_LENGTH", "_HEX_DIGITS",
                                 "_validate_dataset_identity",
                                 "UINT64_MAX", "_validate_uint64",
                                 "_require_exact_bool", "issuperset"):
        assert owned_by_the_sampler not in names, owned_by_the_sampler
    # ...and neither state method names a nested schema key as a literal,
    # which is what re-implementing one of its rules would require. Asked
    # of the string **constants** rather than of the source text, because
    # this module explains at length what it delegates and a prose mention
    # would fail a substring check that is meant to be about behavior.
    nested_keys = (SAMPLER_ROOT_KEYS | DATASET_KEYS) - {"format",
                                                        "format_version"}
    for name in ("state_dict", "load_state_dict"):
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == name)
        literals = {node.value for node in ast.walk(function)
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)}
        assert literals & nested_keys == set(), (name, literals & nested_keys)
    # Negative control: the same scan finds them in the module that really
    # does own them.
    sampler_tree = ast.parse(
        (REPO_ROOT / SAMPLER_SOURCE).read_text(encoding="utf-8"))
    validator = next(node for node in ast.walk(sampler_tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "_validate_state")
    sampler_literals = {node.value for node in ast.walk(validator)
                        if isinstance(node, ast.Constant)
                        and isinstance(node.value, str)}
    assert sampler_literals & nested_keys, "the literal scan found nothing"


@pytest.mark.parametrize("mutate,expected,reason", [
    pytest.param(lambda s: [s.update(extra=1), s.__setitem__("format", 7)],
                 ValueError, "key set before format type",
                 id="keys-before-format"),
    pytest.param(lambda s: [s.__setitem__("format", "nope"),
                            s.__setitem__("format_version", "1")],
                 ValueError, "format value before version type",
                 id="format-before-version"),
    pytest.param(lambda s: [s.__setitem__("format_version", 9),
                            s.__setitem__("sampler", None)],
                 ValueError, "version value before nested container type",
                 id="version-before-nested"),
    pytest.param(lambda s: [s.__setitem__("sampler", None),
                            s.update(extra=1)],
                 ValueError, "root key set before nested container type",
                 id="root-keys-before-nested"),
    pytest.param(lambda s: [s.__setitem__("sampler", 7),
                            s.__setitem__("format", "nope")],
                 ValueError, "wrapper format before nested container",
                 id="wrapper-format-before-nested"),
    pytest.param(lambda s: [s.__setitem__("sampler", 7)],
                 TypeError, "nested container type",
                 id="nested-container"),
    pytest.param(lambda s: [s["sampler"].__setitem__("cursor", 99),
                            s.__setitem__("format_version", 2)],
                 ValueError, "wrapper version before nested cursor range",
                 id="wrapper-version-before-nested-rules"),
    pytest.param(lambda s: [s["sampler"]["dataset"].__setitem__("samples", 99),
                            s["sampler"].__setitem__("seed", "x")],
                 ValueError, "nested compatibility before nested config type",
                 id="nested-compat-before-config"),
])
def test_the_load_precedence_is_exactly_the_contracted_order(mutate, expected,
                                                             reason):
    """§12.5, probed by exception **type** so no message is a contract."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    state = loader.state_dict()
    mutate(state)
    with pytest.raises(expected):
        loader.load_state_dict(state)
    dataset.close()


@needs_backend
def test_the_guards_outrank_every_schema_rule_together():
    """All four fault classes at once: closed, transacting, iterating, and
    malformed. The most specific lifecycle answer wins each time."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    garbage = {"format": 7, "sampler": None, "extra": 1}
    iterator = iter(loader)
    with pytest.raises(RuntimeError):           # active iteration, not schema
        loader.load_state_dict(garbage)
    iterator.close()
    with pytest.raises(ValueError):             # now the schema answers
        loader.load_state_dict(garbage)
    loader.close()
    with pytest.raises(RuntimeError):           # closed outranks the schema
        loader.load_state_dict(garbage)
    dataset.close()


def test_nothing_is_normalized_coerced_or_repaired():
    """§11.5: ``1`` is not ``True``, ``True`` is not ``1``, ``1.0`` is not
    ``1``, ``"1"`` is not ``1``, a NumPy scalar is not a Python one, a
    missing key is not a default, an unknown key is not ignored, and a
    cursor past the end is not clamped."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    before = world(loader)
    root_cases = [
        ("format_version", True), ("format_version", 1.0),
        ("format_version", "1"), ("format_version", np.int64(1)),
        ("format", b"tensorforge.native_data_loader"),
    ]
    nested_cases = [
        ("shuffle", 1), ("shuffle", 0), ("drop_last", 1),
        ("batch_size", True), ("seed", True), ("epoch", True),
        ("cursor", True), ("batch_size", 3.0), ("seed", "0"),
        ("format_version", 1.0), ("cursor", np.int64(0)),
        ("batch_size", np.int64(3)), ("shuffle", np.bool_(True)),
        ("cursor", 3), ("cursor", 99),
    ]
    for field, value in root_cases:
        state = loader.state_dict()
        state[field] = value
        with pytest.raises((TypeError, ValueError)):
            loader.load_state_dict(state)
    for field, value in nested_cases:
        state = loader.state_dict()
        state["sampler"][field] = value
        with pytest.raises((TypeError, ValueError)):
            loader.load_state_dict(state)
    # A tuple where a dict is required, at both levels.
    for wrong in (tuple(loader.state_dict().items()),):
        with pytest.raises(TypeError):
            loader.load_state_dict(wrong)
    state = loader.state_dict()
    state["sampler"] = tuple(state["sampler"].items())
    with pytest.raises(TypeError):
        loader.load_state_dict(state)
    assert world(loader) == before
    dataset.close()


def test_the_one_documented_container_exception_still_holds():
    """§11.2's single exception: ``dataset.feature_shape`` may arrive as a
    tuple, while emitted state always uses a list. No other normalization
    exception is added."""
    dataset = make_dataset(8, width=3)
    loader = NativeDataLoader(NativeBatchSampler(dataset, batch_size=3))
    state = loader.state_dict()
    state["sampler"]["dataset"]["feature_shape"] = tuple(
        state["sampler"]["dataset"]["feature_shape"])
    assert loader.load_state_dict(state) is None
    emitted = loader.state_dict()["sampler"]["dataset"]["feature_shape"]
    assert type(emitted) is list
    # The wrapper itself takes no such latitude: its root is a dict, full
    # stop, and so is the nested sampler object.
    with pytest.raises(TypeError):
        loader.load_state_dict(tuple(state.items()))
    dataset.close()


# ===========================================================================
# 5. Wrong-wrapper and wrong-object states (§11.3)
# ===========================================================================

def test_a_raw_sampler_state_is_not_a_loader_state():
    """The whole reason the tag exists: without it the two would be the
    same JSON and the confusion would be accepted silently."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    before = world(loader)
    with pytest.raises(ValueError):
        loader.load_state_dict(sampler.state_dict())
    assert world(loader) == before
    dataset.close()


def test_a_loader_state_is_not_a_sampler_state():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    state = loader.state_dict()
    before = world(loader)
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)
    assert world(loader) == before
    dataset.close()


@pytest.mark.parametrize("foreign", [
    pytest.param({"format": "tensorforge.native_checkpoint",
                  "format_version": 3, "sampler": {}}, id="checkpoint-tag"),
    pytest.param({"samples": 8, "feature_shape": [2],
                  "feature_dtype": "float64", "fingerprint": "0" * 64},
                 id="dataset-identity"),
    pytest.param({"format": LOADER_FORMAT, "format_version": 1,
                  "sampler": {}, "future_field": 1}, id="unknown-future-key"),
    pytest.param({"format_version": 1, "sampler": {}}, id="missing-tag"),
    pytest.param({"format": LOADER_FORMAT, "format_version": 2,
                  "sampler": {}}, id="wrong-wrapper-version"),
])
def test_a_foreign_or_future_state_object_is_rejected(foreign):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    before = world(loader)
    with pytest.raises((TypeError, ValueError)):
        loader.load_state_dict(foreign)
    assert world(loader) == before
    dataset.close()


def test_sampler_fields_copied_to_the_loader_root_are_rejected():
    """A "flattened" loader state is not a loader state: the wrapper's key
    set is exact, so duplicated configuration at the root fails."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    flattened = dict(sampler.state_dict())
    flattened["format"] = LOADER_FORMAT
    flattened["sampler"] = sampler.state_dict()
    with pytest.raises(ValueError):
        loader.load_state_dict(flattened)
    # ...and so does a correct wrapper with one field duplicated beside it.
    state = loader.state_dict()
    state["cursor"] = state["sampler"]["cursor"]
    with pytest.raises(ValueError):
        loader.load_state_dict(state)
    dataset.close()


def test_a_native_checkpoint_archive_dict_is_not_a_loader_state():
    """The checkpoint's own manifest shape is refused, in both the
    obvious and the near-miss form."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    for archive in (
        {"format": native_checkpoint._FORMAT,
         "format_version": native_checkpoint._FORMAT_VERSION,
         "sampler": loader.state_dict()["sampler"]},
        {"format": LOADER_FORMAT,
         "format_version": native_checkpoint._FORMAT_VERSION,
         "sampler": loader.state_dict()["sampler"]},
    ):
        with pytest.raises(ValueError):
            loader.load_state_dict(archive)
    dataset.close()


# ===========================================================================
# 6. Dataset compatibility (§6, §12.4 step 8)
# ===========================================================================

def test_each_dataset_mismatch_is_rejected_independently():
    """The four identity fields, each on its own, so a mismatch names one
    understandable difference."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    base = loader.state_dict()

    state = loader.state_dict()
    state["sampler"]["dataset"]["samples"] = 9
    with pytest.raises(ValueError) as samples_error:
        loader.load_state_dict(state)
    assert "9" in str(samples_error.value)

    state = loader.state_dict()
    state["sampler"]["dataset"]["feature_shape"] = [3]
    with pytest.raises(ValueError) as shape_error:
        loader.load_state_dict(state)
    assert "shape" in str(shape_error.value)

    state = loader.state_dict()
    state["sampler"]["dataset"]["feature_dtype"] = "float32"
    with pytest.raises(ValueError) as dtype_error:
        loader.load_state_dict(state)
    assert "dtype" in str(dtype_error.value)

    # The fingerprint is the one that catches "same samples, same shape,
    # same dtype, different data", which structural fields cannot.
    different = NativeTensorDataset(
        np.arange(16, dtype=np.float64).reshape(8, 2) + 1.0,
        np.arange(8, dtype=np.int64) % 3)
    other = NativeDataLoader(NativeBatchSampler(different, batch_size=3))
    foreign = other.state_dict()
    identity = foreign["sampler"]["dataset"]
    assert identity["samples"] == base["sampler"]["dataset"]["samples"]
    assert (identity["feature_shape"]
            == base["sampler"]["dataset"]["feature_shape"])
    assert (identity["feature_dtype"]
            == base["sampler"]["dataset"]["feature_dtype"])
    assert identity["fingerprint"] != base["sampler"]["dataset"]["fingerprint"]
    with pytest.raises(ValueError) as fingerprint_error:
        loader.load_state_dict(foreign)
    assert "fingerprint" in str(fingerprint_error.value)
    assert loader.state_dict() == base
    dataset.close()
    different.close()


def test_the_compatibility_order_is_samples_shape_dtype_then_fingerprint():
    """Structural first, digest last: "the fingerprints differ" is only
    useful once the shapes agree."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    state = loader.state_dict()
    identity = state["sampler"]["dataset"]
    identity["samples"] = 5
    identity["feature_shape"] = [9]
    identity["feature_dtype"] = "float32"
    identity["fingerprint"] = "b" * 64
    with pytest.raises(ValueError) as samples_first:
        loader.load_state_dict(state)
    assert "samples" in str(samples_first.value)
    assert "fingerprint" not in str(samples_first.value)

    identity["samples"] = 8
    with pytest.raises(ValueError) as shape_next:
        loader.load_state_dict(state)
    assert "shape" in str(shape_next.value)
    assert "fingerprint" not in str(shape_next.value)

    identity["feature_shape"] = [2]
    with pytest.raises(ValueError) as dtype_next:
        loader.load_state_dict(state)
    assert "dtype" in str(dtype_next.value)
    assert "fingerprint" not in str(dtype_next.value)

    identity["feature_dtype"] = "float64"
    with pytest.raises(ValueError) as digest_last:
        loader.load_state_dict(state)
    assert "fingerprint" in str(digest_last.value)
    assert (sampler.epoch, sampler.cursor) == (0, 0)
    dataset.close()


def test_a_state_from_an_equal_but_distinct_dataset_loads():
    """Compatibility is decided by the four identity fields, never by
    object identity — which is the whole point of a content fingerprint
    that survives a process boundary."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    twin_dataset = make_dataset(8)
    twin = NativeDataLoader(NativeBatchSampler(twin_dataset, batch_size=2))
    assert twin_dataset is not dataset
    assert (twin.state_dict()["sampler"]["dataset"]
            == loader.state_dict()["sampler"]["dataset"])
    assert loader.load_state_dict(twin.state_dict()) is None
    assert loader.dataset is dataset
    assert loader.dataset is not twin_dataset
    assert sampler.batch_size == 2
    dataset.close()
    twin_dataset.close()


# ===========================================================================
# 7. The transactional rejection guarantee (§12.7, §17.5)
# ===========================================================================

def _broken(mutate):
    """Turn an in-place mutation into a "build the rejected candidate"
    function, so a case that replaces the whole object and a case that
    edits one field are the same shape."""
    def build(state):
        mutate(state)
        return state
    return build


REJECTIONS = [
    pytest.param(lambda s: None, id="not-a-dict-none"),
    pytest.param(lambda s: [("format", LOADER_FORMAT)], id="not-a-dict-list"),
    pytest.param(lambda s: json.dumps(s), id="not-a-dict-json-string"),
    pytest.param(_broken(lambda s: s.pop("format")), id="missing-format"),
    pytest.param(_broken(lambda s: s.update(extra=1)), id="extra-root-key"),
    pytest.param(_broken(lambda s: s.__setitem__("format", SAMPLER_FORMAT)),
                 id="wrong-format"),
    pytest.param(_broken(lambda s: s.__setitem__("format_version", 2)),
                 id="wrong-version"),
    pytest.param(_broken(lambda s: s.__setitem__("format_version", True)),
                 id="bool-version"),
    pytest.param(_broken(lambda s: s.__setitem__("sampler", None)),
                 id="nested-not-a-dict"),
    pytest.param(_broken(lambda s: s["sampler"].pop("epoch")),
                 id="nested-missing-key"),
    pytest.param(_broken(lambda s: s["sampler"].update(extra=1)),
                 id="nested-extra-key"),
    pytest.param(_broken(lambda s: s["sampler"].__setitem__("seed", -1)),
                 id="nested-seed-range"),
    pytest.param(_broken(lambda s: s["sampler"].__setitem__("shuffle", 1)),
                 id="nested-shuffle-type"),
    pytest.param(_broken(lambda s: s["sampler"].__setitem__("cursor", 99)),
                 id="nested-cursor-range"),
    pytest.param(_broken(lambda s: [s["sampler"].__setitem__("batch_size", 99),
                                    s["sampler"].__setitem__("drop_last",
                                                             True)]),
                 id="nested-zero-batch"),
    pytest.param(
        _broken(lambda s: s["sampler"]["dataset"].__setitem__("samples", 99)),
        id="dataset-samples"),
    pytest.param(
        _broken(lambda s: s["sampler"]["dataset"].__setitem__("feature_shape",
                                                              [7])),
        id="dataset-shape"),
    pytest.param(
        _broken(lambda s: s["sampler"]["dataset"].__setitem__("feature_dtype",
                                                              "float32")),
        id="dataset-dtype"),
    pytest.param(
        _broken(lambda s: s["sampler"]["dataset"].__setitem__("fingerprint",
                                                              "c" * 64)),
        id="dataset-fingerprint"),
]


@needs_backend
@pytest.mark.parametrize("build", REJECTIONS)
def test_a_rejected_load_leaves_the_whole_observable_world_unchanged(
        build, live_storages):
    """§12.7/§17.5: nothing is mutated. Every field, the position, the
    configuration, the planning results, the permutation cache's behavior,
    the iterator slot, the transaction record, and the native live-storage
    count are exactly as they were."""
    baseline = settled(live_storages)
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=20240612)
    # Start from a non-origin position, so a partial commit would show.
    moved = loader.state_dict()
    moved["sampler"]["epoch"], moved["sampler"]["cursor"] = 5, 2
    loader.load_state_dict(moved)
    sampler.epoch_permutation()                   # populate the cache
    before = world(loader)
    cached_order = sampler.epoch_permutation()
    cached_plan = sampler.plan()
    live_before = settled(live_storages)

    candidate = build(loader.state_dict())
    with pytest.raises((TypeError, ValueError)):
        loader.load_state_dict(candidate)

    assert world(loader) == before
    assert loader.state_dict() == moved
    assert loader.sampler is sampler
    assert loader.dataset is dataset
    assert loader.closed is False
    assert loader._iterator is None
    assert sampler._transaction is None
    assert sampler._active_iterations == set()
    # The cache is not state, but its **behavior** must be unchanged.
    assert sampler.epoch_permutation() == cached_order
    assert sampler.plan() == cached_plan
    assert settled(live_storages) == live_before == baseline
    # ...and a subsequent good load still works, so nothing was poisoned.
    assert loader.load_state_dict(moved) is None
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_rejected_load_creates_supersedes_and_closes_no_iterator():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    features, _ = next(iterator)
    features.close()
    iterator.close()
    assert loader._iterator is None
    bad = loader.state_dict()
    bad["sampler"]["cursor"] = 99
    with pytest.raises(ValueError):
        loader.load_state_dict(bad)
    assert loader._iterator is None
    assert sampler._active_iterations == set()
    # ...and iteration still works afterwards, from the committed position.
    remaining = [targets.copy() for _, targets in _drain(loader)]
    assert len(remaining) == 2
    loader.close()
    dataset.close()


def _drain(loader):
    """Every batch of one iteration, closing each feature tensor exactly as
    a caller must."""
    for features, targets in loader:
        yield features, targets
        features.close()


# ===========================================================================
# 8. Successful adoption (§12.4, §12.5)
# ===========================================================================

def test_all_six_configuration_and_position_values_are_adopted():
    """§12.4's split, through the wrapper: structural facts about the live
    dataset are validated and never adopted; configuration the state
    carries **is** adopted. The target is deliberately built wrong in
    every one of the six fields, so the proof cannot pass vacuously."""
    source_dataset = make_dataset(12)
    source = NativeDataLoader(NativeBatchSampler(
        source_dataset, batch_size=3, shuffle=True, seed=7, drop_last=False))
    state = source.state_dict()
    state["sampler"]["epoch"], state["sampler"]["cursor"] = 3, 2
    source.load_state_dict(state)

    target_dataset = make_dataset(12)
    target_sampler = NativeBatchSampler(target_dataset, batch_size=5,
                                        shuffle=False, seed=0, drop_last=True)
    target = NativeDataLoader(target_sampler)
    assert (target_sampler.batch_size, target_sampler.shuffle,
            target_sampler.seed, target_sampler.drop_last,
            target_sampler.epoch, target_sampler.cursor) != (
        3, True, 7, False, 3, 2)

    assert target.load_state_dict(state) is None
    assert target_sampler.batch_size == 3
    assert target_sampler.shuffle is True
    assert target_sampler.seed == 7
    assert target_sampler.drop_last is False
    assert target_sampler.epoch == 3
    assert target_sampler.cursor == 2
    assert target.state_dict() == state
    assert (target_sampler.next_batch_indices()
            == source.sampler.next_batch_indices())
    source_dataset.close()
    target_dataset.close()


def test_object_identity_is_preserved_absolutely():
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3)
    loader = NativeDataLoader(sampler)
    loader_id, sampler_id, dataset_id = id(loader), id(sampler), id(dataset)

    other_dataset = make_dataset(8)
    other = NativeDataLoader(NativeBatchSampler(other_dataset, batch_size=2,
                                                shuffle=True, seed=5))
    assert loader.load_state_dict(other.state_dict()) is None
    assert loader.sampler is sampler
    assert loader.dataset is dataset
    assert loader.dataset is not other_dataset
    assert sampler.dataset is dataset
    assert (id(loader), id(sampler), id(dataset)) == (loader_id, sampler_id,
                                                      dataset_id)
    assert loader.closed is False
    assert loader._iterator is None
    assert sampler.seed == 5
    assert sampler.batch_size == 2
    dataset.close()
    other_dataset.close()


def test_a_load_returns_none_and_invalidates_the_cache_correctly():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    first_order = sampler.epoch_permutation()
    state = loader.state_dict()
    state["sampler"]["seed"] = 0xFEDCBA9876543210
    state["sampler"]["epoch"] = 4
    assert loader.load_state_dict(state) is None
    assert sampler._cache_key is None
    assert sampler._cache_order is None
    second_order = sampler.epoch_permutation()
    assert second_order != first_order
    # Recomputing after a dropped cache gives the same answer, because the
    # order is a pure function of (seed, epoch, samples).
    sampler._cache_key = None
    sampler._cache_order = None
    assert sampler.epoch_permutation() == second_order
    dataset.close()


@needs_backend
def test_a_successful_load_allocates_nothing_and_creates_no_iterator(
        live_storages):
    baseline = settled(live_storages)
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True)
    state = loader.state_dict()
    state["sampler"]["cursor"] = 2
    for _ in range(3):
        assert loader.load_state_dict(state) is None
    assert loader._iterator is None
    assert sampler._transaction is None
    assert sampler._active_iterations == set()
    assert settled(live_storages) == baseline
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_next_batch_changes_to_the_restored_candidate():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    plan = sampler.plan()
    assert sampler.next_batch_indices() == plan[0]
    state = loader.state_dict()
    state["sampler"]["cursor"] = 2
    loader.load_state_dict(state)
    assert sampler.next_batch_indices() == plan[2]
    features, targets = next(iter(loader))
    assert features.shape[0] == len(plan[2])
    assert targets.tolist() == [dataset.target_batch(plan[2])[i]
                                for i in range(len(plan[2]))]
    features.close()
    loader.close()
    dataset.close()


# ===========================================================================
# 9. The exact mid-epoch resume proof — J4's exit gate
# ===========================================================================

def _resume_proof(dtype, samples, batch_size, shuffle, seed, drop_last,
                  consumed, live_storages, rank=2):
    """Advance a source loader by ``consumed`` batches, restore its state
    into a **separately constructed** dataset/sampler/loader graph, and
    prove the remaining sequence matches exactly.

    Returns ``(tail_indices, next_epoch_position)`` so a caller can assert
    the interruption really was mid-epoch.
    """
    baseline = settled(live_storages)

    # --- the source graph.
    dataset_a = make_dataset(samples, dtype=dtype, rank=rank)
    sampler_a = NativeBatchSampler(dataset_a, batch_size=batch_size,
                                   shuffle=shuffle, seed=seed,
                                   drop_last=drop_last)
    loader_a = NativeDataLoader(sampler_a)

    # --- the restored graph: separately constructed, identical logical
    #     content, deliberately different valid configuration.
    dataset_b = make_dataset(samples, dtype=dtype, rank=rank)
    sampler_b = NativeBatchSampler(dataset_b, batch_size=1, shuffle=not shuffle,
                                   seed=(seed + 1) % (2**64), drop_last=False)
    loader_b = NativeDataLoader(sampler_b)
    assert dataset_a is not dataset_b
    assert sampler_a is not sampler_b
    assert loader_a is not loader_b
    # Equal through the four compatibility fields, and through nothing
    # process-local.
    assert dataset_a.identity() == dataset_b.identity()

    # --- consume the prefix.
    if consumed:
        iterator = iter(loader_a)
        for _ in range(consumed):
            features, _ = next(iterator)
            features.close()
        iterator.close()

    state = loader_a.state_dict()
    assert loader_b.load_state_dict(state) is None

    # 1. The wrapper round-trips exactly.
    assert loader_b.state_dict() == state
    assert loader_b.state_dict() == loader_a.state_dict()
    # 2. The next batch's indices agree.
    assert sampler_b.next_batch_indices() == sampler_a.next_batch_indices()
    # 3-7. Every remaining batch of the active epoch, index for index, bit
    #      for bit, target for target — plus dtype, shape, device,
    #      contiguity, ownership, and the read-only flag.
    tail = []
    remaining_a = sampler_a.remaining
    assert sampler_b.remaining == remaining_a
    iterator_a, iterator_b = iter(loader_a), iter(loader_b)
    for _ in range(remaining_a):
        indices_a = sampler_a.next_batch_indices()
        indices_b = sampler_b.next_batch_indices()
        assert indices_a == indices_b
        tail.append(indices_a)
        features_a, targets_a = next(iterator_a)
        features_b, targets_b = next(iterator_b)
        host_a, host_b = features_a.to_numpy(), features_b.to_numpy()
        assert host_a.dtype == host_b.dtype
        assert features_a.dtype == features_b.dtype == (dtype or "float64")
        assert features_a.shape == features_b.shape
        assert features_a.device == features_b.device == "cpu"
        assert features_a.contiguous and features_b.contiguous
        assert bit_view(host_a) == bit_view(host_b)
        assert targets_a.dtype == targets_b.dtype == np.int64
        assert targets_a.shape == targets_b.shape
        assert targets_a.tolist() == targets_b.tolist()
        for array in (targets_a, targets_b):
            assert array.flags["C_CONTIGUOUS"]
            assert array.flags["OWNDATA"]
            assert array.flags["WRITEABLE"] is False
        features_a.close()
        features_b.close()
    # Both stop at the same place, and neither restarts an epoch.
    with pytest.raises(StopIteration):
        next(iterator_a)
    with pytest.raises(StopIteration):
        next(iterator_b)
    # 8. The canonical next-epoch position.
    assert loader_a.state_dict() == loader_b.state_dict()
    final = loader_b.state_dict()["sampler"]
    assert final["cursor"] == 0
    next_position = (final["epoch"], final["cursor"])
    # 9. The **following whole epoch** matches too, planned and delivered.
    assert sampler_a.plan() == sampler_b.plan()
    for _ in range(2):
        next_a = [(f.to_numpy().copy(), t.copy()) for f, t in _drain(loader_a)]
        next_b = [(f.to_numpy().copy(), t.copy()) for f, t in _drain(loader_b)]
        assert len(next_a) == len(next_b) == sampler_a.batches_per_epoch
        for (fa, ta), (fb, tb) in zip(next_a, next_b):
            assert bit_view(fa) == bit_view(fb)
            assert ta.tolist() == tb.tolist()
        assert loader_a.state_dict() == loader_b.state_dict()

    # 10-11. Everything the proof allocated is closed, explicitly.
    loader_a.close()
    loader_b.close()
    dataset_a.close()
    dataset_b.close()
    assert settled(live_storages) == baseline
    return tail, next_position


@needs_backend
@pytest.mark.parametrize("dtype", [None, "float64", "float32"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_a_restored_loader_reproduces_the_exact_remaining_sequence(
        dtype, shuffle, drop_last, live_storages):
    """**The J4 exit gate.** A genuinely mid-epoch interruption, restored
    into an entirely separate object graph, reproduces every remaining
    batch exactly — indices, raw feature bits, and targets — and then the
    whole following epoch as well."""
    tail, next_position = _resume_proof(
        dtype=dtype, samples=11, batch_size=3, shuffle=shuffle, seed=7,
        drop_last=drop_last, consumed=1, live_storages=live_storages)
    # Non-vacuous by construction: the interruption is mid-epoch, so a
    # restoration that did not take would be visible immediately.
    assert len(tail) >= 1
    assert next_position[0] >= 1


@needs_backend
def test_the_resume_proof_is_not_vacuous(live_storages):
    """The negative control: **without** the restoration the two loaders'
    remaining sequences must be **unequal**, so the proof above cannot
    pass by accident."""
    baseline = settled(live_storages)
    dataset_a = make_dataset(11)
    sampler_a = NativeBatchSampler(dataset_a, batch_size=3, shuffle=True,
                                   seed=7)
    loader_a = NativeDataLoader(sampler_a)
    dataset_b = make_dataset(11)
    sampler_b = NativeBatchSampler(dataset_b, batch_size=1, shuffle=False,
                                   seed=8)
    loader_b = NativeDataLoader(sampler_b)

    iterator = iter(loader_a)
    features, _ = next(iterator)
    features.close()
    iterator.close()

    # No load_state_dict here — that is the point.
    assert sampler_a.next_batch_indices() != sampler_b.next_batch_indices()
    assert sampler_a.remaining != sampler_b.remaining
    assert loader_a.state_dict() != loader_b.state_dict()
    tail_a = [f.to_numpy().copy() for f, _ in _drain(loader_a)]
    tail_b = [f.to_numpy().copy() for f, _ in _drain(loader_b)]
    assert len(tail_a) != len(tail_b)

    loader_a.close()
    loader_b.close()
    dataset_a.close()
    dataset_b.close()
    assert settled(live_storages) == baseline


@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_resume_proof_compares_each_dtype_only_against_itself(
        dtype, live_storages):
    _resume_proof(dtype=dtype, samples=9, batch_size=4, shuffle=True, seed=0,
                  drop_last=False, consumed=1, live_storages=live_storages)


@needs_backend
def test_batch_indices_are_identical_across_equivalent_dtypes(live_storages):
    """§14.4: the permutation is a pure function of ``(seed, epoch,
    samples)`` and carries no dtype at all, so the *order* is
    dtype-independent even though the *values* are compared only against
    themselves. The two states are **not** interchangeable — the dataset
    identity's ``feature_dtype`` and fingerprint differ — which is
    asserted in both directions."""
    baseline = settled(live_storages)
    wide = make_dataset(11, dtype="float64")
    narrow = make_dataset(11, dtype="float32")
    wide_loader = NativeDataLoader(NativeBatchSampler(
        wide, batch_size=3, shuffle=True, seed=20240612))
    narrow_loader = NativeDataLoader(NativeBatchSampler(
        narrow, batch_size=3, shuffle=True, seed=20240612))

    wide_iterator, narrow_iterator = iter(wide_loader), iter(narrow_loader)
    for _ in range(2):
        features, _ = next(wide_iterator)
        features.close()
        features, _ = next(narrow_iterator)
        features.close()
    wide_iterator.close()
    narrow_iterator.close()

    wide_state = wide_loader.state_dict()
    narrow_state = narrow_loader.state_dict()
    assert (wide_state["sampler"]["epoch"], wide_state["sampler"]["cursor"]) \
        == (narrow_state["sampler"]["epoch"],
            narrow_state["sampler"]["cursor"])
    assert (wide_loader.sampler.next_batch_indices()
            == narrow_loader.sampler.next_batch_indices())
    for epoch in range(3):
        assert (wide_loader.sampler.plan(epoch)
                == narrow_loader.sampler.plan(epoch))
        assert (wide_loader.sampler.epoch_permutation(epoch)
                == narrow_loader.sampler.epoch_permutation(epoch))

    # ...and the two states are not interchangeable.
    assert (wide_state["sampler"]["dataset"]["feature_dtype"]
            != narrow_state["sampler"]["dataset"]["feature_dtype"])
    with pytest.raises(ValueError):
        wide_loader.load_state_dict(narrow_state)
    with pytest.raises(ValueError):
        narrow_loader.load_state_dict(wide_state)

    wide_loader.close()
    narrow_loader.close()
    wide.close()
    narrow.close()
    assert settled(live_storages) == baseline


# ===========================================================================
# 10. Required restoration positions (§7.4, §7.6, §9.3)
# ===========================================================================

@needs_backend
@pytest.mark.parametrize("samples,batch_size,drop_last,consumed,note", [
    pytest.param(9, 3, False, 0, "fresh", id="fresh"),
    pytest.param(9, 3, False, 1, "mid-epoch", id="genuine-mid-epoch"),
    pytest.param(9, 3, False, 2, "final batch ahead", id="final-batch"),
    pytest.param(9, 3, False, 3, "epoch boundary", id="epoch-boundary"),
    pytest.param(11, 3, False, 2, "short final batch", id="short-final-batch"),
    pytest.param(12, 3, False, 2, "exactly divisible", id="exact-divisible"),
    pytest.param(12, 3, True, 2, "exactly divisible, drop-last",
                 id="exact-divisible-drop-last"),
    pytest.param(11, 3, True, 1, "drop-last tail", id="drop-last-tail"),
    pytest.param(6, 6, False, 0, "one-batch epoch, before",
                 id="one-batch-before"),
    pytest.param(6, 6, False, 1, "one-batch epoch, after",
                 id="one-batch-after"),
    pytest.param(1, 1, False, 0, "one sample, before", id="one-sample-before"),
    pytest.param(1, 1, False, 1, "one sample, after", id="one-sample-after"),
    pytest.param(8, 20, False, 0, "batch larger than dataset",
                 id="batch-larger-than-dataset"),
])
def test_restoration_is_exact_at_every_required_position(
        samples, batch_size, drop_last, consumed, note, live_storages):
    for shuffle in (False, True):
        tail, _ = _resume_proof(
            dtype="float32", samples=samples, batch_size=batch_size,
            shuffle=shuffle, seed=7, drop_last=drop_last, consumed=consumed,
            live_storages=live_storages)
        assert isinstance(tail, list), note


@needs_backend
def test_a_genuine_mid_epoch_interruption_yields_exactly_the_tail():
    """§9.3: one iterator consumes the batches remaining in the **current**
    epoch — the whole epoch from a fresh position, and exactly the tail
    from a restored mid-epoch one."""
    loader, sampler, dataset = make_loader(9, batch_size=3, shuffle=True,
                                           seed=7)
    plan = sampler.plan()
    assert len(plan) == 3
    iterator = iter(loader)
    features, _ = next(iterator)
    features.close()
    iterator.close()
    assert (sampler.epoch, sampler.cursor) == (0, 1)     # genuinely mid-epoch

    restored_dataset = make_dataset(9)
    restored = NativeDataLoader(NativeBatchSampler(restored_dataset,
                                                   batch_size=2, seed=0))
    restored.load_state_dict(loader.state_dict())
    tail = [restored.sampler.next_batch_indices()]
    delivered = 0
    for features, _ in _drain(restored):
        delivered += 1
        if restored.sampler.remaining and restored.sampler.cursor:
            tail.append(restored.sampler.next_batch_indices())
    assert delivered == 2
    assert tail[0] == plan[1]
    assert restored.state_dict()["sampler"]["epoch"] == 1
    assert restored.state_dict()["sampler"]["cursor"] == 0
    dataset.close()
    restored_dataset.close()


@needs_backend
def test_a_final_batch_position_yields_one_batch_then_stops():
    loader, sampler, dataset = make_loader(9, batch_size=3, shuffle=True,
                                           seed=7)
    plan = sampler.plan()
    state = loader.state_dict()
    state["sampler"]["cursor"] = 2                 # the last batch of epoch 0
    loader.load_state_dict(state)
    assert sampler.remaining == 1
    assert sampler.next_batch_indices() == plan[2]
    delivered = [targets.copy() for _, targets in _drain(loader)]
    assert len(delivered) == 1
    assert len(delivered[0]) == len(plan[2])
    # ...and it canonicalizes to the next epoch rather than stopping at a
    # cursor equal to the batch count.
    assert loader.state_dict()["sampler"] == {**state["sampler"],
                                              "epoch": 1, "cursor": 0}
    loader.close()
    dataset.close()


@needs_backend
def test_an_epoch_boundary_state_begins_the_new_epoch_from_its_first_batch():
    """§7.4: the boundary is canonicalized immediately, so a save taken
    right after the final batch reads ``(epoch + 1, 0)`` and there is
    exactly one representation of the position."""
    loader, sampler, dataset = make_loader(6, batch_size=3, shuffle=True,
                                           seed=7)
    for features, _ in _drain(loader):
        pass
    boundary = loader.state_dict()
    assert boundary["sampler"]["epoch"] == 1
    assert boundary["sampler"]["cursor"] == 0

    restored_dataset = make_dataset(6)
    restored = NativeDataLoader(NativeBatchSampler(restored_dataset,
                                                   batch_size=1, seed=0))
    restored.load_state_dict(boundary)
    assert restored.sampler.remaining == restored.sampler.batches_per_epoch
    assert (restored.sampler.next_batch_indices()
            == sampler.plan(1)[0] == restored.sampler.plan(1)[0])
    assert restored.state_dict() == boundary
    dataset.close()
    restored_dataset.close()


@needs_backend
def test_a_later_epoch_state_restores_without_replaying_earlier_epochs():
    """A restored sampler must reproduce epoch 9's order without having
    consumed epochs 0 through 8 — which is exactly why a permutation is
    indexed by an epoch rather than by a call count."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    state = loader.state_dict()
    state["sampler"]["epoch"] = 9
    state["sampler"]["cursor"] = 1
    reference_order = sampler.epoch_permutation(9)

    restored_dataset = make_dataset(8)
    restored = NativeDataLoader(NativeBatchSampler(restored_dataset,
                                                   batch_size=2, seed=0))
    assert restored.load_state_dict(state) is None
    assert restored.sampler.epoch == 9
    assert restored.sampler.epoch_permutation() == reference_order
    assert restored.sampler.next_batch_indices() == restored.sampler.plan()[1]
    assert restored.state_dict() == state
    dataset.close()
    restored_dataset.close()


# ===========================================================================
# 11. No checkpoint coupling, in either direction (§13.1, §13.6)
# ===========================================================================

@pytest.mark.parametrize("pipeline", [LOADER_SOURCE, SAMPLER_SOURCE,
                                      DATASET_SOURCE])
def test_no_pipeline_module_imports_the_checkpoint(pipeline):
    names = code_identifiers(pipeline)
    for forbidden in ("native_checkpoint", "save_native_checkpoint",
                      "load_native_checkpoint", "_validated_metadata",
                      "_native_checkpoint_transaction"):
        assert forbidden not in names, (pipeline, forbidden)
    for module, _ in imported_pairs(pipeline):
        assert "checkpoint" not in module, (pipeline, module)


def test_the_checkpoint_module_names_no_pipeline_object():
    checkpoint = (REPO_ROOT / CHECKPOINT_SOURCE).read_text(encoding="utf-8")
    for pipeline in ("native_dataset", "native_sampler", "native_data_loader",
                     "_native_permutation", "NativeTensorDataset",
                     "NativeBatchSampler", "NativeDataLoader",
                     LOADER_FORMAT, SAMPLER_FORMAT):
        assert pipeline not in checkpoint, pipeline
    names = code_identifiers(CHECKPOINT_SOURCE)
    for pipeline in ("NativeDataLoader", "NativeBatchSampler",
                     "NativeTensorDataset", "loader", "sampler", "epoch",
                     "cursor"):
        assert pipeline not in names, pipeline


def test_no_checkpoint_call_discovers_a_loader(tmp_path, monkeypatch):
    """Saving and loading a real archive touches no loader: not through a
    registry, not through module traversal, not through the model."""
    if not cpp.is_available():
        pytest.skip("experimental C++ backend not built")
    from tensorforge.experimental import NativeLinear, save_native_checkpoint
    from tensorforge.experimental import load_native_checkpoint

    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    before = world(loader)
    touched = []
    monkeypatch.setattr(NativeDataLoader, "state_dict",
                        lambda self: touched.append("state") or {})
    monkeypatch.setattr(NativeDataLoader, "load_state_dict",
                        lambda self, state: touched.append("load"))
    model = NativeLinear(2, 2)
    path = tmp_path / "archive.tfnc"
    save_native_checkpoint(path, model, metadata={"next_step": 1})
    metadata = load_native_checkpoint(path, model)
    assert metadata == {"next_step": 1}
    assert touched == [], "a checkpoint call reached a loader"
    monkeypatch.undo()
    assert world(loader) == before
    for parameter in model.parameters():
        parameter.close()
    loader.close()
    dataset.close()


def test_no_global_registry_of_loaders_exists():
    """A loader is reachable only from the name a caller bound it to."""
    for module in (loader_module, sampler_module):
        for attribute in dir(module):
            value = getattr(module, attribute)
            assert not isinstance(value, (NativeDataLoader,
                                          NativeBatchSampler,
                                          NativeTensorDataset)), attribute
            if isinstance(value, (list, set, dict, tuple)) \
                    and not attribute.startswith("__"):
                assert not value or all(
                    not isinstance(item, (NativeDataLoader,
                                          NativeBatchSampler))
                    for item in (value.values() if isinstance(value, dict)
                                 else value)), attribute
    for name in ("registry", "REGISTRY", "_LOADERS", "_loaders", "_registry",
                 "_instances", "_active"):
        assert not hasattr(loader_module, name), name
        assert not hasattr(sampler_module, name), name


def test_the_checkpoint_format_and_versions_did_not_move():
    from tensorforge.experimental import native_optimizer_state

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert 4 not in native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert native_optimizer_state.FORMAT_VERSION == 1


# ===========================================================================
# 12. J4 non-goals — what this milestone must not have added
# ===========================================================================

def test_no_capability_registry_or_dtype_boundary_moved():
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    assert cpp.backend_info()["dtype"] == "float64"
    assert cpp.backend_info()["stable_framework_integration"] is False
    # No dtype or device argument reached either new method.
    for method in (NativeDataLoader.state_dict,
                   NativeDataLoader.load_state_dict):
        parameters = inspect.signature(method).parameters
        assert "dtype" not in parameters
        assert "device" not in parameters
        assert "map_location" not in parameters
        assert "strict" not in parameters


def test_no_c_abi_ctest_example_or_benchmark_surface_moved():
    """J4 is pure Python: no export, no CTest, no example, no benchmark."""
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                source.read_text(encoding="utf-8"), re.S))
    assert len(names) == 56, sorted(names)
    forbidden = re.compile(
        r"^tf_(dataset|sampler|loader|batch|shuffle|permut|gather|state)",
        re.I)
    for name in sorted(names):
        assert not forbidden.search(name), name
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    # Phase K, milestone K1 took the native CTest inventory from 24 to 25 (cpp/tests/test_dtype_int64_storage.cpp), which is the first movement since Phase I. The number is updated rather than the assertion relaxed: this test still pins an exact inventory, and still fails on an unrecorded addition.
    assert len(re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)) == 27
    examples = [p.name for p in (REPO_ROOT / "examples").glob("*.py")
                if p.name != "__init__.py"]
    benchmarks = [p.name for p in (REPO_ROOT / "benchmarks").glob("*.py")
                  if p.name != "__init__.py"]
    # 15 when J4 landed; 16 since **J6** added the one training example, and
    # 17 since **K6** added the one integer-indexing example. 8 benchmarks
    # when J4 landed; 9 since **J8** added exactly one. Each is named rather
    # than pattern-matched, so every *other* artifact still fails here.
    assert len(examples) == 17, sorted(examples)
    assert "native_minibatch_training.py" in examples
    assert "native_integer_indexing.py" in examples
    assert len(benchmarks) == 9, sorted(benchmarks)
    assert "benchmark_native_data_pipeline.py" in benchmarks
    for name in examples + benchmarks:
        if name in ("native_minibatch_training.py",                   # J6
                    "benchmark_native_data_pipeline.py"):             # J8
            continue
        for artifact in ("data_pipeline", "minibatch", "data_loader",
                         "loader_state"):
            assert artifact not in name, name


def test_no_later_milestone_test_landed():
    """J9's closure module belongs to a later milestone.

    ``test_native_data_checkpoint.py`` moved out of this list at **J5**,
    ``test_native_minibatch_training.py`` at **J6**,
    ``test_native_data_hardening.py`` at **J7**, and
    ``test_native_data_benchmark.py`` at **J8** — each in the milestone
    that shipped it, on the same "presence and absence are one split, and
    only the milestone that ships a name may move it" discipline J4
    applied to the loader's two state methods. The absence half below is
    otherwise untouched, and J7 and J8 each shipped **no production
    module**.
    """
    assert (REPO_ROOT / "tests" / "test_native_data_checkpoint.py").exists()
    assert (REPO_ROOT / "tests"
            / "test_native_minibatch_training.py").exists()
    assert (REPO_ROOT / "tests" / "test_native_data_hardening.py").exists()
    assert (REPO_ROOT / "tests" / "test_native_data_benchmark.py").exists()
    # J9's closure guardrails, likewise a module of their own. This line
    # asserted their **absence** through J8, which expired at closure.
    assert (REPO_ROOT / "tests"
            / "test_native_phase_j_closure.py").exists()
    for later in ("_native_data_hardening.py", "native_data_hardening.py",
                  "native_loader_state.py", "native_data_state.py"):
        assert not (PACKAGE / later).exists(), later


def test_the_loader_adds_no_worker_thread_lock_queue_or_async_surface():
    names = code_identifiers(LOADER_SOURCE)
    for forbidden in ("threading", "Thread", "Lock", "RLock", "queue",
                      "Queue", "multiprocessing", "asyncio", "concurrent",
                      "Future", "state_transaction", "os", "time", "random",
                      "secrets"):
        assert forbidden not in names, forbidden
    tree = ast.parse((REPO_ROOT / LOADER_SOURCE).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.AsyncFunctionDef, ast.Await,
                                     ast.AsyncFor, ast.AsyncWith)), node
    for absent in ("__aiter__", "__anext__", "num_workers", "prefetch",
                   "collate_fn", "worker_init_fn", "pin_memory", "timeout",
                   "persistent_workers", "transform", "callback", "on_batch"):
        assert not hasattr(NativeDataLoader, absent), absent
        assert not hasattr(loader_module._NativeBatchIterator, absent), absent


def test_the_state_methods_take_no_lock_and_join_no_lock_order():
    """§16.3: the Phase-J objects join neither the process-wide guard nor
    the universal generator order — there is nothing for them to serialize
    with, because no participant can reach one."""
    source = (REPO_ROOT / LOADER_SOURCE).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name in ("state_dict", "load_state_dict"):
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == name)
        for node in ast.walk(function):
            assert not isinstance(node, ast.With), name
            if isinstance(node, ast.Call) and isinstance(node.func,
                                                         ast.Attribute):
                assert node.func.attr not in ("acquire", "state_transaction",
                                              "snapshot_generator_states"), name


def test_nothing_from_this_milestone_entered_the_stable_public_api():
    for name in ("NativeDataLoader", "NativeBatchSampler",
                 "NativeTensorDataset", "native_data_loader",
                 "loader_state", "NativeLoaderState"):
        assert not hasattr(tensorforge, name), name
        assert name not in tensorforge.__all__, name
    assert hasattr(tensorforge, "batches")
    assert "batches" in tensorforge.__all__
    # The stable line still imports no experimental module.
    stable = REPO_ROOT / "src" / "tensorforge"
    offenders = [
        path.name for path in sorted(stable.glob("*.py"))
        if re.search(r"^\s*(from|import)\s+.*\bexperimental\b",
                     path.read_text(encoding="utf-8"), re.M)
    ]
    assert offenders == []


def test_the_dataset_and_sampler_gained_no_state_surface_at_j4():
    """The dataset is input, not training state, and the sampler's own
    surface is unchanged: J4 wraps it rather than extending it."""
    for absent in ("state_dict", "load_state_dict", "save", "load"):
        assert not hasattr(NativeTensorDataset, absent), absent
    sampler_public = {name for name in dir(NativeBatchSampler)
                      if not name.startswith("_")}
    assert sampler_public == {
        "dataset", "batch_size", "shuffle", "seed", "drop_last", "epoch",
        "cursor", "batches_per_epoch", "remaining", "epoch_permutation",
        "plan", "next_batch_indices", "state_dict", "load_state_dict",
    }
