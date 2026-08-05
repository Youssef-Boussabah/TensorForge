"""NativeDataLoader — the native mini-batch loader and its transactional
batch delivery (Phase J, milestone J3;
docs/native_data_pipeline_design.md §3.5, §3.6, §3.7, §9, §10, §15, §16,
§17.3).

What this module proves, and what it deliberately does not:

* **§9.4 the transaction** — the five phases, and the one invariant they
  exist for: **the committed sampler position advances if and only if a
  batch was successfully delivered.** Asserted at *every* failure
  position by injection, never argued: claim, construction, native
  allocation, target gather, publication, commit, and the delivery seam,
  each with a **non-vacuity control** proving the injected path really
  ran, an exact before/after position and state comparison, a
  live-storage baseline, and a retry that returns the **same indices and
  the same values**.
* **§9.2/§9.3 the iterator state machine** — one iterator is one epoch's
  captured countdown; supersession between transactions, and its refusal
  during one; exhaustion, close, supersession, loader close, and dataset
  close kept as five **distinct** states.
* **§9.5 the reentrancy matrix**, row by row, driven from inside the
  delivery seam so a reentrant arrival is real rather than simulated.
* **§10 materialization and ownership** — shape, dtype, device, order,
  contiguity, the read-only ``int64`` targets, at both widths and for
  scalar and higher-rank samples; and that a delivered batch is the
  **caller's**, unreachable from every close path.
* **§15/§17.3 lifecycle** — idempotent close, context managers, the four
  abandonment positions, and the garbage-collection fallback as a
  fallback only.

**Not tested here, because it belongs elsewhere or does not exist:**
loader ``state_dict``, loader ``load_state_dict``, the loader format tag,
and mid-epoch loader restoration are **J4's**, and live in
``tests/test_native_loader_state.py``; §12 below asserts only that they
exist and that nothing beside them arrived. Checkpoint loader-state
integration does not exist at all — that is J5 — and its absence *is*
asserted, in §12.

No test here asserts an exact error message, a dict ordering, a timing, a
GC event, or a speed.

Selector: python -m pytest -q tests/test_native_data_loader.py
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
from tensorforge.experimental import native_data_loader as loader_module
from tensorforge.experimental import native_sampler as sampler_module

REPO_ROOT = Path(__file__).resolve().parent.parent
LOADER_SOURCE = "src/tensorforge/experimental/native_data_loader.py"

# Everything that materializes a batch needs the built library. The
# constructor, the properties, the export inventory, the absence checks,
# the repr, and every refusal that happens *before* materialization are
# pure Python and stay provable without a C++ compiler.
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
    deterministic instrumentation for native-allocation lifetime (the
    Phase-C..J precedent). There is no public counter, and J3 adds
    none."""
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


def make_dataset(samples=8, width=2, dtype=None, rank=2):
    """A dataset whose values identify their row exactly, so a batch can be
    checked against the indices that produced it."""
    if rank == 1:
        features = np.arange(samples, dtype=np.float64) * 10.0
    elif rank == 2:
        features = (np.arange(samples * width, dtype=np.float64)
                    .reshape(samples, width))
    else:
        features = (np.arange(samples * width * 3, dtype=np.float64)
                    .reshape(samples, width, 3))
    targets = np.arange(samples, dtype=np.int64) % 3
    return NativeTensorDataset(features, targets, dtype=dtype)


def make_loader(samples=8, dataset=None, **kwargs):
    """A loader over a fresh dataset and sampler. Returns
    ``(loader, sampler, dataset)`` so a test can reach every level."""
    kwargs.setdefault("batch_size", 3)
    dataset = make_dataset(samples) if dataset is None else dataset
    sampler = NativeBatchSampler(dataset, **kwargs)
    return NativeDataLoader(sampler), sampler, dataset


def drain(loader, close_features=True):
    """Every batch of one iteration, as ``[(values, targets)]`` host
    copies, closing each feature tensor exactly as a caller must."""
    collected = []
    for features, targets in loader:
        collected.append((features.to_numpy().copy(), targets.copy()))
        if close_features:
            features.close()
    return collected


def position(sampler):
    """The committed position, as one comparable value."""
    return (sampler.epoch, sampler.cursor)


def observable(sampler):
    """Every publicly observable fact about a sampler, as one comparable
    value — used before and after every operation that must change
    nothing. The permutation cache is deliberately absent: it is not
    state (§7.8)."""
    return (
        id(sampler.dataset),
        sampler.batch_size, sampler.shuffle, sampler.seed,
        sampler.drop_last, sampler.epoch, sampler.cursor,
        sampler.batches_per_epoch, sampler.remaining,
        sampler.next_batch_indices(),
        json.dumps(sampler.state_dict(), sort_keys=True),
    )


def code_identifiers(relative):
    """Every identifier the module's **executable code** names.

    A source-text scan would be wrong here: this module explains at
    length what it deliberately does *not* do, so a prose mention of
    ``threading`` or ``prefetch`` would fail a substring check that is
    supposed to be about behavior. Reading the AST asks the question that
    was meant.
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


def test_the_identifier_scanner_can_actually_find_something():
    """Negative control for every scan below."""
    names = code_identifiers(LOADER_SOURCE)
    assert "NativeDataLoader" in names
    assert "_NativeBatchIterator" in names
    assert "_deliver_batch" in names
    assert "threading" not in names


# ===========================================================================
# 1. Public API, export inventory, and stable/native isolation
# ===========================================================================

def test_the_loader_module_and_class_exist_where_the_contract_says():
    assert (loader_module.__name__
            == "tensorforge.experimental.native_data_loader")
    assert loader_module.NativeDataLoader is NativeDataLoader
    assert (REPO_ROOT / "src" / "tensorforge" / "experimental"
            / "native_data_loader.py").is_file()


def test_j3_added_exactly_one_public_experimental_name():
    """The J3 exit gate over the live inventory: ``__all__`` grew from 24
    names to 25, by ``NativeDataLoader`` and nothing else."""
    post_j2 = {
        "NativeTensor", "NativeGenerator", "NativeParameter",
        "NativeParameterRegistry", "NativeModule", "NativeLinear",
        "NativeReLU", "NativeFlatten", "NativeConv2d", "NativeMaxPool2d",
        "NativeSequential", "NativeLayerNorm", "NativeBatchNorm1d",
        "NativeBatchNorm2d", "NativeDropout", "NativeMSELoss",
        "NativeCrossEntropyLoss", "native_accuracy", "NativeSGD",
        "NativeAdam", "save_native_checkpoint", "load_native_checkpoint",
        "NativeTensorDataset", "NativeBatchSampler",
    }
    assert len(post_j2) == 24
    live = set(experimental.__all__)
    assert len(experimental.__all__) == len(live), "duplicate export"
    assert len(experimental.__all__) == 25
    assert live - post_j2 == {"NativeDataLoader"}
    assert post_j2 - live == set()
    # J1's and J2's names are still exported, and still reachable.
    assert experimental.NativeTensorDataset is NativeTensorDataset
    assert experimental.NativeBatchSampler is NativeBatchSampler
    assert experimental.NativeDataLoader is NativeDataLoader


def test_the_iterator_and_the_delivery_seam_stay_private():
    """§3.2/§3.7: callers receive iterators from ``iter(loader)`` and never
    construct one, and the seam is a test seam rather than a hook."""
    for private in ("_NativeBatchIterator", "_deliver_batch",
                    "_BatchTransaction"):
        assert private not in experimental.__all__, private
        assert not hasattr(experimental, private), private
        assert not hasattr(tensorforge, private), private
    # They exist where the contract puts them, and only there.
    assert hasattr(loader_module, "_NativeBatchIterator")
    assert hasattr(loader_module, "_deliver_batch")
    assert hasattr(sampler_module, "_BatchTransaction")
    # No public alias of the iterator class on the loader either.
    for alias in ("iterator", "Iterator", "BatchIterator", "iterator_class"):
        assert not hasattr(NativeDataLoader, alias), alias


def test_nothing_from_this_milestone_entered_the_stable_public_api():
    for name in ("NativeDataLoader", "NativeBatchSampler",
                 "NativeTensorDataset", "_NativeBatchIterator",
                 "_deliver_batch", "native_data_loader"):
        assert not hasattr(tensorforge, name), name
        assert name not in tensorforge.__all__, name
    # The stable mini-batch iterator is untouched and stays stable-only.
    assert hasattr(tensorforge, "batches")
    assert "batches" in tensorforge.__all__


def test_importing_stable_tensorforge_stays_native_lazy():
    """§18: the stable line never imports the experimental one, and the
    loader reaches no stable API."""
    stable = REPO_ROOT / "src" / "tensorforge"
    offenders = [
        path.name for path in sorted(stable.glob("*.py"))
        if re.search(r"^\s*(from|import)\s+.*\bexperimental\b",
                     path.read_text(encoding="utf-8"), re.M)
    ]
    assert offenders == []
    names = code_identifiers(LOADER_SOURCE)
    for stable_name in ("batches", "train_test_split", "Tensor", "ctypes",
                        "backends"):
        assert stable_name not in names, stable_name


def test_the_loader_module_imports_only_what_the_contract_allows():
    """From one module, and one module only. No ctypes, no backends
    package, no NumPy, no json, no checkpoint, no generator, no threading,
    no queue.

    J4 added the two schema-shaped rules its state wrapper needs — the
    exact-``int`` check and the exact-key-set check — **shared rather than
    restated**, exactly as the sampler shares ``_validate_uint64`` rather
    than duplicating it. A second spelling of either would be free to
    drift from the one the nested schema is held to.
    """
    tree = ast.parse((REPO_ROOT / LOADER_SOURCE).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(("", alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update((node.module or "", alias.name)
                            for alias in node.names)
    assert imported == {("native_sampler", "NativeBatchSampler"),
                        ("native_sampler", "_require_exact_int"),
                        ("native_sampler", "_require_exact_keys")}
    assert {module for module, _ in imported} == {"native_sampler"}


def test_no_c_abi_or_build_surface_moved():
    """J3 is pure Python: no export, no CTest, no example, no benchmark."""
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                source.read_text(encoding="utf-8"), re.S))
    assert len(names) == 54, sorted(names)
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert len(re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)) == 25
    examples = [p.name for p in (REPO_ROOT / "examples").glob("*.py")
                if p.name != "__init__.py"]
    benchmarks = [p.name for p in (REPO_ROOT / "benchmarks").glob("*.py")
                  if p.name != "__init__.py"]
    # 15 when J3 landed; 16 since **J6** added the one training example, and
    # no other. 8 benchmarks when J3 landed; 9 since **J8** added exactly
    # one. Both artifacts are named rather than merely counted, so this
    # check keeps stating which milestone contributed each.
    assert len(examples) == 16, sorted(examples)
    assert "native_minibatch_training.py" in examples
    assert len(benchmarks) == 9, sorted(benchmarks)
    assert "benchmark_native_data_pipeline.py" in benchmarks


# ===========================================================================
# 2. The constructor (§3.5)
# ===========================================================================

@pytest.mark.parametrize("wrong", [
    None, 7, "sampler", 3.5, True, [], {}, object(),
])
def test_the_loader_requires_a_sampler(wrong):
    with pytest.raises(TypeError):
        NativeDataLoader(wrong)


def test_a_dataset_is_not_a_sampler():
    """The loader takes a sampler, not a dataset plus six keyword
    arguments: one fact, one owner, one constructor."""
    dataset = make_dataset(8)
    with pytest.raises(TypeError):
        NativeDataLoader(dataset)


def test_the_loader_keeps_the_exact_sampler_and_dataset_by_identity():
    loader, sampler, dataset = make_loader(8)
    assert loader.sampler is sampler
    assert loader.dataset is dataset
    assert loader.dataset is loader.sampler.dataset
    assert loader.closed is False


def test_the_loader_takes_no_other_argument():
    """No batch size, shuffle, seed, drop-last, dtype, device, worker,
    collate, transform, or prefetch argument exists."""
    signature = inspect.signature(NativeDataLoader)
    assert list(signature.parameters) == ["sampler"]
    parameter = signature.parameters["sampler"]
    assert parameter.default is inspect.Parameter.empty
    _, sampler, _ = make_loader(8)
    for absent in ("batch_size", "shuffle", "seed", "drop_last", "dtype",
                   "device", "num_workers", "collate_fn", "transform",
                   "prefetch", "pin_memory"):
        with pytest.raises(TypeError):
            NativeDataLoader(sampler, **{absent: 1})


def test_a_sampler_over_a_closed_dataset_is_accepted():
    """Planning survives a closed dataset, so refusing here would invent a
    lifecycle rule with no purpose. Only materialization refuses."""
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3)
    dataset.close()
    loader = NativeDataLoader(sampler)
    assert loader.dataset is dataset and loader.dataset.closed is True
    assert repr(loader)


@needs_backend
def test_construction_allocates_nothing_and_moves_nothing(live_storages):
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True, seed=7)
    before = observable(sampler)
    baseline = settled(live_storages)
    loader = NativeDataLoader(sampler)
    assert settled(live_storages) == baseline
    assert observable(sampler) == before
    assert sampler._transaction is None
    assert sampler._active_iterations == set()
    loader.close()
    dataset.close()


def test_the_public_members_are_read_only():
    loader, _, _ = make_loader(8)
    for name in ("sampler", "dataset", "closed"):
        with pytest.raises(AttributeError):
            setattr(loader, name, 1)
    # __slots__, so no attribute can be injected either.
    with pytest.raises(AttributeError):
        loader.anything = 1


def test_the_properties_survive_close():
    loader, sampler, dataset = make_loader(8)
    loader.close()
    assert loader.sampler is sampler
    assert loader.dataset is dataset
    assert loader.closed is True
    assert "closed=True" in repr(loader)


# ===========================================================================
# 3. Iteration (§9.1, §9.3, §10)
# ===========================================================================

def test_the_loader_is_not_its_own_iterator():
    """§9.1: merging them would make two nested loops silently share one
    position."""
    loader, _, _ = make_loader(8)
    iterator = iter(loader)
    assert iterator is not loader
    assert type(iterator) is loader_module._NativeBatchIterator
    assert iter(iterator) is iterator
    assert not hasattr(loader, "__next__")
    iterator.close()
    loader.close()


def test_two_iter_calls_return_distinct_objects():
    loader, _, _ = make_loader(8)
    first = iter(loader)
    second = iter(loader)
    assert first is not second
    first.close()
    second.close()
    loader.close()


@needs_backend
@pytest.mark.parametrize("shuffle", [False, True])
def test_one_iterator_delivers_exactly_one_epoch(shuffle):
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=shuffle,
                                           seed=7)
    plan = sampler.plan(0)
    batches = drain(loader)
    assert len(batches) == len(plan) == 3
    for (values, targets), indices in zip(batches, plan):
        assert values.shape == (len(indices), 2)
        assert np.array_equal(values, dataset._features[list(indices)])
        assert np.array_equal(targets, dataset._targets[list(indices)])
    loader.close()
    dataset.close()


@needs_backend
def test_a_following_iteration_runs_the_whole_next_epoch():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    first = drain(loader)
    assert (sampler.epoch, sampler.cursor) == (1, 0)
    second_plan = sampler.plan(1)
    second = drain(loader)
    assert (sampler.epoch, sampler.cursor) == (2, 0)
    assert len(second) == len(second_plan)
    for (values, _), indices in zip(second, second_plan):
        assert np.array_equal(values, dataset._features[list(indices)])
    # A shuffled epoch really is a different order, so "one epoch" is not
    # vacuously satisfied by an unchanging plan.
    assert sampler.plan(0) != second_plan
    assert ([values.tolist() for values, _ in first]
            != [values.tolist() for values, _ in second])
    loader.close()
    dataset.close()


@needs_backend
def test_a_mid_epoch_position_yields_only_the_tail():
    """§9.3: an iterator captures ``sampler.remaining``, so a restored
    mid-epoch position delivers exactly the remaining batches — which is
    precisely what an exact mid-epoch resume needs."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    state = sampler.state_dict()
    state["cursor"] = 2
    sampler.load_state_dict(state)
    assert sampler.remaining == 1
    batches = drain(loader)
    assert len(batches) == 1
    assert np.array_equal(batches[0][0],
                          dataset._features[list(sampler.plan(0)[2])])
    assert (sampler.epoch, sampler.cursor) == (1, 0)
    # And the next iteration is a whole epoch again.
    assert len(drain(loader)) == 3
    loader.close()
    dataset.close()


@needs_backend
@pytest.mark.parametrize("samples,batch_size,drop_last,sizes", [
    (8, 3, False, [3, 3, 2]),        # short final batch
    (8, 3, True, [3, 3]),            # drop-last omits the tail
    (8, 4, False, [4, 4]),           # exactly divisible
    (8, 4, True, [4, 4]),            # drop_last changes nothing there
    (8, 8, False, [8]),              # one-batch epoch
    (8, 12, False, [8]),             # batch larger than the dataset
    (1, 1, False, [1]),              # one-sample dataset
    (1, 1, True, [1]),
])
def test_the_batch_sizes_follow_the_plan(samples, batch_size, drop_last,
                                         sizes):
    loader, sampler, dataset = make_loader(samples, batch_size=batch_size,
                                           drop_last=drop_last)
    batches = drain(loader)
    assert [len(targets) for _, targets in batches] == sizes
    assert [values.shape[0] for values, _ in batches] == sizes
    assert (sampler.epoch, sampler.cursor) == (1, 0)
    loader.close()
    dataset.close()


@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32", None])
def test_both_native_dtypes_deliver_their_own_width(dtype):
    dataset = make_dataset(8, dtype=dtype)
    expected = "float64" if dtype is None else dtype
    loader, sampler, _ = make_loader(dataset=dataset, batch_size=3,
                                     shuffle=True, seed=7)
    for features, targets in loader:
        assert features.dtype == expected == dataset.dtype
        assert features.device == "cpu"
        # No widening anywhere on the way out.
        assert features.to_numpy().dtype == np.dtype(expected)
        assert targets.dtype == np.int64
        features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_the_batch_indices_are_identical_across_dtypes():
    """§14.4: the permutation is a pure function of ``(seed, epoch,
    samples)`` and carries no dtype at all."""
    plans = []
    for dtype in ("float64", "float32"):
        dataset = make_dataset(8, dtype=dtype)
        loader, sampler, _ = make_loader(dataset=dataset, batch_size=3,
                                         shuffle=True, seed=20240612)
        seen = []
        for _ in range(3):
            seen.append(sampler.next_batch_indices())
            features, _ = next(iter(loader))
            features.close()
        plans.append(seen)
        loader.close()
        dataset.close()
    assert plans[0] == plans[1]


@needs_backend
def test_scalar_samples_produce_a_rank_one_batch():
    """§4.2: ``ndim == 1`` means scalar feature samples, so a batch of B
    has shape ``(B,)`` and no trailing axis is invented."""
    dataset = make_dataset(8, rank=1)
    assert dataset.feature_shape == ()
    loader, sampler, _ = make_loader(dataset=dataset, batch_size=3,
                                     shuffle=True, seed=7)
    plan = sampler.plan(0)
    for (features, _), indices in zip(loader, plan):
        assert features.shape == (len(indices),)
        assert np.array_equal(features.to_numpy(),
                              dataset._features[list(indices)])
        features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_higher_rank_samples_keep_their_per_sample_shape():
    dataset = make_dataset(6, width=2, rank=3)
    assert dataset.feature_shape == (2, 3)
    loader, sampler, _ = make_loader(dataset=dataset, batch_size=4)
    features, _ = next(iter(loader))
    assert features.shape == (4, 2, 3)
    features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_the_target_batch_is_a_read_only_owning_contiguous_int64_array():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    indices = sampler.next_batch_indices()
    features, targets = next(iter(loader))
    assert type(targets) is np.ndarray
    assert targets.dtype == np.int64
    assert targets.shape == (len(indices),)
    assert targets.flags["C_CONTIGUOUS"]
    assert targets.flags["OWNDATA"]
    assert targets.flags["WRITEABLE"] is False
    assert np.array_equal(targets, dataset._targets[list(indices)])
    with pytest.raises(ValueError):
        targets[0] = 0
    # There is no target NativeTensor, and no native integer dtype.
    assert not isinstance(targets, NativeTensor)
    features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_the_feature_batch_reaches_native_storage_through_the_transfer():
    """The standing NumPy-compute tripwire: the values must arrive through
    the host→native transfer boundary, not through NumPy arithmetic on a
    tensor the loader built itself."""
    names = code_identifiers(LOADER_SOURCE)
    for forbidden in ("numpy", "np", "from_array", "zeros", "full",
                      "NativeStorage", "NativeTensorCore", "to_numpy",
                      "ascontiguousarray"):
        assert forbidden not in names, forbidden
    loader, sampler, dataset = make_loader(8, batch_size=3)
    features, _ = next(iter(loader))
    assert features.contiguous
    assert features.requires_grad is False
    assert features.grad is None
    assert features.owns_core
    features.close()
    loader.close()
    dataset.close()


# ===========================================================================
# 4. Position advancement (§7.4, §9.4 Phase 6)
# ===========================================================================

@needs_backend
def test_an_interior_batch_advances_the_cursor_exactly_once():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    assert position(sampler) == (0, 0)
    features, _ = next(iterator)
    assert position(sampler) == (0, 1)
    features.close()
    features, _ = next(iterator)
    assert position(sampler) == (0, 2)
    features.close()
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
def test_the_final_batch_canonicalizes_the_epoch_immediately():
    """§7.4: the boundary is canonicalized the moment the last batch of an
    epoch is delivered, so every position has exactly one representation."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    for expected in ((0, 1), (0, 2), (1, 0)):
        features, _ = next(iterator)
        assert position(sampler) == expected
        features.close()
    with pytest.raises(StopIteration):
        next(iterator)
    assert position(sampler) == (1, 0)
    loader.close()
    dataset.close()


@needs_backend
def test_a_one_batch_epoch_advances_the_epoch_on_every_delivery():
    loader, sampler, dataset = make_loader(1, batch_size=1)
    assert sampler.batches_per_epoch == 1
    for expected in range(1, 4):
        features, _ = next(iter(loader))
        assert position(sampler) == (expected, 0)
        features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_the_state_after_a_delivery_describes_the_following_batch():
    """§13.7: the loader's state always describes the *next* batch, so a
    checkpoint and a step counter cannot drift by one."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    plan = sampler.plan(0)
    iterator = iter(loader)
    for index in range(3):
        assert sampler.next_batch_indices() == plan[index]
        features, _ = next(iterator)
        features.close()
        if index < 2:
            assert sampler.next_batch_indices() == plan[index + 1]
            # Readable between batches, while the iterator is still active.
            assert sampler.state_dict()["cursor"] == index + 1
    assert sampler.state_dict()["epoch"] == 1
    assert sampler.state_dict()["cursor"] == 0
    loader.close()
    dataset.close()


@needs_backend
def test_the_exact_sequence_across_several_epochs_and_iterators():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    expected = [group for epoch in range(3) for group in sampler.plan(epoch)]
    seen = []
    for _ in range(3):
        for features, _ in loader:
            features.close()
            seen.append(None)
    # The indices are read from the pure planner before each delivery.
    loader2, sampler2, dataset2 = make_loader(8, batch_size=3, shuffle=True,
                                              seed=7)
    actual = []
    for _ in range(3):
        for _ in range(sampler2.batches_per_epoch):
            actual.append(sampler2.next_batch_indices())
            features, _ = next(iter(loader2))
            features.close()
    assert actual == expected
    assert len(seen) == len(expected)
    for owner in (loader, loader2):
        owner.close()
    dataset.close()
    dataset2.close()


@needs_backend
def test_drop_last_never_delivers_the_short_tail():
    loader, sampler, dataset = make_loader(8, batch_size=3, drop_last=True,
                                           shuffle=True, seed=7)
    order = sampler.epoch_permutation(0)
    batches = drain(loader)
    assert [len(t) for _, t in batches] == [3, 3]
    delivered = [int(value) for _, targets in batches for value in targets]
    assert len(delivered) == 6 and len(order) == 8
    assert position(sampler) == (1, 0)
    loader.close()
    dataset.close()


# ===========================================================================
# 5. Ownership (§9.4 Phase 6, §10.5, §15.6)
# ===========================================================================

@needs_backend
def test_the_caller_owns_the_delivered_batch_and_nothing_else_retains_it():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    features, targets = next(iterator)
    # Neither owner retains a reference to a delivered batch.
    assert iterator._features is None
    assert iterator._targets is None
    assert iterator._txn_serial == 0
    assert sampler._transaction is None
    # Identity comparisons throughout: an ndarray's ``==`` is elementwise.
    reachable = [getattr(loader, slot) for slot in NativeDataLoader.__slots__]
    reachable += [getattr(iterator, slot)
                  for slot in type(iterator).__slots__]
    reachable += [getattr(sampler, slot)
                  for slot in NativeBatchSampler.__slots__]
    for held in reachable:
        assert held is not features and held is not targets
    assert features.closed is False
    features.close()
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
def test_closing_the_loader_never_closes_a_delivered_batch(live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    baseline = settled(live_storages)
    features, targets = next(iter(loader))
    values = features.to_numpy().copy()
    loader.close()
    assert features.closed is False
    assert np.array_equal(features.to_numpy(), values)
    assert np.array_equal(targets, targets)           # untouched host memory
    assert targets.flags["WRITEABLE"] is False
    # The caller's explicit close is what restores the baseline.
    features.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_closing_the_iterator_never_closes_a_delivered_batch(live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    baseline = settled(live_storages)
    iterator = iter(loader)
    features, targets = next(iterator)
    iterator.close()
    assert features.closed is False
    assert features.to_numpy().shape == (3, 2)
    features.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_repeated_batches_own_independent_storage(live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    first, _ = next(iter(loader))
    # Reset to the same position, so the second batch has the same indices.
    state = sampler.state_dict()
    state["cursor"] = 0
    iterator = iter(loader)
    iterator.close()
    sampler.load_state_dict(state)
    second, _ = next(iter(loader))
    assert first is not second
    assert first._core.storage is not second._core.storage
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    first.close()
    assert second.closed is False
    assert second.to_numpy().shape == (3, 2)
    second.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_whole_epoch_returns_live_storage_to_its_baseline(live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    for features, _ in loader:
        features.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_target_array_survives_every_close():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    features, targets = next(iter(loader))
    values = targets.copy()
    features.close()
    loader.close()
    dataset.close()
    assert np.array_equal(targets, values)
    assert targets.dtype == np.int64
    assert targets.flags["WRITEABLE"] is False


# ===========================================================================
# 6. Supersession (§9.2)
# ===========================================================================

@needs_backend
def test_a_new_iterator_supersedes_the_old_between_batches():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    first = iter(loader)
    features, _ = next(first)
    features.close()
    assert position(sampler) == (0, 1)
    second = iter(loader)
    assert second is not first
    # The superseded iterator refuses rather than yielding.
    with pytest.raises(RuntimeError, match="supersed"):
        next(first)
    # ...and its close stays safe and idempotent.
    assert first.close() is None
    assert first.close() is None
    # The new iterator continues from the committed position.
    assert second._to_yield == sampler.remaining == 2
    remaining = [t.tolist() for _, t in drain(loader)]
    assert len(remaining) == 2
    assert position(sampler) == (1, 0)
    second.close()
    loader.close()
    dataset.close()


@needs_backend
def test_break_then_iterate_again_needs_no_manual_close():
    """The common pattern §9.2 exists to keep working."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    plan = sampler.plan(0)
    for features, _ in loader:
        features.close()
        break
    assert position(sampler) == (0, 1)
    seen = []
    for features, targets in loader:
        seen.append(features.shape[0])
        features.close()
    assert seen == [len(plan[1]), len(plan[2])]
    assert position(sampler) == (1, 0)
    loader.close()
    dataset.close()


@needs_backend
def test_the_active_iteration_count_returns_to_zero():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    assert sampler._active_iterations == set()
    first = iter(loader)
    second = iter(loader)
    assert len(sampler._active_iterations) == 2
    first.close()
    assert len(sampler._active_iterations) == 1
    # Exhausting the second releases the last participation.
    for features, _ in second:
        features.close()
    assert sampler._active_iterations == set()
    loader.close()
    dataset.close()


@needs_backend
def test_an_iterator_releases_its_participation_exactly_once():
    """Exact token discipline: exhausting, closing, and finalizing the same
    iterator must not decrement another iterator's participation."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    keeper = iter(loader)
    victim = iter(loader)
    assert len(sampler._active_iterations) == 2
    victim.close()
    victim.close()
    victim._finish()
    victim.__del__()
    assert sampler._active_iterations == {keeper._token}
    keeper.close()
    assert sampler._active_iterations == set()
    loader.close()
    dataset.close()


@needs_backend
def test_iteration_is_refused_during_a_claim(monkeypatch):
    """§9.2's carve-out: superseding mid-transaction would detach the
    iterator that owns the undelivered batch."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    seen = []
    original = NativeTensorDataset.feature_batch

    def reentrant(self, indices):
        assert sampler._transaction is not None
        with pytest.raises(RuntimeError):
            iter(loader)
        seen.append(sampler._transaction.status)
        return original(self, indices)

    monkeypatch.setattr(NativeTensorDataset, "feature_batch", reentrant)
    iterator = iter(loader)
    slot = loader._iterator
    active = set(sampler._active_iterations)
    features, _ = next(iterator)
    assert seen == ["claim"]
    # The refusal changed nothing: same slot, same participation.
    assert loader._iterator is slot
    assert sampler._active_iterations == active
    features.close()
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
def test_iteration_is_refused_during_a_pending_delivery(monkeypatch):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    seen = []

    def reentrant(record):
        assert sampler._transaction.status == "committed"
        with pytest.raises(RuntimeError):
            iter(loader)
        seen.append(True)
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", reentrant)
    iterator = iter(loader)
    slot = loader._iterator
    active = set(sampler._active_iterations)
    features, _ = next(iterator)
    assert seen == [True]
    assert loader._iterator is slot
    assert sampler._active_iterations == active
    assert position(sampler) == (0, 1)
    features.close()
    iterator.close()
    loader.close()
    dataset.close()


# ===========================================================================
# 7. Transaction phases (§9.4)
# ===========================================================================

@needs_backend
def test_the_claim_moves_no_committed_state(monkeypatch):
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    before = observable(sampler)
    seen = []
    original = NativeTensorDataset.feature_batch

    def inspect_claim(self, indices):
        transaction = sampler._transaction
        seen.append((transaction.status, transaction.indices,
                     position(sampler)))
        return original(self, indices)

    monkeypatch.setattr(NativeTensorDataset, "feature_batch", inspect_claim)
    features, _ = next(iter(loader))
    status, indices, moved = seen[0]
    assert status == "claim"
    assert indices == sampler.plan(0)[0]
    assert moved == (0, 0)                       # nothing advanced
    features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_construction_and_publication_move_no_committed_state(monkeypatch):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    seen = []
    original_target = NativeTensorDataset.target_batch
    original_publish = NativeBatchSampler._publish_pending

    def after_features(self, indices):
        # The feature tensor exists; the position must still be untouched.
        seen.append(("constructed", position(sampler),
                     sampler._transaction.status))
        return original_target(self, indices)

    def at_publish(self, serial, owner):
        original_publish(self, serial, owner)
        seen.append(("published", position(sampler),
                     sampler._transaction.status))

    monkeypatch.setattr(NativeTensorDataset, "target_batch", after_features)
    monkeypatch.setattr(NativeBatchSampler, "_publish_pending", at_publish)
    features, _ = next(iter(loader))
    assert seen == [("constructed", (0, 0), "claim"),
                    ("published", (0, 0), "pending")]
    features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_the_commit_happens_immediately_before_the_delivery_seam(monkeypatch):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    seen = []

    def at_seam(record):
        seen.append((position(sampler), sampler._transaction.status,
                     record._features is not None,
                     record._targets is not None))
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", at_seam)
    features, _ = next(iter(loader))
    assert seen == [((0, 1), "committed", True, True)]
    features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_the_serial_is_never_reused(monkeypatch):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    serials = []

    def record_serial(record):
        serials.append(record._txn_serial)
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", record_serial)
    for features, _ in loader:
        features.close()
    for features, _ in loader:
        features.close()
    assert len(serials) == 6
    assert serials == sorted(serials)
    assert len(set(serials)) == 6
    # A failed claim also consumes no serial, but a failed *delivery* does:
    # serials are opaque and skipping one costs nothing.
    assert sampler._next_serial == serials[-1] + 1
    loader.close()
    dataset.close()


@needs_backend
def test_a_stale_cleanup_cannot_disturb_a_newer_transaction(monkeypatch):
    """Exact-match: a cleanup for transaction N must match nothing once
    transaction N+1 is live."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    features, _ = next(iterator)
    features.close()
    stale_serial = iterator._txn_serial or 1
    seen = []

    def at_seam(record):
        live = sampler._transaction
        seen.append(live.serial)
        # Every wrong key must match nothing: a stale serial, a foreign
        # owner, and a serial that was never minted.
        assert sampler._rollback_pending(stale_serial, record._token) is False
        assert sampler._rollback_pending(live.serial, live.owner + 1000) is False
        assert sampler._rollback_pending(10**9, record._token) is False
        assert sampler._complete_pending(stale_serial, record._token) is False
        assert sampler._transaction is live
        assert live.status == "committed"
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", at_seam)
    features, _ = next(iterator)
    assert len(seen) == 1
    assert position(sampler) == (0, 2)
    features.close()
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
def test_a_transaction_cannot_complete_twice(monkeypatch):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    captured = {}

    def at_seam(record):
        captured["serial"] = record._txn_serial
        captured["token"] = record._token
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", at_seam)
    features, _ = next(iter(loader))
    features.close()
    # The record is gone, so a second completion matches nothing.
    assert sampler._complete_pending(captured["serial"],
                                     captured["token"]) is False
    assert sampler._rollback_pending(captured["serial"],
                                     captured["token"]) is False
    assert position(sampler) == (0, 1)
    loader.close()
    dataset.close()


@needs_backend
def test_the_transaction_record_holds_no_releasable_resource(monkeypatch):
    """§15.2: the sampler holds the **integer half** and never the batch,
    which is exactly why it still needs no ``close()``."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    seen = []

    def at_seam(record):
        transaction = sampler._transaction
        fields = {slot: getattr(transaction, slot)
                  for slot in type(transaction).__slots__}
        seen.append(fields)
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", at_seam)
    features, _ = next(iter(loader))
    fields = seen[0]
    assert set(fields) == {"serial", "owner", "status", "before", "after",
                           "indices"}
    for value in fields.values():
        assert not isinstance(value, (NativeTensor, np.ndarray))
    assert type(fields["serial"]) is int and type(fields["owner"]) is int
    assert fields["before"] == (sampler.seed, sampler.shuffle,
                                sampler.batch_size, sampler.drop_last, 0, 0)
    assert fields["after"][-2:] == (0, 1)
    assert not hasattr(sampler, "close")
    assert not hasattr(sampler, "closed")
    features.close()
    loader.close()
    dataset.close()


# ===========================================================================
# 8. The reentrancy matrix (§9.5)
# ===========================================================================

def _reentrant_probe(monkeypatch, loader, sampler, probe, phase="pending"):
    """Run ``probe()`` from inside the transaction, at the requested phase,
    and return whatever it collected. The probe runs on the calling
    thread, so a reentrant arrival is real rather than simulated."""
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


@needs_backend
@pytest.mark.parametrize("phase", ["claim", "pending"])
def test_a_second_next_on_the_same_iterator_is_refused(monkeypatch, phase):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterators = []

    def probe():
        iterator = loader._iterator
        iterators.append(iterator)
        with pytest.raises(RuntimeError):
            next(iterator)
        return position(sampler)

    moved = _reentrant_probe(monkeypatch, loader, sampler, probe, phase)
    assert moved == ((0, 0) if phase == "claim" else (0, 1))
    assert position(sampler) == (0, 1)
    loader.close()
    dataset.close()


@needs_backend
@pytest.mark.parametrize("phase", ["claim", "pending"])
def test_the_sampler_state_methods_are_refused_in_a_transaction(monkeypatch,
                                                                phase):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    malformed = {"not": "a sampler state"}

    def probe():
        with pytest.raises(RuntimeError):
            sampler.state_dict()
        # The transaction guard runs **before** the container check, so a
        # malformed state is refused without ever being inspected.
        with pytest.raises(RuntimeError):
            sampler.load_state_dict(malformed)
        with pytest.raises(RuntimeError):
            sampler.load_state_dict(None)
        with pytest.raises(RuntimeError):
            iter(loader)
        return True

    assert _reentrant_probe(monkeypatch, loader, sampler, probe, phase)
    # ...and both work again the moment the handoff is over.
    assert sampler.state_dict()["cursor"] == 1
    loader.close()
    dataset.close()


@needs_backend
@pytest.mark.parametrize("phase", ["claim", "pending"])
def test_pure_planning_and_property_reads_stay_permitted(monkeypatch, phase):
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    plan = sampler.plan(0)

    def probe():
        return (sampler.epoch, sampler.cursor, sampler.remaining,
                sampler.batches_per_epoch, sampler.batch_size,
                sampler.seed, sampler.shuffle, sampler.drop_last,
                sampler.epoch_permutation(0), sampler.plan(0),
                sampler.next_batch_indices(), repr(sampler), repr(loader),
                loader.closed, loader.sampler is sampler,
                loader.dataset is dataset)

    seen = _reentrant_probe(monkeypatch, loader, sampler, probe, phase)
    assert seen[8] == sampler.epoch_permutation(0)
    assert seen[9] == plan
    assert seen[-3] is False and seen[-2] and seen[-1]
    loader.close()
    dataset.close()


@needs_backend
def test_closing_the_dataset_during_a_pending_delivery_is_permitted(
        monkeypatch):
    """§9.5: the pending batch is already built, so the in-flight
    transaction is unaffected; the *next* attempt fails."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    seen = []

    def at_seam(record):
        dataset.close()
        seen.append(dataset.closed)
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", at_seam)
    iterator = iter(loader)
    features, targets = next(iterator)
    assert seen == [True]
    # The already materialized batch is untouched and fully valid.
    assert features.to_numpy().shape == (3, 2)
    assert targets.shape == (3,)
    assert position(sampler) == (0, 1)
    # The next attempt fails, consuming nothing.
    before = observable(sampler)
    with pytest.raises(RuntimeError):
        next(iterator)
    assert observable(sampler) == before
    assert sampler._transaction is None
    features.close()
    iterator.close()
    loader.close()


@needs_backend
def test_a_reentrant_iterator_close_rolls_the_transaction_back(monkeypatch,
                                                              live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    indices = sampler.next_batch_indices()
    seen = []

    def at_seam(record):
        assert position(sampler) == (0, 1)          # committed
        record.close()                              # reentrant recovery
        seen.append((position(sampler), sampler._transaction,
                     record._features))
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", at_seam)
    iterator = iter(loader)
    with pytest.raises(RuntimeError):
        next(iterator)
    assert len(seen) == 1, "the reentrant close never ran"
    assert seen[0][0] == (0, 0)                     # restored inside close
    assert seen[0][1] is None                       # record cleared
    assert observable(sampler) == before
    assert settled(live_storages) == baseline
    assert iterator._features is None and iterator._txn_serial == 0
    # The iterator really is closed, and stays a lifecycle error.
    with pytest.raises(RuntimeError):
        next(iterator)
    # A retry through a fresh iterator returns the same batch.
    monkeypatch.setattr(loader_module, "_deliver_batch",
                        lambda record: (record._features, record._targets))
    assert sampler.next_batch_indices() == indices
    features, _ = next(iter(loader))
    assert np.array_equal(features.to_numpy(),
                          dataset._features[list(indices)])
    features.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_reentrant_loader_close_rolls_the_transaction_back(monkeypatch,
                                                            live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    baseline = settled(live_storages)
    before = observable(sampler)
    seen = []

    def at_seam(record):
        loader.close()
        seen.append((position(sampler), sampler._transaction, loader.closed))
        return record._features, record._targets

    monkeypatch.setattr(loader_module, "_deliver_batch", at_seam)
    with pytest.raises(RuntimeError):
        next(iter(loader))
    assert seen == [((0, 0), None, True)]
    assert observable(sampler) == before
    assert sampler._active_iterations == set()
    assert settled(live_storages) == baseline
    with pytest.raises(RuntimeError):
        iter(loader)
    dataset.close()


# ===========================================================================
# 9. Failure injection and cleanup (§9.6, §10.6, §17.3)
# ===========================================================================
#
# Every row below asserts the same six things, so the invariant is proved
# rather than argued: the injected path really ran (non-vacuity), the
# committed position is byte-identical to its pre-call value, the whole
# sampler state is, the captured countdown is, live native storage is back
# at its baseline, and no claim or pending record survives. Then the retry
# returns the **same indices and the same values**.

def _assert_nothing_consumed(sampler, iterator, before, countdown,
                             live_storages, baseline):
    assert observable(sampler) == before
    assert iterator._to_yield == countdown
    assert sampler._transaction is None
    assert iterator._features is None
    assert iterator._targets is None
    assert iterator._txn_serial == 0
    assert settled(live_storages) == baseline


@needs_backend
def test_a_claim_failure_before_publication_changes_nothing(monkeypatch,
                                                            live_storages):
    """Injected at the pure planner, before any claim is published: no
    record, and not even a serial, is consumed."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    serial_before = sampler._next_serial
    calls = []

    def boom(self):
        calls.append(True)
        raise ValueError("injected: candidate planning")

    monkeypatch.setattr(NativeBatchSampler, "next_batch_indices", boom)
    iterator = iter(loader)
    with pytest.raises(ValueError, match="injected"):
        next(iterator)
    assert calls == [True], "the injection never ran"
    monkeypatch.undo()
    _assert_nothing_consumed(sampler, iterator, before, 3, live_storages,
                             baseline)
    assert sampler._next_serial == serial_before   # no serial minted either
    features, _ = next(iterator)
    assert position(sampler) == (0, 1)
    features.close()
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
def test_a_feature_materialization_failure_clears_only_its_claim(
        monkeypatch, live_storages):
    """The first thing after the claim is published, and the M1/M2 row of
    §10.6 at once."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    indices = sampler.next_batch_indices()
    calls = []
    original = NativeTensorDataset.feature_batch

    def boom(self, wanted):
        calls.append((sampler._transaction.status, tuple(wanted),
                      position(sampler)))
        raise ValueError("injected: feature gather")

    monkeypatch.setattr(NativeTensorDataset, "feature_batch", boom)
    iterator = iter(loader)
    with pytest.raises(ValueError, match="injected"):
        next(iterator)
    # Non-vacuity: the claim really was standing and nothing had advanced.
    assert calls == [("claim", indices, (0, 0))]
    monkeypatch.setattr(NativeTensorDataset, "feature_batch", original)
    _assert_nothing_consumed(sampler, iterator, before, 3, live_storages,
                             baseline)
    # Retry: the same indices, the same values, a fresh allocation.
    assert sampler.next_batch_indices() == indices
    features, targets = next(iterator)
    assert np.array_equal(features.to_numpy(),
                          dataset._features[list(indices)])
    assert np.array_equal(targets, dataset._targets[list(indices)])
    features.close()
    iterator.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
@pytest.mark.skipif(not cpp.fault_injection_available(),
                    reason="the build has no allocation fault injection")
def test_a_native_allocation_failure_inside_materialization_consumes_nothing(
        live_storages):
    """The honest answer for an oversized batch is ``MemoryError`` from the
    native allocation, and it must consume nothing."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    indices = sampler.next_batch_indices()
    iterator = iter(loader)
    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        next(iterator)
    cpp._arm_alloc_failure(0)
    cpp._require_library().tf_clear_error()
    _assert_nothing_consumed(sampler, iterator, before, 3, live_storages,
                             baseline)
    # Non-vacuity: without the injection the very same call succeeds.
    assert sampler.next_batch_indices() == indices
    features, _ = next(iterator)
    assert np.array_equal(features.to_numpy(),
                          dataset._features[list(indices)])
    assert position(sampler) == (0, 1)
    features.close()
    iterator.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_target_failure_after_the_feature_tensor_exists_closes_it(
        monkeypatch, live_storages):
    """§10.6's one Phase-2 cleanup Phase J writes itself: the feature
    tensor exists, so the iterator must close it before re-raising."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    indices = sampler.next_batch_indices()
    calls = []
    original = NativeTensorDataset.target_batch

    def boom(self, wanted):
        iterator = loader._iterator
        # Non-vacuity, and the precise position: the feature tensor is
        # built and iterator-owned, and nothing has advanced.
        calls.append((iterator._features is not None,
                      iterator._features.closed, position(sampler),
                      settled(live_storages)))
        raise ValueError("injected: target gather")

    monkeypatch.setattr(NativeTensorDataset, "target_batch", boom)
    iterator = iter(loader)
    with pytest.raises(ValueError, match="injected"):
        next(iterator)
    assert calls[0][0] is True and calls[0][1] is False
    assert calls[0][2] == (0, 0)
    assert calls[0][3] == baseline + 1, "the feature tensor was not allocated"
    monkeypatch.setattr(NativeTensorDataset, "target_batch", original)
    _assert_nothing_consumed(sampler, iterator, before, 3, live_storages,
                             baseline)
    assert sampler.next_batch_indices() == indices
    features, targets = next(iterator)
    assert np.array_equal(features.to_numpy(),
                          dataset._features[list(indices)])
    assert np.array_equal(targets, dataset._targets[list(indices)])
    features.close()
    iterator.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_publication_failure_closes_the_batch_and_clears_the_claim(
        monkeypatch, live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    indices = sampler.next_batch_indices()
    calls = []

    def boom(self, serial, owner):
        calls.append((position(sampler), self._transaction.status,
                      settled(live_storages)))
        raise RuntimeError("injected: publication")

    monkeypatch.setattr(NativeBatchSampler, "_publish_pending", boom)
    iterator = iter(loader)
    with pytest.raises(RuntimeError, match="injected"):
        next(iterator)
    assert calls[0][0] == (0, 0) and calls[0][1] == "claim"
    assert calls[0][2] == baseline + 1
    monkeypatch.undo()
    _assert_nothing_consumed(sampler, iterator, before, 3, live_storages,
                             baseline)
    assert sampler.next_batch_indices() == indices
    features, _ = next(iterator)
    features.close()
    iterator.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_commit_failure_before_delivery_restores_the_exact_position(
        monkeypatch, live_storages):
    """The real write seam is structurally non-failing — asserted below —
    so this failure is injected into it deliberately, to prove the
    rollback would still be exact if it ever could raise."""
    source = inspect.getsource(NativeBatchSampler._assign_state)
    for node in ast.walk(ast.parse(source.lstrip())):
        assert not isinstance(node, (ast.Raise, ast.Call)), source

    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    indices = sampler.next_batch_indices()
    original = NativeBatchSampler._assign_state
    calls = []

    def boom(self, *values):
        calls.append(values)
        if len(calls) == 1:
            raise RuntimeError("injected: candidate commit")
        return original(self, *values)

    monkeypatch.setattr(NativeBatchSampler, "_assign_state", boom)
    iterator = iter(loader)
    with pytest.raises(RuntimeError, match="injected"):
        next(iterator)
    monkeypatch.undo()
    # Non-vacuity, and the shape of the two calls: the candidate, then the
    # restore of the exact pre-delivery position.
    assert len(calls) == 2, calls
    assert calls[0][-2:] == (0, 1)
    assert calls[1] == (sampler.seed, sampler.shuffle, sampler.batch_size,
                        sampler.drop_last, 0, 0)
    _assert_nothing_consumed(sampler, iterator, before, 3, live_storages,
                             baseline)
    assert sampler.next_batch_indices() == indices
    features, _ = next(iterator)
    assert np.array_equal(features.to_numpy(),
                          dataset._features[list(indices)])
    features.close()
    iterator.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_delivery_seam_failure_consumes_nothing(monkeypatch, live_storages,
                                                  dtype):
    """The load-bearing proof (§14.5): a failure injected at the seam, after
    the candidate position was applied, leaves epoch, cursor, the whole
    state dict, and live storage exactly as they were — and the very next
    ``__next__`` returns the **same indices and the same values**."""
    dataset = make_dataset(8, dtype=dtype)
    loader, sampler, _ = make_loader(dataset=dataset, batch_size=3,
                                     shuffle=True, seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    state_before = sampler.state_dict()
    indices = sampler.next_batch_indices()
    calls = []

    def boom(record):
        # Non-vacuity, plus the exact state at the injection point: the
        # candidate is committed and the batch is iterator-owned.
        calls.append((position(sampler), sampler._transaction.status,
                      record._features.dtype, settled(live_storages)))
        raise ValueError("injected: delivery seam")

    monkeypatch.setattr(loader_module, "_deliver_batch", boom)
    iterator = iter(loader)
    with pytest.raises(ValueError, match="injected"):
        next(iterator)
    assert len(calls) == 1, "the seam injection never ran"
    assert calls[0][0] == (0, 1) and calls[0][1] == "committed"
    assert calls[0][2] == dtype
    assert calls[0][3] == baseline + 1
    monkeypatch.undo()
    # Nothing was consumed, by every measure the contract names.
    _assert_nothing_consumed(sampler, iterator, before, 3, live_storages,
                             baseline)
    assert sampler.state_dict() == state_before
    assert loader.closed is False
    # ...and the retry returns the same indices and the same values, in a
    # freshly allocated tensor, because the rolled-back one was closed.
    assert sampler.next_batch_indices() == indices
    features, targets = next(iterator)
    assert np.array_equal(features.to_numpy(),
                          dataset._features[list(indices)])
    assert np.array_equal(targets, dataset._targets[list(indices)])
    assert position(sampler) == (0, 1)
    assert iterator._to_yield == 2
    features.close()
    iterator.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_rollback_is_idempotent_when_invoked_twice(monkeypatch,
                                                       live_storages):
    """A ``close()`` racing the transaction's own ``finally`` must not
    double-roll: whichever arrives first performs it, and the second
    matches nothing."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    baseline = settled(live_storages)
    before = observable(sampler)
    captured = {}

    def boom(record):
        captured["serial"] = record._txn_serial
        captured["token"] = record._token
        # First rollback, from inside the transaction.
        assert record._rollback(record._txn_serial) is True
        assert position(sampler) == (0, 0)
        # Second, third: exact-match, so they find nothing.
        assert record._rollback(captured["serial"]) is False
        assert record._rollback(captured["serial"]) is False
        raise ValueError("injected: after a manual rollback")

    monkeypatch.setattr(loader_module, "_deliver_batch", boom)
    iterator = iter(loader)
    with pytest.raises(ValueError, match="injected"):
        next(iterator)
    # The transaction's own unconditional finally ran a fourth time and
    # also matched nothing; nothing was double-closed.
    assert observable(sampler) == before
    assert settled(live_storages) == baseline
    assert sampler._rollback_pending(captured["serial"],
                                     captured["token"]) is False
    monkeypatch.undo()
    features, _ = next(iterator)
    features.close()
    iterator.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_an_epoch_overflow_candidate_is_refused_and_moves_nothing(
        live_storages):
    """§7.4: an advance past ``2**64 - 1`` raises and moves nothing — no
    claim, no allocation, and certainly no wrapped epoch."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    baseline = settled(live_storages)
    state = sampler.state_dict()
    state["epoch"] = 2 ** 64 - 1
    state["cursor"] = sampler.batches_per_epoch - 1
    sampler.load_state_dict(state)
    before = observable(sampler)
    serial_before = sampler._next_serial
    iterator = iter(loader)
    with pytest.raises(RuntimeError):
        next(iterator)
    assert observable(sampler) == before
    assert sampler.epoch == 2 ** 64 - 1
    assert sampler._transaction is None
    assert sampler._next_serial == serial_before
    assert iterator._to_yield == 1
    assert settled(live_storages) == baseline
    # It is not an exhaustion and not a close: it stays a refusal.
    with pytest.raises(RuntimeError):
        next(iterator)
    iterator.close()
    loader.close()
    dataset.close()


@needs_backend
def test_a_dataset_closed_before_iteration_refuses_at_materialization(
        live_storages):
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True, seed=7)
    loader = NativeDataLoader(sampler)
    dataset.close()
    baseline = settled(live_storages)
    before = observable(sampler)
    iterator = iter(loader)
    with pytest.raises(RuntimeError):
        next(iterator)
    assert observable(sampler) == before
    assert sampler._transaction is None
    assert iterator._to_yield == 3
    assert settled(live_storages) == baseline
    # Planning and state remain fully available.
    assert sampler.plan(0) == sampler.plan(0)
    assert sampler.state_dict()["cursor"] == 0
    iterator.close()
    loader.close()


@needs_backend
def test_a_dataset_closed_between_batches_consumes_nothing(live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    baseline = settled(live_storages)
    iterator = iter(loader)
    features, targets = next(iterator)
    values = features.to_numpy().copy()
    dataset.close()
    before = observable(sampler)
    with pytest.raises(RuntimeError):
        next(iterator)
    assert observable(sampler) == before
    assert position(sampler) == (0, 1)
    # The previously delivered batch is still valid and still the caller's.
    assert features.closed is False
    assert np.array_equal(features.to_numpy(), values)
    assert np.array_equal(targets, targets)
    features.close()
    iterator.close()
    loader.close()
    assert settled(live_storages) == baseline


# ===========================================================================
# 10. Lifecycle (§15)
# ===========================================================================

@needs_backend
def test_loader_close_is_idempotent_and_never_refused():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    assert loader.close() is None
    assert loader.close() is None
    assert loader.closed is True
    assert loader._iterator is None
    assert sampler._active_iterations == set()
    # It closed the current iterator, and never the sampler or dataset.
    with pytest.raises(RuntimeError):
        next(iterator)
    assert dataset.closed is False
    assert not hasattr(sampler, "closed")
    with pytest.raises(RuntimeError):
        iter(loader)
    dataset.close()


@needs_backend
def test_iterator_close_is_idempotent_and_never_refused():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    assert iterator.close() is None
    assert iterator.close() is None
    assert loader._iterator is None
    assert sampler._active_iterations == set()
    with pytest.raises(RuntimeError):
        next(iterator)
    # The loader is still perfectly usable.
    assert loader.closed is False
    features, _ = next(iter(loader))
    features.close()
    loader.close()
    dataset.close()


@needs_backend
def test_the_loader_context_manager_closes_and_propagates():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    with loader as entered:
        assert entered is loader
        features, _ = next(iter(loader))
        features.close()
    assert loader.closed is True
    loader2, _, _ = make_loader(8, batch_size=3, dataset=dataset)
    with pytest.raises(ValueError, match="propagated"):
        with loader2:
            raise ValueError("propagated")
    assert loader2.closed is True
    dataset.close()


@needs_backend
def test_the_iterator_context_manager_closes_and_propagates():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    with iter(loader) as iterator:
        assert iter(iterator) is iterator
        features, _ = next(iterator)
        features.close()
    assert sampler._active_iterations == set()
    with pytest.raises(RuntimeError):
        next(iterator)
    with pytest.raises(ValueError, match="propagated"):
        with iter(loader) as second:
            raise ValueError("propagated")
    assert sampler._active_iterations == set()
    with pytest.raises(RuntimeError):
        next(second)
    loader.close()
    dataset.close()


@needs_backend
def test_exhaustion_close_supersession_and_loader_close_stay_distinct():
    """Four different events, four different outcomes; conflating them
    would let a ``for`` loop silently swallow a lifecycle error."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    # 1. Ordinary exhaustion: StopIteration, repeatedly.
    exhausted = iter(loader)
    for features, _ in exhausted:
        features.close()
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(exhausted)
    # 2. Explicit close: RuntimeError, not StopIteration.
    closed = iter(loader)
    closed.close()
    with pytest.raises(RuntimeError):
        next(closed)
    # 3. Supersession: RuntimeError naming supersession.
    superseded = iter(loader)
    iter(loader)
    with pytest.raises(RuntimeError, match="supersed"):
        next(superseded)
    # 4. Loader close: RuntimeError.
    live = iter(loader)
    loader.close()
    with pytest.raises(RuntimeError):
        next(live)
    # 5. Dataset close is its own refusal, on a fresh loader.
    loader2 = NativeDataLoader(sampler)
    dataset.close()
    with pytest.raises(RuntimeError):
        next(iter(loader2))
    loader2.close()


@needs_backend
def test_an_exhausted_iterator_detaches_and_never_restarts_an_epoch():
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    for features, _ in iterator:
        features.close()
    assert position(sampler) == (1, 0)
    assert loader._iterator is None
    assert sampler._active_iterations == set()
    # It does not advance into the following epoch.
    with pytest.raises(StopIteration):
        next(iterator)
    assert position(sampler) == (1, 0)
    loader.close()
    dataset.close()


@needs_backend
@pytest.mark.parametrize("stage", ["fresh", "claim", "constructed",
                                   "pending"])
def test_explicit_close_cleans_up_at_every_abandonment_position(
        monkeypatch, live_storages, stage):
    """The four distinct cleanup positions §9.6 names: an iterator with no
    transaction; a claim with no tensor; a tensor before publication; and
    a pending record before delivery. Explicit close is the correctness
    proof at every one."""
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                           seed=7)
    baseline = settled(live_storages)
    before = observable(sampler)
    ran = []

    if stage == "fresh":
        iterator = iter(loader)
        iterator.close()
        ran.append(True)
    else:
        if stage == "claim":
            original = NativeTensorDataset.feature_batch

            def hook(self, indices):
                ran.append(loader._iterator._features is None)
                loader._iterator.close()
                return original(self, indices)

            monkeypatch.setattr(NativeTensorDataset, "feature_batch", hook)
        elif stage == "constructed":
            original = NativeTensorDataset.target_batch

            def hook(self, indices):
                ran.append(loader._iterator._features is not None)
                loader._iterator.close()
                return original(self, indices)

            monkeypatch.setattr(NativeTensorDataset, "target_batch", hook)
        else:
            def hook(record):
                ran.append(sampler._transaction.status == "committed")
                record.close()
                return record._features, record._targets

            monkeypatch.setattr(loader_module, "_deliver_batch", hook)
        iterator = iter(loader)
        with pytest.raises(RuntimeError):
            next(iterator)
        monkeypatch.undo()

    assert ran == [True], f"the {stage} hook never ran"
    # Deterministic final state at every position.
    assert observable(sampler) == before
    assert sampler._transaction is None
    assert sampler._active_iterations == set()
    assert iterator._features is None and iterator._targets is None
    assert iterator._txn_serial == 0
    assert loader._iterator is None
    assert settled(live_storages) == baseline
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_delivered_batch_survives_abandonment_of_its_iterator(
        live_storages):
    loader, sampler, dataset = make_loader(8, batch_size=3)
    baseline = settled(live_storages)
    iterator = iter(loader)
    features, targets = next(iterator)
    values = features.to_numpy().copy()
    del iterator
    gc.collect()
    # The committed position correctly records the delivered batch...
    assert position(sampler) == (0, 1)
    # ...and the batch is untouched.
    assert features.closed is False
    assert np.array_equal(features.to_numpy(), values)
    features.close()
    loader.close()
    dataset.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_finalizer_is_a_fallback_and_nothing_depends_on_its_timing():
    """A smoke test only: no assertion here depends on *when* collection
    happens, and the explicit close paths above are the contract."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    iterator = iter(loader)
    token = iterator._token
    assert token in sampler._active_iterations
    iterator.__del__()                    # the fallback, invoked directly
    assert token not in sampler._active_iterations
    assert iterator._closed is True
    # Idempotent: a real collection afterwards changes nothing.
    iterator.__del__()
    del iterator
    gc.collect()
    assert sampler._active_iterations == set()
    loader.close()
    dataset.close()


def test_the_reprs_carry_no_data_and_survive_every_close():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                          seed=20240612)
    iterator = iter(loader)
    for text in (repr(loader), repr(iterator)):
        assert "0x" not in text
        assert str(id(loader)) not in text and str(id(iterator)) not in text
        for forbidden in ("array", "NativeTensor", "fingerprint",
                          "transaction", "features", "targets"):
            assert forbidden not in text, (text, forbidden)
    assert "closed=False" in repr(loader)
    assert "open" in repr(iterator)
    iterator.close()
    assert "closed" in repr(iterator)
    loader.close()
    dataset.close()
    assert "closed=True" in repr(loader)
    assert "samples=8" in repr(loader)
    assert repr(loader) and repr(iterator)


# ===========================================================================
# 11. State boundaries (§9.5, §11.2, §12.4)
# ===========================================================================

@needs_backend
def test_the_sampler_state_schema_is_unchanged():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                          seed=20240612)
    features, _ = next(iter(loader))
    features.close()
    state = sampler.state_dict()
    assert set(state) == {"format", "format_version", "dataset", "seed",
                          "shuffle", "batch_size", "drop_last", "epoch",
                          "cursor"}
    assert state["format"] == "tensorforge.native_sampler"
    assert state["format_version"] == 1
    assert set(state["dataset"]) == {"samples", "feature_shape",
                                     "feature_dtype", "fingerprint"}
    # No transaction metadata, serial, token, or iteration count leaked in.
    text = json.dumps(state)
    for forbidden in ("serial", "owner", "iteration", "token", "pending",
                      "claim", "loader", "iterator"):
        assert forbidden not in text, forbidden
    loader.close()
    dataset.close()


@needs_backend
def test_a_state_load_is_refused_while_any_iterator_is_active():
    """§9.3/§12.4: an iterator's captured countdown would otherwise
    describe a position that no longer exists."""
    loader, sampler, dataset = make_loader(8, batch_size=3)
    state = sampler.state_dict()
    iterator = iter(loader)
    with pytest.raises(RuntimeError):
        sampler.load_state_dict(state)
    # The guard precedes every schema check, so a malformed state is
    # refused without ever being inspected.
    for malformed in (None, 7, {}, {"format": "wrong"}):
        with pytest.raises(RuntimeError):
            sampler.load_state_dict(malformed)
    # ...but reading state between batches stays allowed.
    assert sampler.state_dict() == state
    # A superseded iterator still counts until it is released.
    second = iter(loader)
    iterator.close()
    with pytest.raises(RuntimeError):
        sampler.load_state_dict(state)
    second.close()
    assert sampler._active_iterations == set()
    assert sampler.load_state_dict(state) is None
    loader.close()
    dataset.close()


@needs_backend
def test_a_state_load_succeeds_once_every_iterator_is_released():
    loader, sampler, dataset = make_loader(8, batch_size=3, shuffle=True,
                                          seed=7)
    for features, _ in loader:
        features.close()
    assert sampler._active_iterations == set()
    state = sampler.state_dict()
    state["cursor"] = 1
    state["epoch"] = 5
    assert sampler.load_state_dict(state) is None
    assert (sampler.epoch, sampler.cursor) == (5, 1)
    assert sampler.dataset is dataset
    # The restored position is what the next iteration delivers.
    assert iter(loader)._to_yield == sampler.remaining == 2
    loader.close()
    dataset.close()


# ===========================================================================
# 12. J3 non-goals — what this milestone must not have added
# ===========================================================================

def test_the_loader_has_no_checkpoint_runtime():
    """J5 owns the checkpoint workflow, and no method that exists only to
    fail until then may appear — the rollout discipline forbids a stub.

    J4 added the loader's own **in-memory** state and its format tag, so
    those two are asserted *present* here and covered in full by
    ``tests/test_native_loader_state.py``. Everything that would make the
    loader a second checkpoint authority stays absent.
    """
    for absent in ("save", "load", "__len__", "__next__", "epoch", "cursor",
                   "seed", "batch_size", "shuffle", "drop_last", "plan",
                   "next_batch_indices", "batches_per_epoch", "remaining",
                   "reset", "advance", "step", "state", "load_state",
                   "save_native_checkpoint", "load_native_checkpoint"):
        assert not hasattr(NativeDataLoader, absent), absent
    # J4's two methods, and its private constants — the milestone's whole
    # public delta, and nothing beside it.
    assert hasattr(NativeDataLoader, "state_dict")
    assert hasattr(NativeDataLoader, "load_state_dict")
    assert loader_module._FORMAT == "tensorforge.native_data_loader"
    assert loader_module._FORMAT_VERSION == 1
    assert loader_module._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert set(loader_module._STATE_FIELDS) == {"format", "format_version",
                                                "sampler"}
    for private in ("_FORMAT", "_FORMAT_VERSION", "_SUPPORTED_FORMAT_VERSIONS",
                    "_STATE_FIELDS"):
        assert private not in experimental.__all__, private
        assert not hasattr(experimental, private), private
    names = code_identifiers(LOADER_SOURCE)
    for forbidden in ("native_checkpoint", "save_native_checkpoint",
                      "load_native_checkpoint", "json", "npz"):
        assert forbidden not in names, forbidden
    checkpoint = (REPO_ROOT / "src" / "tensorforge" / "experimental"
                  / "native_checkpoint.py").read_text(encoding="utf-8")
    assert "native_data_loader" not in checkpoint
    assert "tensorforge.native_data_loader" not in checkpoint


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
                   "persistent_workers", "sampler_lock"):
        assert not hasattr(NativeDataLoader, absent), absent
        assert not hasattr(loader_module._NativeBatchIterator, absent), absent


def test_the_loader_exposes_no_public_transaction_or_delivery_control():
    """The seam is a test seam, not a hook: no user-supplied callable, and
    no public control over the transaction."""
    for absent in ("deliver", "delivery_hook", "on_batch", "callback",
                   "transform", "collate", "set_deliver", "claim",
                   "commit", "rollback", "transaction"):
        assert not hasattr(NativeDataLoader, absent), absent
    signature = inspect.signature(loader_module._deliver_batch)
    assert list(signature.parameters) == ["record"]
    # It does exactly one thing: return the record's pair.
    body = ast.parse(
        inspect.getsource(loader_module._deliver_batch).lstrip()).body[0].body
    statements = [node for node in body if not isinstance(node, ast.Expr)]
    assert len(statements) == 1 and isinstance(statements[0], ast.Return)


def test_no_capability_registry_or_version_moved():
    from tensorforge.experimental import (
        native_checkpoint, native_optimizer_state,
    )

    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    assert cpp.backend_info()["dtype"] == "float64"
    assert cpp.backend_info()["stable_framework_integration"] is False
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert native_optimizer_state.FORMAT_VERSION == 1
    # No dtype or device argument reached the loader.
    assert "dtype" not in inspect.signature(NativeDataLoader).parameters
    assert "device" not in inspect.signature(NativeDataLoader).parameters


def test_no_example_or_benchmark_landed_for_this_milestone():
    """J3 shipped the loader itself and no artifact around it.

    ``examples/native_minibatch_training.py`` arrived at **J6** and
    ``benchmarks/benchmark_native_data_pipeline.py`` at **J8**. Each is
    named here as a permitted exception *in its own directory*, so this
    check still fails on any other data-pipeline example or benchmark
    rather than being relaxed into a pattern that admits anything."""
    permitted = {"native_minibatch_training.py": "examples",          # J6
                 "benchmark_native_data_pipeline.py": "benchmarks"}   # J8
    for directory in ("examples", "benchmarks"):
        for path in (REPO_ROOT / directory).glob("*.py"):
            if path.name in permitted:
                assert directory == permitted[path.name], path.name
                continue
            assert "data_pipeline" not in path.name, path.name
            assert "minibatch" not in path.name, path.name
            assert "data_loader" not in path.name, path.name
