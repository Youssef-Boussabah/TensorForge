"""NativeBatchSampler — the deterministic native batch planner (Phase J,
milestone J2; docs/native_data_pipeline_design.md §3.4, §7, §8, §11.2,
§11.4, §11.5, §12.3, §12.4, §15.2, §16, §18-§20).

What this module proves:

* **§8 derivation** — the committed §8.9 reference vectors as **hard-coded
  known answers**, written here as literals rather than generated from the
  production helper; the downward Fisher-Yates direction; the unbiased
  rejection rule, forced directly; and draw accounting.
* **§8.8 live equivalence** — that the Python finalizer and golden draw
  schedule really are the shipped C++ ones, by predicting the compiled
  ``tf_core_dropout_forward`` kernel's keep/drop pattern from them, with a
  non-vacuity control proving the prediction is falsifiable — while
  keeping the sampler's *own* domain-separated key schedule distinct from
  Dropout's, which is asserted rather than glossed.
* **§7 architecture** — every §7.6 boundary, the position semantics, and
  the §7.7 no-consumption properties in both directions.
* **§11/§12 state** — the exact schema, JSON compatibility, checkpoint
  metadata-validator compatibility, the exact validation order, and
  transactional loading whose rejection leaves an observably identical
  sampler.

**Not tested here, because it does not exist:** iteration, batch
delivery, successful-delivery cursor advancement, batch materialization,
loader state, and checkpoint loader-state integration. ``NativeDataLoader``
(J3) has not started, and §11 of this module asserts its absence.

No test here asserts a complete error message, a dict ordering, a timing,
or a GC event. Where two faults raise the same exception type, precedence
is probed with a short field keyword — the J1 idiom — never a whole
message.

Selector: python -m pytest -q tests/test_native_sampler.py
"""

import ast
import gc
import inspect
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest

import tensorforge
import tensorforge.experimental as experimental
from tensorforge.backends import cpp
from tensorforge.experimental import NativeBatchSampler, NativeTensorDataset
from tensorforge.experimental import _native_permutation as perm
from tensorforge.experimental import native_sampler as sampler_module
from tensorforge.experimental import native_checkpoint as native_checkpoint_module
from tensorforge.experimental import native_generator as native_generator_module

REPO_ROOT = Path(__file__).resolve().parent.parent

# Only the §8.8 live cross-check needs the compiled library. Everything
# else — the derivation, planning, state, and the whole lifecycle — is
# pure Python over integers and stays provable on a machine with no C++
# compiler. J2 allocates nothing native anywhere.
needs_backend = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

UINT64_MAX = 2**64 - 1

# The four reference seeds of §8.9: zero, a small value, a nontrivial
# large one, and the accepted upper bound of the seed domain.
SEEDS = (0, 7, 0xFEDCBA9876543210, 0xFFFFFFFFFFFFFFFF)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    deterministic instrumentation for native-allocation lifetime (the
    Phase-C..J precedent). J2 must never move this count at all."""
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
    gc.collect()
    return len(live_storages)


def make_dataset(samples=8, width=2, dtype=None):
    """A dataset of ``samples`` rows. Values are distinct so a fingerprint
    change is detectable, but nothing here depends on them."""
    features = np.arange(samples * width, dtype=np.float64)
    features = features.reshape(samples, width)
    targets = np.arange(samples, dtype=np.int64) % 3
    return NativeTensorDataset(features, targets, dtype=dtype)


def make_sampler(samples=8, **kwargs):
    kwargs.setdefault("batch_size", 3)
    return NativeBatchSampler(make_dataset(samples), **kwargs)


def code_identifiers(relative):
    """Every identifier the module's **executable code** names — imports,
    names, and attributes.

    Source-text scans are the obvious way to assert "this module never
    touches X", and they are wrong here: these modules explain at length
    what they deliberately do *not* do, so a prose mention of
    ``NativeGenerator`` or ``ctypes`` would fail a substring check that is
    supposed to be about behavior. Reading the AST asks the question that
    was meant — docstrings and comments carry no identifier.
    """
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
            names.update(argument.arg
                         for argument in node.args.args + node.args.kwonlyargs)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
    return names


SAMPLER_SOURCE = "src/tensorforge/experimental/native_sampler.py"
PERMUTATION_SOURCE = "src/tensorforge/experimental/_native_permutation.py"


def test_the_identifier_scanner_can_actually_find_something():
    """Negative control for every scan below: a checker that silently
    stopped matching would pass forever."""
    names = code_identifiers(SAMPLER_SOURCE)
    assert "NativeBatchSampler" in names
    assert "_validate_uint64" in names
    assert "native_generator" in names
    assert "ctypes" not in names
    permutation = code_identifiers(PERMUTATION_SOURCE)
    assert "splitmix64_mix" in permutation
    assert "MASK" in permutation
    # ...and it really does see an import that is present elsewhere.
    assert "numpy" in code_identifiers(
        "src/tensorforge/experimental/native_dataset.py")


def observable(sampler):
    """Every publicly observable fact about a sampler, as one comparable
    value. Used before and after every operation that must change
    nothing. The permutation cache is deliberately **absent** — it is not
    state (§7.8) — and the planning methods are included, because "the
    behavior did not change" is the property that actually matters."""
    return (
        id(sampler.dataset),
        sampler.batch_size, sampler.shuffle, sampler.seed,
        sampler.drop_last, sampler.epoch, sampler.cursor,
        sampler.batches_per_epoch, sampler.remaining,
        sampler.epoch_permutation(),
        sampler.plan(),
        sampler.next_batch_indices(),
        json.dumps(sampler.state_dict(), sort_keys=True),
        repr(sampler),
    )


# ===========================================================================
# 1. Public API, export inventory, and stable/native isolation
# ===========================================================================

def test_the_sampler_module_and_class_exist_where_the_contract_says():
    assert sampler_module.__name__ == "tensorforge.experimental.native_sampler"
    assert sampler_module.NativeBatchSampler is NativeBatchSampler
    assert (REPO_ROOT / "src" / "tensorforge" / "experimental"
            / "native_sampler.py").is_file()


def test_j2_added_exactly_one_public_experimental_name():
    """The J2 exit gate over the live inventory: ``__all__`` grew from 23
    names to 24, by ``NativeBatchSampler`` and nothing else."""
    post_j1 = {
        "NativeTensor", "NativeGenerator", "NativeParameter",
        "NativeParameterRegistry", "NativeModule", "NativeLinear",
        "NativeReLU", "NativeFlatten", "NativeConv2d", "NativeMaxPool2d",
        "NativeSequential", "NativeLayerNorm", "NativeBatchNorm1d",
        "NativeBatchNorm2d", "NativeDropout", "NativeMSELoss",
        "NativeCrossEntropyLoss", "native_accuracy", "NativeSGD",
        "NativeAdam", "save_native_checkpoint", "load_native_checkpoint",
        "NativeTensorDataset",
    }
    assert len(post_j1) == 23
    live = set(experimental.__all__)
    assert len(experimental.__all__) == len(live), "duplicate export"
    assert len(experimental.__all__) == 24
    assert live - post_j1 == {"NativeBatchSampler"}
    assert post_j1 - live == set()
    # J1's name is still exported, and reachable.
    assert experimental.NativeTensorDataset is NativeTensorDataset
    assert experimental.NativeBatchSampler is NativeBatchSampler


def test_the_loader_milestone_has_not_landed():
    """J3's name and module stay absent: an exported name whose class does
    not work is exactly the over-claim the rollout discipline prevents."""
    assert not hasattr(experimental, "NativeDataLoader")
    assert "NativeDataLoader" not in experimental.__all__
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    assert not (package / "native_data_loader.py").exists()


def test_the_permutation_helper_module_stays_private():
    """§3.2: a public bit-generation API would be a second RNG surface
    beside NativeGenerator, which §20 forbids."""
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    assert (package / "_native_permutation.py").is_file()
    assert perm.__name__.rsplit(".", 1)[-1].startswith("_")
    for name in ("_native_permutation", "splitmix64_mix", "epoch_key",
                 "draw_bits", "bounded", "permutation", "sample_order",
                 "batch_plan", "batches_per_epoch"):
        assert name not in experimental.__all__, name
        assert not hasattr(tensorforge, name), name
    # No helper is re-exported from the sampler module's public surface
    # either: reaching one requires naming the private module.
    assert not hasattr(sampler_module, "permutation")
    assert not hasattr(sampler_module, "draw_bits")


def test_nothing_from_this_milestone_entered_the_stable_public_api():
    for name in ("NativeBatchSampler", "NativeTensorDataset",
                 "NativeDataLoader", "_native_permutation"):
        assert not hasattr(tensorforge, name), name
        assert name not in tensorforge.__all__, name
    # The stable mini-batch iterator is untouched and stays stable-only.
    assert hasattr(tensorforge, "batches")
    assert "batches" in tensorforge.__all__


def test_importing_stable_tensorforge_stays_native_lazy():
    """§18: the stable line never imports the experimental one, and the
    Phase-J modules live entirely under ``tensorforge.experimental``."""
    stable = REPO_ROOT / "src" / "tensorforge"
    offenders = [
        path.name for path in sorted(stable.glob("*.py"))
        if re.search(r"^\s*(from|import)\s+.*\bexperimental\b",
                     path.read_text(encoding="utf-8"), re.M)
    ]
    assert offenders == []
    for module in (SAMPLER_SOURCE, PERMUTATION_SOURCE):
        names = code_identifiers(module)
        assert "batches" not in names, module
        assert "train_test_split" not in names, module
        assert "Tensor" not in names, module


def test_the_derivation_module_reaches_no_ctypes_or_native_layer():
    """§3.2/§8.1: ``_native_permutation`` is ordinary Python integer
    arithmetic, with no import of ctypes, the backends package, a storage
    class, a tensor, a generator, NumPy, or a random library."""
    tree = ast.parse(
        (REPO_ROOT / PERMUTATION_SOURCE).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    # It imports **nothing at all**: the whole derivation is built-in
    # integer and sequence arithmetic.
    assert imported == [], imported
    names = code_identifiers(PERMUTATION_SOURCE)
    for forbidden in ("ctypes", "backends", "cpp", "NativeStorage",
                      "NativeTensor", "NativeGenerator", "numpy", "np",
                      "random", "secrets", "time", "environ", "getenv",
                      "hash", "id"):
        assert forbidden not in names, forbidden
    # No floating-point arithmetic anywhere in the derivation.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            assert not isinstance(node.value, float), node.value


# ===========================================================================
# 2. Constructor validation (§7.2, §12.3)
# ===========================================================================

@pytest.mark.parametrize("bad", [
    None, 7, "dataset", object(), [1, 2, 3], np.zeros((4, 2)),
    NativeTensorDataset,
])
def test_the_dataset_must_be_a_native_tensor_dataset(bad):
    with pytest.raises(TypeError):
        NativeBatchSampler(bad, batch_size=2)


def test_a_closed_dataset_is_accepted_and_planning_keeps_working():
    """§7.2: ``samples`` and the identity metadata survive close, the
    sampler needs nothing else, and refusing here would create a
    lifecycle rule with no purpose."""
    dataset = make_dataset(6)
    dataset.close()
    assert dataset.closed
    sampler = NativeBatchSampler(dataset, batch_size=2, shuffle=True, seed=7)
    assert sampler.dataset is dataset
    assert len(sampler.epoch_permutation()) == 6
    assert len(sampler.plan()) == 3
    assert sampler.next_batch_indices() == sampler.plan()[0]
    assert sampler.state_dict()["dataset"]["samples"] == 6
    assert "closed" not in repr(sampler)


def test_batch_size_is_required_and_keyword_only():
    dataset = make_dataset(4)
    with pytest.raises(TypeError):
        NativeBatchSampler(dataset)
    with pytest.raises(TypeError):
        NativeBatchSampler(dataset, 2)
    signature = inspect.signature(NativeBatchSampler)
    for name in ("batch_size", "shuffle", "seed", "drop_last"):
        assert (signature.parameters[name].kind
                is inspect.Parameter.KEYWORD_ONLY), name
    assert signature.parameters["batch_size"].default is inspect.Parameter.empty
    assert signature.parameters["shuffle"].default is False
    assert signature.parameters["seed"].default == 0
    assert signature.parameters["drop_last"].default is False


@pytest.mark.parametrize("bad", [
    True, False, 2.0, "2", None, np.int64(2), np.int32(2), [2],
])
def test_batch_size_rejects_every_inexact_int(bad):
    with pytest.raises(TypeError):
        NativeBatchSampler(make_dataset(4), batch_size=bad)


@pytest.mark.parametrize("bad", [0, -1, -1000])
def test_batch_size_below_one_is_a_value_error(bad):
    with pytest.raises(ValueError):
        NativeBatchSampler(make_dataset(4), batch_size=bad)


def test_batch_size_has_no_upper_bound_when_the_tail_is_kept():
    """§7.2: ``batch_size > samples`` with ``drop_last=False`` is legal and
    gives one short batch. The platform limit is reached where a *native*
    batch is allocated, which J2 never does."""
    sampler = NativeBatchSampler(make_dataset(4), batch_size=10_000)
    assert sampler.batches_per_epoch == 1
    assert sampler.plan() == ((0, 1, 2, 3),)
    assert sampler.next_batch_indices() == (0, 1, 2, 3)


@pytest.mark.parametrize("field", ["shuffle", "drop_last"])
@pytest.mark.parametrize("bad", [
    0, 1, "", "yes", None, np.bool_(True), np.bool_(False), 1.0, [], object(),
])
def test_the_boolean_flags_reject_every_non_bool(field, bad):
    with pytest.raises(TypeError):
        NativeBatchSampler(make_dataset(4), batch_size=2, **{field: bad})


@pytest.mark.parametrize("field", ["shuffle", "drop_last"])
def test_the_boolean_flags_accept_exactly_true_and_false(field):
    for value in (True, False):
        sampler = NativeBatchSampler(make_dataset(4), batch_size=2,
                                     **{field: value})
        assert getattr(sampler, field) is value


@pytest.mark.parametrize("bad", [
    True, False, 1.0, "7", None, np.uint64(7), np.int64(7), object(),
])
def test_the_seed_rejects_every_inexact_int(bad):
    with pytest.raises(TypeError):
        NativeBatchSampler(make_dataset(4), batch_size=2, seed=bad)


@pytest.mark.parametrize("bad", [-1, -(2**64), 2**64, 2**64 + 1, 2**128])
def test_the_seed_domain_is_exactly_unsigned_sixty_four_bit(bad):
    with pytest.raises(ValueError):
        NativeBatchSampler(make_dataset(4), batch_size=2, seed=bad)


@pytest.mark.parametrize("good", [0, 1, UINT64_MAX, UINT64_MAX - 1])
def test_the_seed_boundaries_are_accepted(good):
    assert NativeBatchSampler(make_dataset(4), batch_size=2,
                              seed=good).seed == good


def test_the_seed_uses_the_same_validator_as_the_generator():
    """§8.3: the phase invents no second seed contract. The sampler holds
    a plain integer in the *same* domain, checked by the *same* function —
    shared rather than restated, so the two cannot drift."""
    assert (sampler_module._validate_uint64
            is native_generator_module._validate_uint64)
    assert sampler_module.UINT64_MAX == native_generator_module.UINT64_MAX
    assert sampler_module.UINT64_MAX == UINT64_MAX
    # ...and the two really do agree at every boundary, in both directions.
    for value in (0, 1, UINT64_MAX):
        assert NativeBatchSampler(make_dataset(4), batch_size=2,
                                  seed=value).seed == value
        assert native_generator_module.NativeGenerator(seed=value).seed == value
    for bad, error in ((-1, ValueError), (2**64, ValueError),
                       (True, TypeError), (1.0, TypeError)):
        with pytest.raises(error):
            NativeBatchSampler(make_dataset(4), batch_size=2, seed=bad)
        with pytest.raises(error):
            native_generator_module.NativeGenerator(seed=bad)


def test_the_zero_batch_configuration_is_rejected_at_construction():
    """§7.5: with ``drop_last=True`` and ``batch_size > samples`` an epoch
    would hold no batch at all, so the position could never advance and
    iteration would spin forever. Refusing here is what makes
    ``batches_per_epoch >= 1`` an invariant."""
    with pytest.raises(ValueError) as excinfo:
        NativeBatchSampler(make_dataset(4), batch_size=5, drop_last=True)
    message = str(excinfo.value)
    assert "5" in message and "4" in message
    # The equality boundary is legal: batch_size == samples gives one batch.
    sampler = NativeBatchSampler(make_dataset(4), batch_size=4, drop_last=True)
    assert sampler.batches_per_epoch == 1
    # ...and the same configuration without drop_last is legal too.
    kept = NativeBatchSampler(make_dataset(4), batch_size=5, drop_last=False)
    assert kept.batches_per_epoch == 1


# ``VALID`` is replaced by a real dataset inside the test; every other
# value is passed through as the (invalid) dataset argument.
VALID = "<a real dataset>"


@pytest.mark.parametrize("dataset,kwargs,expected,reason", [
    pytest.param(None, dict(batch_size=0), TypeError,
                 "dataset before batch size", id="dataset-first"),
    pytest.param(None, dict(batch_size="x", shuffle=1, seed=-1), TypeError,
                 "dataset before everything", id="dataset-before-all"),
    pytest.param(VALID, dict(batch_size=0, shuffle="yes"), ValueError,
                 "batch size before shuffle", id="batch-size-before-shuffle"),
    pytest.param(VALID, dict(batch_size=2, shuffle="yes", seed=-1), TypeError,
                 "shuffle before seed", id="shuffle-before-seed"),
    pytest.param(VALID, dict(batch_size=2, seed=-1, drop_last="yes"),
                 ValueError, "seed range before drop_last",
                 id="seed-before-drop-last"),
    pytest.param(VALID, dict(batch_size=99, seed="x", drop_last=True),
                 TypeError, "seed before the joint rule",
                 id="seed-before-joint-rule"),
    pytest.param(VALID, dict(batch_size=99, shuffle=0, drop_last=True),
                 TypeError, "shuffle before the joint rule",
                 id="shuffle-before-joint-rule"),
    pytest.param(VALID, dict(batch_size=0, drop_last="yes"), ValueError,
                 "batch size range before drop_last type",
                 id="batch-size-before-drop-last"),
])
def test_the_construction_precedence_is_exactly_the_contracted_order(
        dataset, kwargs, expected, reason):
    """§12.3, probed by exception *type* so no message is a contract: a
    caller who got two things wrong is told about the more basic one."""
    if dataset is VALID:
        dataset = make_dataset(4)
    with pytest.raises(expected):
        NativeBatchSampler(dataset, **kwargs)


def test_a_fresh_sampler_starts_at_the_origin_position():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    assert sampler.epoch == 0
    assert sampler.cursor == 0
    assert sampler.remaining == sampler.batches_per_epoch


@pytest.mark.parametrize("bad", ["epoch", "cursor", "position", "device",
                                 "dtype", "generator", "num_workers"])
def test_no_position_device_or_dtype_argument_exists(bad):
    """§7.2, §19.7: a position arrives only through ``load_state_dict``,
    and the sampler owns no dtype-bearing state, so a ``dtype`` argument
    would be a second authority that could disagree with the data."""
    assert bad not in inspect.signature(NativeBatchSampler).parameters
    with pytest.raises(TypeError):
        NativeBatchSampler(make_dataset(4), batch_size=2, **{bad: 1})


def test_the_dataset_is_held_by_identity_and_never_copied():
    dataset = make_dataset(6)
    sampler = NativeBatchSampler(dataset, batch_size=2)
    assert sampler.dataset is dataset
    assert sampler.state_dict()["dataset"] == dataset.identity()
    # ...and the identity dict handed out is a fresh container, not the
    # dataset's own or the sampler's.
    assert sampler.state_dict()["dataset"] is not dataset.identity()


def test_the_public_configuration_and_position_are_read_only():
    sampler = make_sampler(8)
    for name in ("dataset", "batch_size", "shuffle", "seed", "drop_last",
                 "epoch", "cursor", "batches_per_epoch", "remaining"):
        with pytest.raises(AttributeError):
            setattr(sampler, name, 1)
    # __slots__, so no attribute can be injected either.
    with pytest.raises(AttributeError):
        sampler.anything = 1


# ===========================================================================
# 3. Position, batch counts, and every §7.6 boundary
# ===========================================================================

@pytest.mark.parametrize("samples,batch_size,drop_last,expected", [
    (1, 1, False, 1), (1, 1, True, 1),
    (8, 4, False, 2), (8, 4, True, 2),          # exactly divisible
    (8, 3, False, 3), (8, 3, True, 2),          # non-divisible
    (5, 2, False, 3), (5, 2, True, 2),
    (7, 1, False, 7), (7, 1, True, 7),
    (4, 5, False, 1),                           # short batch, tail kept
    (4, 4, True, 1),                            # equality boundary
])
def test_batches_per_epoch_at_every_boundary(samples, batch_size, drop_last,
                                             expected):
    sampler = NativeBatchSampler(make_dataset(samples), batch_size=batch_size,
                                 drop_last=drop_last)
    assert sampler.batches_per_epoch == expected
    assert sampler.batches_per_epoch >= 1
    assert len(sampler.plan()) == expected


def test_the_final_batch_is_short_when_the_tail_is_kept():
    sampler = NativeBatchSampler(make_dataset(8), batch_size=3)
    plan = sampler.plan()
    assert [len(batch) for batch in plan] == [3, 3, 2]
    assert plan == ((0, 1, 2), (3, 4, 5), (6, 7))


def test_the_tail_is_omitted_when_drop_last_is_true():
    sampler = NativeBatchSampler(make_dataset(8), batch_size=3, drop_last=True)
    plan = sampler.plan()
    assert [len(batch) for batch in plan] == [3, 3]
    assert plan == ((0, 1, 2), (3, 4, 5))
    # The dropped tail is not systematically excluded: a different epoch
    # has a different permutation (§7.6).
    shuffled = NativeBatchSampler(make_dataset(8), batch_size=3,
                                  drop_last=True, shuffle=True, seed=7)
    assert shuffled.plan(0) != shuffled.plan(1)


def test_drop_last_changes_nothing_when_the_division_is_exact():
    kept = NativeBatchSampler(make_dataset(8), batch_size=4, shuffle=True,
                              seed=7)
    dropped = NativeBatchSampler(make_dataset(8), batch_size=4, shuffle=True,
                                 seed=7, drop_last=True)
    assert kept.plan() == dropped.plan()
    assert kept.batches_per_epoch == dropped.batches_per_epoch == 2


def test_a_one_sample_dataset_gives_one_batch_at_every_configuration():
    for drop_last in (False, True):
        for shuffle in (False, True):
            sampler = NativeBatchSampler(make_dataset(1), batch_size=1,
                                         shuffle=shuffle, seed=7,
                                         drop_last=drop_last)
            assert sampler.batches_per_epoch == 1
            assert sampler.epoch_permutation() == (0,)
            assert sampler.plan() == ((0,),)
            assert sampler.next_batch_indices() == (0,)


def test_remaining_is_the_batches_left_in_the_active_epoch():
    sampler = make_sampler(8, batch_size=3)
    assert sampler.remaining == sampler.batches_per_epoch == 3
    # A canonical state never stores cursor == batches_per_epoch, so
    # ``remaining`` is always in [1, batches_per_epoch].
    for cursor in range(sampler.batches_per_epoch):
        state = sampler.state_dict()
        state["cursor"] = cursor
        sampler.load_state_dict(state)
        assert sampler.remaining == 3 - cursor
        assert 1 <= sampler.remaining <= sampler.batches_per_epoch


def test_the_sampler_owns_nothing_releasable_and_has_no_close():
    """§15.2: a ``close()`` would advertise a lifetime this object does
    not have — NativeGenerator's stated reason, applied rather than
    replaced."""
    sampler = make_sampler(8)
    for name in ("close", "closed", "__enter__", "__exit__", "__del__"):
        assert not hasattr(sampler, name), name
    with pytest.raises(TypeError):
        with sampler:
            pass


def test_no_iteration_or_public_advance_surface_exists():
    """J2 is a planner and a state holder. J3 owns iteration and
    successful-delivery advancement."""
    sampler = make_sampler(8)
    for name in ("__iter__", "__next__", "next", "advance", "step",
                 "reset", "next_epoch", "advance_epoch", "advance_cursor",
                 "consume", "deliver", "_deliver_batch", "__len__"):
        assert not hasattr(sampler, name), name
    with pytest.raises(TypeError):
        iter(sampler)
    with pytest.raises(TypeError):
        len(sampler)


# ===========================================================================
# 4. The derivation: constants, and the §8.9 known answers
# ===========================================================================

def test_the_python_constants_are_the_shipped_cpp_constants():
    """§8.2: the algorithm is reused, not replaced. The constants are read
    out of the C++ header and source, so a change to either side fails."""
    header = (REPO_ROOT / "cpp" / "include"
              / "tf_random_internal.h").read_text(encoding="utf-8")
    source = (REPO_ROOT / "cpp" / "src" / "random.cpp").read_text(
        encoding="utf-8")
    assert perm.GOLDEN == 0x9E3779B97F4A7C15
    assert perm.MIX_MUL_1 == 0xBF58476D1CE4E5B9
    assert perm.MIX_MUL_2 == 0x94D049BB133111EB
    assert perm.MASK == 2**64 - 1
    assert f"{perm.GOLDEN:#018X}"[2:] in header.upper()
    for constant in (perm.MIX_MUL_1, perm.MIX_MUL_2):
        assert f"{constant:X}" in source.upper(), hex(constant)
    # The shifts, in order, are the locked 30 / 27 / 31.
    assert re.search(r">>\s*30.*?>>\s*27.*?>>\s*31",
                     source, re.S)
    python_source = (REPO_ROOT / "src" / "tensorforge" / "experimental"
                     / "_native_permutation.py").read_text(encoding="utf-8")
    assert re.search(r">>\s*30.*?>>\s*27.*?>>\s*31", python_source, re.S)


def test_the_sampler_domain_constant_is_the_ascii_bytes_tf_sampl():
    """§8.4: one additive constant, and it is domain separation rather
    than a cryptographic claim."""
    assert perm.SAMPLER_DOMAIN == 0x54465F53414D504C
    assert perm.SAMPLER_DOMAIN.to_bytes(8, "big") == b"TF_SAMPL"


@pytest.mark.parametrize("value,expected", [
    (0x0000000000000000, 0x0000000000000000),
    (0x0000000000000001, 0x5692161D100B05E5),
    (0x9E3779B97F4A7C15, 0xE220A8397B1DCDAF),
    (0xFFFFFFFFFFFFFFFF, 0xB4D055FCF2CBBD7B),
])
def test_the_splitmix64_finalizer_reproduces_its_known_answers(value,
                                                               expected):
    """§8.9, hard-coded. These are also the C++ function's answers, so a
    Python implementation that drifts from the kernel fails here first."""
    assert perm.splitmix64_mix(value) == expected


def test_the_finalizer_stays_inside_the_sixty_four_bit_window():
    """Python ints are arbitrary precision, so the explicit ``& MASK`` is
    what makes the width exactly 64 bits on every platform."""
    for value in (0, 1, perm.GOLDEN, UINT64_MAX, 2**64, 2**70 + 3, -1):
        assert 0 <= perm.splitmix64_mix(value) <= UINT64_MAX
    # A value above the window is masked into it rather than widening.
    assert perm.splitmix64_mix(2**64) == perm.splitmix64_mix(0)
    assert perm.splitmix64_mix(2**64 + 1) == perm.splitmix64_mix(1)


EPOCH_KEYS = {
    (0, 0): 0x66F32B8D4EDCDEF0,
    (0, 1): 0xE205B4E09628466F,
    (0, 7): 0x4487E9B41C8E68DF,
    (7, 0): 0xE9D3E585001C46A4,
    (7, 1): 0xED6B991DDB3B74AF,
    (7, 7): 0xEDF383949681C2A9,
    (0xFEDCBA9876543210, 0): 0x5EAC5CE0C7928FA5,
    (0xFEDCBA9876543210, 1): 0x418ADC598C6E56E9,
    (0xFEDCBA9876543210, 7): 0x4D3B0EE9DD189AB2,
    (0xFFFFFFFFFFFFFFFF, 0): 0xA20C5EE669FCA87A,
    (0xFFFFFFFFFFFFFFFF, 1): 0x097D6A1D7039DBCA,
    (0xFFFFFFFFFFFFFFFF, 7): 0xBEF20F97FF2FBF91,
}


@pytest.mark.parametrize("key,expected", sorted(EPOCH_KEYS.items()))
def test_the_epoch_key_reproduces_its_known_answers(key, expected):
    """§8.9, hard-coded rather than generated by the production helper —
    a known-answer set, not a regression convenience."""
    seed, epoch = key
    assert perm.epoch_key(seed, epoch) == expected


def test_the_epoch_key_covers_every_required_seed_and_epoch():
    assert {seed for seed, _ in EPOCH_KEYS} == set(SEEDS)
    assert {epoch for _, epoch in EPOCH_KEYS} == {0, 1, 7}
    assert len(EPOCH_KEYS) == 12


# The complete §8.9 permutation table, keyed by (length, seed, epoch).
PERMUTATIONS = {
    (1, 0, 0): (0,), (1, 0, 7): (0,),
    (1, 7, 0): (0,), (1, 7, 7): (0,),
    (1, 0xFEDCBA9876543210, 0): (0,), (1, 0xFEDCBA9876543210, 7): (0,),
    (1, 0xFFFFFFFFFFFFFFFF, 0): (0,), (1, 0xFFFFFFFFFFFFFFFF, 7): (0,),
    (2, 0, 0): (0, 1), (2, 0, 7): (1, 0),
    (2, 7, 0): (1, 0), (2, 7, 7): (1, 0),
    (2, 0xFEDCBA9876543210, 0): (0, 1), (2, 0xFEDCBA9876543210, 7): (1, 0),
    (2, 0xFFFFFFFFFFFFFFFF, 0): (0, 1), (2, 0xFFFFFFFFFFFFFFFF, 7): (0, 1),
    (5, 0, 0): (1, 0, 3, 4, 2), (5, 0, 7): (2, 1, 3, 0, 4),
    (5, 7, 0): (1, 2, 4, 3, 0), (5, 7, 7): (3, 0, 2, 1, 4),
    (5, 0xFEDCBA9876543210, 0): (2, 1, 0, 4, 3),
    (5, 0xFEDCBA9876543210, 7): (4, 3, 2, 1, 0),
    (5, 0xFFFFFFFFFFFFFFFF, 0): (4, 0, 3, 2, 1),
    (5, 0xFFFFFFFFFFFFFFFF, 7): (3, 2, 0, 4, 1),
    (8, 0, 0): (3, 6, 7, 0, 2, 5, 4, 1),
    (8, 0, 7): (4, 2, 0, 1, 7, 3, 5, 6),
    (8, 7, 0): (7, 5, 4, 0, 1, 3, 6, 2),
    (8, 7, 7): (1, 4, 7, 0, 3, 5, 6, 2),
    (8, 0xFEDCBA9876543210, 0): (1, 0, 5, 2, 6, 7, 4, 3),
    (8, 0xFEDCBA9876543210, 7): (2, 1, 6, 7, 4, 5, 3, 0),
    (8, 0xFFFFFFFFFFFFFFFF, 0): (0, 3, 1, 4, 6, 5, 2, 7),
    (8, 0xFFFFFFFFFFFFFFFF, 7): (6, 7, 1, 2, 5, 0, 4, 3),
}


@pytest.mark.parametrize("key,expected", sorted(PERMUTATIONS.items()))
def test_every_committed_permutation_is_reproduced_exactly(key, expected):
    length, seed, epoch = key
    assert perm.permutation(seed, epoch, length) == expected


def test_the_permutation_table_covers_every_required_combination():
    assert {length for length, _, _ in PERMUTATIONS} == {1, 2, 5, 8}
    assert {seed for _, seed, _ in PERMUTATIONS} == set(SEEDS)
    assert {epoch for _, _, epoch in PERMUTATIONS} == {0, 7}
    assert len(PERMUTATIONS) == 32


@pytest.mark.parametrize("key,expected", sorted(PERMUTATIONS.items()))
def test_every_committed_permutation_is_a_real_permutation(key, expected):
    length = key[0]
    assert sorted(expected) == list(range(length))
    assert len(set(expected)) == length


def test_the_identity_rows_are_legal_and_are_not_special_cased():
    """§8.9: several reference rows *are* the identity — every length-1
    row and four length-2 rows — and that is not a defect. Excluding the
    identity would bias the sampler, so it must never be "fixed"."""
    identity_rows = [key for key, value in PERMUTATIONS.items()
                     if value == tuple(range(key[0]))]
    # Eight length-1 rows plus the length-2 ones that came out identity.
    assert len([key for key in identity_rows if key[0] == 1]) == 8
    assert len([key for key in identity_rows if key[0] == 2]) == 4
    # ...and the production helper really does return them.
    for key in identity_rows:
        length, seed, epoch = key
        assert perm.permutation(seed, epoch, length) == tuple(range(length))


def test_a_permutation_varies_from_epoch_to_epoch():
    """Different epochs give different orders — the property that makes
    each epoch a fresh shuffle rather than a repeat."""
    for seed in SEEDS:
        orders = {perm.permutation(seed, epoch, 8) for epoch in range(6)}
        assert len(orders) >= 5, (seed, orders)
    # ...and different seeds give different orders at one epoch.
    assert len({perm.permutation(seed, 0, 8) for seed in SEEDS}) == 4


@pytest.mark.parametrize("length", [1, 2, 3, 5, 8, 13, 64, 257])
@pytest.mark.parametrize("epoch", [0, 1, 7, 2**63, UINT64_MAX])
def test_every_derived_permutation_is_valid_at_every_length(length, epoch):
    order = perm.permutation(0xFEDCBA9876543210, epoch, length)
    assert isinstance(order, tuple)
    assert sorted(order) == list(range(length))
    assert all(type(index) is int for index in order)


SEQUENTIAL = {1: (0,), 2: (0, 1), 5: (0, 1, 2, 3, 4),
              8: (0, 1, 2, 3, 4, 5, 6, 7)}


@pytest.mark.parametrize("length,expected", sorted(SEQUENTIAL.items()))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("epoch", [0, 1, 7])
def test_sequential_order_is_the_identity_at_every_seed_and_epoch(
        length, expected, seed, epoch):
    """§8.6/§8.9: ``shuffle=False`` is not "a shuffle with a fixed seed";
    it is a different, cheaper branch."""
    assert perm.sample_order(seed, epoch, length, False) == expected


def test_sequential_order_consumes_no_derivation_at_all(monkeypatch):
    """The cheaper branch is proved cheaper: a sequential order must not
    touch the bit path even once."""
    calls = []
    monkeypatch.setattr(perm, "draw_bits",
                        lambda *args: calls.append(args) or 0)
    monkeypatch.setattr(perm, "epoch_key",
                        lambda *args: calls.append(args) or 0)
    for length in (1, 2, 5, 8, 1000):
        for seed in SEEDS:
            assert perm.sample_order(seed, 3, length, False) == tuple(
                range(length))
    assert calls == []
    # ...and the same through the public surface.
    sampler = NativeBatchSampler(make_dataset(8), batch_size=3, seed=7)
    assert sampler.epoch_permutation() == tuple(range(8))
    assert sampler.plan(11) == ((0, 1, 2), (3, 4, 5), (6, 7))
    assert calls == []


# --- Fisher-Yates direction, bounded integers, and draw accounting --------

def test_the_fisher_yates_sweep_runs_downward():
    """§8.6: the upward variant is equally correct and produces different
    permutations from the same bits, so the direction is specification.

    Proved by replaying the exact draws the production helper consumes
    and rebuilding the order both ways: only the downward sweep matches.
    """
    seed, epoch, length = 7, 0, 8
    key = perm.epoch_key(seed, epoch)

    def replay(descending):
        order = list(range(length))
        draws = 0
        span = (range(length - 1, 0, -1) if descending
                else range(0, length - 1))
        for i in span:
            bound = i + 1 if descending else length - i
            j, draws = perm.bounded(key, draws, bound)
            if not descending:
                j += i
            order[i], order[j] = order[j], order[i]
        return tuple(order)

    assert replay(True) == perm.permutation(seed, epoch, length)
    assert replay(True) == PERMUTATIONS[(8, 7, 0)]
    assert replay(False) != replay(True)


def test_bounded_returns_a_value_inside_its_bound_and_advances_the_index():
    for bound in (2, 3, 5, 8, 17, 1000):
        index = 0
        for _ in range(50):
            value, index = perm.bounded(0xABCDEF0123456789, index, bound)
            assert 0 <= value < bound
            assert type(value) is int
    # bound == 1 is degenerate but total: exactly one residue exists.
    value, index = perm.bounded(1234, 0, 1)
    assert (value, index) == (0, 1)


def test_bounded_uses_the_exact_rejection_limit_rather_than_bare_modulo():
    """§8.6: ``bits % bound`` alone is biased whenever ``bound`` does not
    divide ``2**64``. ``limit`` is the largest multiple of ``bound`` that
    fits, so each residue is covered exactly the same number of times."""
    for bound in (3, 5, 7, 1000, 2**32 + 1, 2**63 + 1):
        limit = (1 << 64) - ((1 << 64) % bound)
        assert limit % bound == 0
        assert limit <= 1 << 64
        assert (1 << 64) - limit < bound
    source = inspect.getsource(perm.bounded)
    assert "(1 << 64) % bound" in source
    assert "while True" in source


# A ``bound`` of ``2**63 + 1`` makes ``limit == 2**63 + 1`` too, so roughly
# half of all draws are rejected — the only practical way to reach the
# branch, since at any production bound the rejection probability is below
# ``2**-32``. The keys are plain literals and the expectations were
# computed independently from §8.5's pseudocode, not read off the helper.
FORCED_REJECTION_BOUND = 2**63 + 1
FORCED_REJECTIONS = (
    # (key, rejections before acceptance, accepted residue, final draw index)
    (0, 1, 7960286522194355700, 2),
    (1, 3, 8196980753821780235, 4),
    (2, 4, 5747796768693156649, 5),
)


@pytest.mark.parametrize("key,rejections,residue,final_index",
                         FORCED_REJECTIONS)
def test_the_rejection_branch_is_forced_and_produces_the_expected_residue(
        key, rejections, residue, final_index):
    """§8.9: the rejection branch is part of the specification, and no
    reference case reaches it, so J2 exercises ``bounded`` directly at a
    bound whose ``limit`` is small enough to force one."""
    limit = (1 << 64) - ((1 << 64) % FORCED_REJECTION_BOUND)
    # Non-vacuity: the first ``rejections`` draws really are above the
    # limit, and the next one really is below it. Without this the test
    # could pass on a helper that never rejected anything.
    for index in range(rejections):
        assert perm.draw_bits(key, index) >= limit, index
    accepted = perm.draw_bits(key, rejections)
    assert accepted < limit
    assert accepted % FORCED_REJECTION_BOUND == residue

    value, next_index = perm.bounded(key, 0, FORCED_REJECTION_BOUND)
    assert value == residue
    # Every drawn value counts, accepted and rejected alike: the index
    # advanced past the rejections too.
    assert next_index == final_index == rejections + 1
    assert 0 <= value < FORCED_REJECTION_BOUND


def test_a_rejected_draw_shifts_every_later_draw_by_one():
    """§8.6: counting rejections is what keeps the result a pure function
    of ``(seed, epoch, length)`` regardless of where a rejection lands."""
    key, rejections, _, final_index = FORCED_REJECTIONS[1]
    first, index = perm.bounded(key, 0, FORCED_REJECTION_BOUND)
    assert index == final_index
    # The next bounded call starts exactly where this one stopped, so the
    # sequence is contiguous across a rejection rather than restarting.
    second, index2 = perm.bounded(key, index, 4)
    assert second == perm.draw_bits(key, final_index) % 4
    assert index2 == final_index + 1
    assert rejections > 0


def _count_draws(monkeypatch, function, *args):
    """Run ``function`` with the module's ``draw_bits`` counted."""
    original = perm.draw_bits
    calls = []

    def counting(key, index):
        calls.append(index)
        return original(key, index)

    monkeypatch.setattr(perm, "draw_bits", counting)
    result = function(*args)
    monkeypatch.setattr(perm, "draw_bits", original)
    return result, len(calls)


def test_length_one_consumes_zero_draws(monkeypatch):
    """§8.7: at length 1 there is one order, and consuming a draw to
    discover that would make the accounting depend on a degenerate case."""
    for seed in SEEDS:
        for epoch in (0, 7):
            order, draws = _count_draws(monkeypatch, perm.permutation,
                                        seed, epoch, 1)
            assert order == (0,)
            assert draws == 0


@pytest.mark.parametrize("length", [2, 5, 8, 100, 1000])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("epoch", [0, 1, 2, 3])
def test_the_reference_set_consumes_exactly_length_minus_one_draws(
        monkeypatch, length, seed, epoch):
    """§8.9's draw-count table: **no rejection occurs anywhere in the
    reference set**, so every case takes exactly ``length - 1`` draws."""
    order, draws = _count_draws(monkeypatch, perm.permutation,
                                seed, epoch, length)
    assert draws == length - 1
    assert sorted(order) == list(range(length))


def test_the_draw_counter_can_actually_count():
    """Negative control for the accounting above: a counter that silently
    stopped matching would make every draw-count assertion vacuous."""
    original = perm.draw_bits
    calls = []
    try:
        perm.draw_bits = lambda key, index: calls.append(index) or original(
            key, index)
        perm.permutation(7, 0, 5)
    finally:
        perm.draw_bits = original
    assert len(calls) == 4
    assert calls == [0, 1, 2, 3]


# --- the complete §8.9 batch plans ---------------------------------------

@pytest.mark.parametrize("batch_size,drop_last,expected", [
    (3, False, ((7, 5, 4), (0, 1, 3), (6, 2))),
    (3, True, ((7, 5, 4), (0, 1, 3))),
    (4, False, ((7, 5, 4, 0), (1, 3, 6, 2))),
    (4, True, ((7, 5, 4, 0), (1, 3, 6, 2))),
])
def test_the_committed_length_eight_plans_are_reproduced(batch_size,
                                                         drop_last, expected):
    """§8.9: length 8, seed 7, epoch 0, shuffled — permutation
    ``[7, 5, 4, 0, 1, 3, 6, 2]``."""
    assert perm.batch_plan(7, 0, 8, batch_size, drop_last, True) == expected
    sampler = NativeBatchSampler(make_dataset(8), batch_size=batch_size,
                                 shuffle=True, seed=7, drop_last=drop_last)
    assert sampler.epoch_permutation() == (7, 5, 4, 0, 1, 3, 6, 2)
    assert sampler.plan() == expected


@pytest.mark.parametrize("batch_size,drop_last,expected", [
    (2, False, ((1, 0), (3, 4), (2,))),
    (2, True, ((1, 0), (3, 4))),
    (5, False, ((1, 0, 3, 4, 2),)),
    (5, True, ((1, 0, 3, 4, 2),)),
])
def test_the_committed_length_five_plans_are_reproduced(batch_size, drop_last,
                                                        expected):
    """§8.9: length 5, seed 0, epoch 0, shuffled — permutation
    ``[1, 0, 3, 4, 2]``."""
    assert perm.batch_plan(0, 0, 5, batch_size, drop_last, True) == expected
    sampler = NativeBatchSampler(make_dataset(5), batch_size=batch_size,
                                 shuffle=True, seed=0, drop_last=drop_last)
    assert sampler.epoch_permutation() == (1, 0, 3, 4, 2)
    assert sampler.plan() == expected


@pytest.mark.parametrize("drop_last,expected", [
    (False, ((0, 1), (2, 3), (4,))),
    (True, ((0, 1), (2, 3))),
])
def test_the_committed_sequential_plan_is_reproduced(drop_last, expected):
    """§8.9: length 5, sequential, batch_size 2."""
    for seed in SEEDS:
        assert perm.batch_plan(seed, 0, 5, 2, drop_last, False) == expected
    sampler = NativeBatchSampler(make_dataset(5), batch_size=2,
                                 drop_last=drop_last)
    assert sampler.plan() == expected


def test_a_plan_never_repeats_an_index_and_covers_the_kept_prefix():
    """§10.4: the sampler never produces a duplicate — a permutation
    contains each index exactly once and a batch is a contiguous slice."""
    for drop_last in (False, True):
        sampler = NativeBatchSampler(make_dataset(13), batch_size=4,
                                     shuffle=True, seed=7,
                                     drop_last=drop_last)
        order = sampler.epoch_permutation()
        flat = [index for batch in sampler.plan() for index in batch]
        assert len(flat) == len(set(flat))
        assert flat == list(order[:len(flat)])
        assert set(flat) <= set(range(13))
        if not drop_last:
            assert sorted(flat) == list(range(13))


# ===========================================================================
# 5. §8.8 — live equivalence with the shipped C++ derivation
# ===========================================================================

def _dropout_stream_key(seed, call_index):
    """``tf::dropout_stream_key``, expressed through the **shared** half of
    the sampler's own derivation: the same finalizer and the same golden
    schedule, with ``SAMPLER_DOMAIN`` omitted, which is precisely the one
    difference §8.4 introduces."""
    return perm.splitmix64_mix(
        (seed + perm.GOLDEN * (call_index + 1)) & perm.MASK)


def _predicted_keep(seed, call_index, probability, count):
    """The keep/drop pattern the compiled kernel must produce, predicted
    from ``_native_permutation`` alone."""
    key = _dropout_stream_key(seed, call_index)
    return [
        not ((perm.draw_bits(key, index) >> 11) * 2.0 ** -53 < probability)
        for index in range(count)
    ]


def _observed_keep(values, probability, seed, call_index):
    """The kernel's own answer, through the public Core forward. The input
    is all ones, so an output element is ``0.0`` when dropped and the
    inverted-Dropout scale when kept."""
    core = cpp.NativeTensorCore.from_array(values)
    try:
        out = core.dropout_forward(probability, seed=seed,
                                   call_index=call_index)
        try:
            return [value != 0.0 for value in out.to_numpy()]
        finally:
            out.close()
    finally:
        core.close()


ELEMENTS = 4096
EQUIVALENCE_CASES = [
    (seed, call_index, probability)
    for seed in SEEDS
    for call_index in (0, 1, 5)
    for probability in (0.1, 0.25, 0.5, 0.9)
]


@needs_backend
@pytest.mark.parametrize("seed,call_index,probability", EQUIVALENCE_CASES)
def test_the_python_derivation_predicts_the_compiled_kernel_exactly(
        seed, call_index, probability):
    """§8.8: one algorithm with two implementations can drift, so the
    equality is a **gate**, not an assumption.

    Each element is one bit of evidence about the mixing function, so a
    few thousand elements at several probabilities pin the implementation
    far more tightly than any vector list — and this re-runs on every
    platform the suite runs on, which is exactly where a drift would
    appear.
    """
    values = np.ones(ELEMENTS, dtype=np.float64)
    predicted = _predicted_keep(seed, call_index, probability, ELEMENTS)
    observed = _observed_keep(values, probability, seed, call_index)
    assert predicted == observed
    # Non-vacuity: the comparison must not be trivially satisfied by an
    # all-keep or an all-drop pattern.
    kept = sum(observed)
    assert 0 < kept < ELEMENTS
    assert abs(kept / ELEMENTS - (1.0 - probability)) < 0.05


@needs_backend
def test_the_equivalence_check_can_actually_fail():
    """The non-vacuity control §8.8 requires: a deliberately altered
    constant, shift, or key must make the prediction stop matching. Zero
    mismatches only mean something when the detector is known to work."""
    seed, call_index, probability = 7, 1, 0.5
    values = np.ones(ELEMENTS, dtype=np.float64)
    observed = _observed_keep(values, probability, seed, call_index)
    assert _predicted_keep(seed, call_index, probability, ELEMENTS) == observed

    def altered(mix_constant=perm.MIX_MUL_1, golden=perm.GOLDEN,
                first_shift=30, key_offset=0):
        def mix(x):
            x &= perm.MASK
            x ^= x >> first_shift
            x = (x * mix_constant) & perm.MASK
            x ^= x >> 27
            x = (x * perm.MIX_MUL_2) & perm.MASK
            x ^= x >> 31
            return x

        key = mix((seed + key_offset + golden * (call_index + 1)) & perm.MASK)
        return [
            not ((mix((key + golden * (index + 1)) & perm.MASK) >> 11)
                 * 2.0 ** -53 < probability)
            for index in range(ELEMENTS)
        ]

    # The unaltered spelling reproduces the kernel...
    assert altered() == observed
    # ...and each single mutation breaks it.
    assert altered(mix_constant=perm.MIX_MUL_1 ^ 1) != observed
    assert altered(golden=perm.GOLDEN + 2) != observed
    assert altered(first_shift=29) != observed
    # ...including the domain separator, which is exactly why the sampler's
    # own key schedule is a *different* stream (below).
    assert altered(key_offset=perm.SAMPLER_DOMAIN) != observed


@needs_backend
def test_the_sampler_stream_is_domain_separated_from_the_dropout_stream():
    """§8.4: the shared half is shared and the sampler's half is not. The
    equivalence proof above must not be read as "the sampler and Dropout
    produce the same values" — they deliberately do not."""
    for seed in SEEDS:
        for index in (0, 1, 5, 7):
            dropout = _dropout_stream_key(seed, index)
            sampler = perm.epoch_key(seed, index)
            assert dropout != sampler, (seed, index)
            # The difference is exactly the one additive constant, applied
            # before the shared finalizer — not a second algorithm.
            assert sampler == perm.splitmix64_mix(
                (seed + perm.SAMPLER_DOMAIN
                 + perm.GOLDEN * (index + 1)) & perm.MASK)
    # A caller who deliberately offsets their seed can still align the two,
    # which is why this is domain separation and not a security property.
    assert perm.epoch_key(0, 3) == _dropout_stream_key(perm.SAMPLER_DOMAIN, 3)


@needs_backend
def test_the_sampler_derivation_allocates_no_native_storage(live_storages):
    """§7.7 / J2's invariant: no NativeTensor allocation anywhere in the
    milestone. The equivalence check above is the only test here that
    touches the library at all, and it closes everything it opens."""
    baseline = settled(live_storages)
    sampler = NativeBatchSampler(make_dataset(64), batch_size=7, shuffle=True,
                                 seed=0xFEDCBA9876543210)
    sampler.epoch_permutation()
    sampler.plan()
    sampler.plan(99)
    sampler.next_batch_indices()
    sampler.state_dict()
    sampler.load_state_dict(sampler.state_dict())
    repr(sampler)
    assert settled(live_storages) == baseline
    del sampler
    assert settled(live_storages) == baseline


# ===========================================================================
# 6. Planning: purity, and no consumption (§7.7)
# ===========================================================================

def test_the_current_and_explicit_epoch_agree():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    assert sampler.epoch_permutation() == sampler.epoch_permutation(0)
    assert sampler.plan() == sampler.plan(0)
    state = sampler.state_dict()
    state["epoch"] = 4
    sampler.load_state_dict(state)
    assert sampler.epoch_permutation() == sampler.epoch_permutation(4)
    assert sampler.plan() == sampler.plan(4)
    assert sampler.epoch_permutation() != sampler.epoch_permutation(0)


def test_the_plan_and_permutation_are_tuples_all_the_way_down():
    """§11.4/§12: no list, no NumPy array, and nothing mutable in the
    public structures — a caller cannot reach the sampler's order."""
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    order = sampler.epoch_permutation()
    assert type(order) is tuple
    assert all(type(index) is int for index in order)
    plan = sampler.plan()
    assert type(plan) is tuple
    for batch in plan:
        assert type(batch) is tuple
        assert all(type(index) is int for index in batch)
    assert type(sampler.next_batch_indices()) is tuple
    assert len(order) == sampler.dataset.samples


def test_next_batch_indices_is_exactly_the_plan_entry_at_the_cursor():
    for drop_last in (False, True):
        for shuffle in (False, True):
            sampler = NativeBatchSampler(make_dataset(11), batch_size=3,
                                         shuffle=shuffle, seed=7,
                                         drop_last=drop_last)
            for cursor in range(sampler.batches_per_epoch):
                state = sampler.state_dict()
                state["cursor"] = cursor
                sampler.load_state_dict(state)
                assert (sampler.next_batch_indices()
                        == sampler.plan(sampler.epoch)[cursor])


@pytest.mark.parametrize("bad", [True, False, 1.0, "0", np.int64(0), [0]])
def test_an_explicit_epoch_must_be_an_exact_int(bad):
    sampler = make_sampler(8, shuffle=True, seed=7)
    for method in (sampler.epoch_permutation, sampler.plan):
        with pytest.raises(TypeError):
            method(bad)


def test_an_epoch_of_none_means_the_active_epoch():
    """``None`` is the documented default, not a rejected value: it is
    what makes ``plan()`` and ``plan(sampler.epoch)`` the same call."""
    sampler = make_sampler(8, shuffle=True, seed=7)
    state = sampler.state_dict()
    state["epoch"] = 6
    sampler.load_state_dict(state)
    assert sampler.epoch_permutation(None) == sampler.epoch_permutation(6)
    assert sampler.plan(None) == sampler.plan(6)
    assert sampler.epoch_permutation() == sampler.epoch_permutation(None)


@pytest.mark.parametrize("bad", [-1, 2**64, 2**70])
def test_an_explicit_epoch_must_be_inside_the_uint64_domain(bad):
    sampler = make_sampler(8, shuffle=True, seed=7)
    for method in (sampler.epoch_permutation, sampler.plan):
        with pytest.raises(ValueError):
            method(bad)


@pytest.mark.parametrize("good", [0, 1, 7, 2**63, UINT64_MAX])
def test_the_explicit_epoch_boundaries_are_accepted(good):
    sampler = make_sampler(8, shuffle=True, seed=7)
    assert len(sampler.epoch_permutation(good)) == 8
    assert len(sampler.plan(good)) == 3


def test_every_inspection_leaves_the_sampler_byte_identical():
    """§7.7: the sampler has no consumable stream, so inspection and
    planning consume nothing — by construction rather than by cleanup."""
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    before = observable(sampler)
    for _ in range(3):
        sampler.epoch_permutation()
        sampler.epoch_permutation(0)
        sampler.epoch_permutation(99)
        sampler.plan()
        sampler.plan(0)
        sampler.plan(2**63)
        sampler.next_batch_indices()
        repr(sampler)
        sampler.state_dict()
        (sampler.epoch, sampler.cursor, sampler.remaining,
         sampler.batches_per_epoch, sampler.seed, sampler.shuffle,
         sampler.batch_size, sampler.drop_last, sampler.dataset)
    assert observable(sampler) == before


def test_calls_in_an_arbitrary_order_give_identical_results():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    first = (sampler.epoch_permutation(), sampler.plan(),
             sampler.next_batch_indices())
    # Interleave arbitrary-epoch work between the repeats.
    for epoch in (5, 0, 2**40, 1):
        sampler.plan(epoch)
        sampler.epoch_permutation(epoch)
    second = (sampler.epoch_permutation(), sampler.plan(),
              sampler.next_batch_indices())
    assert first == second


def test_a_rejected_explicit_epoch_changes_nothing():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    before = observable(sampler)
    for bad in (-1, 2**64, True, "x", 1.5):
        with pytest.raises((TypeError, ValueError)):
            sampler.epoch_permutation(bad)
        with pytest.raises((TypeError, ValueError)):
            sampler.plan(bad)
    assert observable(sampler) == before


def test_planning_touches_no_global_or_generator_random_state():
    """§8.1: no Python global ``random``, no NumPy global RNG, no
    ``Generator``, no ambient mutable state, and no NativeGenerator."""
    random.seed(12345)
    np.random.seed(6789)
    generator = native_generator_module.NativeGenerator(seed=7)
    numpy_generator = np.random.default_rng(11)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    generator_state = generator.state()
    drawn = numpy_generator.integers(0, 1000, size=5).tolist()

    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    for epoch in (None, 0, 3, UINT64_MAX):
        sampler.epoch_permutation(epoch)
        sampler.plan(epoch)
    sampler.next_batch_indices()
    sampler.load_state_dict(sampler.state_dict())

    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    assert np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert generator.state() == generator_state
    assert generator.calls == 0
    # A NumPy Generator's own stream is untouched: it resumes where it was.
    assert np.random.default_rng(11).integers(0, 1000, size=5).tolist() == drawn


def test_two_samplers_with_equal_state_agree_forever():
    """§7.7: equal ``(seed, epoch, cursor, batch_size, drop_last,
    shuffle)`` over datasets of equal length produce identical remaining
    batch-index sequences — the property an exact resume rests on."""
    first = NativeBatchSampler(make_dataset(13), batch_size=4, shuffle=True,
                               seed=0xFEDCBA9876543210)
    # A deliberately *differently configured* second sampler, restored.
    second = NativeBatchSampler(make_dataset(13), batch_size=2, shuffle=False,
                                seed=1, drop_last=True)
    state = first.state_dict()
    state["epoch"], state["cursor"] = 6, 2
    first.load_state_dict(state)
    second.load_state_dict(state)
    assert first.state_dict() == second.state_dict()
    for epoch in range(6, 12):
        assert first.plan(epoch) == second.plan(epoch)
    assert first.next_batch_indices() == second.next_batch_indices()
    # ...and the datasets really are distinct objects with equal content.
    assert first.dataset is not second.dataset
    assert first.dataset.fingerprint == second.dataset.fingerprint


def test_the_permutation_is_independent_of_the_dataset_dtype():
    """§14.4: the permutation is a pure function of ``(seed, epoch,
    samples)`` and carries no dtype at all. Proved in both directions."""
    plans = []
    for dtype in ("float64", "float32"):
        dataset = make_dataset(8, dtype=dtype)
        assert dataset.dtype == dtype
        sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True,
                                     seed=7)
        plans.append(sampler.plan())
        assert sampler.state_dict()["dataset"]["feature_dtype"] == dtype
    assert plans[0] == plans[1] == ((7, 5, 4), (0, 1, 3), (6, 2))


# ===========================================================================
# 7. The permutation cache (§7.8) — not state, and never observable
# ===========================================================================

def test_the_cache_key_is_seed_epoch_and_samples():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    assert sampler._cache_key is None
    order = sampler.epoch_permutation()
    assert sampler._cache_key == (7, 0, 8)
    assert sampler._cache_order == order


def test_repeated_current_epoch_calls_reproduce_exactly():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    first = sampler.epoch_permutation()
    for _ in range(5):
        assert sampler.epoch_permutation() == first
        assert sampler.plan()[0] == first[:3]
        assert sampler.next_batch_indices() == first[:3]


def test_an_arbitrary_epoch_call_never_touches_the_cache():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    current = sampler.epoch_permutation()
    key, cached = sampler._cache_key, sampler._cache_order
    for epoch in (1, 5, 2**63, UINT64_MAX):
        other = sampler.epoch_permutation(epoch)
        assert other != current or epoch == 0
        assert sampler._cache_key == key
        assert sampler._cache_order is cached
    assert sampler.epoch_permutation() == current


def test_deleting_or_clearing_the_cache_changes_no_result():
    """§7.8: the value is a pure function of the key, so dropping the
    cache at any moment changes nothing observable."""
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    before = observable(sampler)
    sampler._cache_key = None
    sampler._cache_order = None
    assert observable(sampler) == before
    # ...and a *stale* cache is not readable, because the key is compared.
    sampler.epoch_permutation()
    sampler._cache_key = (7, 0, 8)
    sampler._cache_order = tuple(range(8))
    sampler._cache_key = None
    assert observable(sampler) == before


@pytest.mark.parametrize("field,value", [
    ("seed", 12345), ("epoch", 9), ("shuffle", False), ("batch_size", 4),
])
def test_a_state_load_invalidates_the_cache(field, value):
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    stale = sampler.epoch_permutation()
    assert sampler._cache_key is not None
    state = sampler.state_dict()
    state[field] = value
    sampler.load_state_dict(state)
    assert sampler._cache_key is None
    assert sampler._cache_order is None
    fresh = sampler.epoch_permutation()
    if field == "batch_size":
        assert fresh == stale        # order does not depend on batch size
    else:
        assert fresh != stale
    assert fresh == perm.sample_order(sampler.seed, sampler.epoch, 8,
                                      sampler.shuffle)


def test_a_rejected_state_load_leaves_the_cached_behavior_unchanged():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    sampler.epoch_permutation()
    before = observable(sampler)
    cached = sampler._cache_order
    state = sampler.state_dict()
    state["seed"] = -1
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)
    assert sampler._cache_order is cached
    assert observable(sampler) == before


def test_the_cache_appears_in_no_public_surface():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    order = sampler.epoch_permutation()
    text = repr(sampler)
    assert "cache" not in text.lower()
    assert str(order) not in text
    state = json.dumps(sampler.state_dict())
    assert "cache" not in state
    assert str(list(order)) not in state
    # There is no public cache API of any kind.
    for name in ("cache", "clear_cache", "cache_info", "cached_permutation",
                 "cache_stats", "invalidate_cache"):
        assert not hasattr(sampler, name), name
    public = [name for name in dir(sampler) if not name.startswith("_")]
    assert not any("cache" in name for name in public), public


def test_the_returned_order_cannot_be_used_to_mutate_the_sampler():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    order = sampler.epoch_permutation()
    with pytest.raises(TypeError):
        order[0] = 99
    del order
    assert sampler.epoch_permutation() == PERMUTATIONS[(8, 7, 0)]


# ===========================================================================
# 8. State schema (§11.2, §11.4)
# ===========================================================================

def test_the_state_carries_exactly_the_contracted_keys():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=20240612,
                           drop_last=True)
    state = sampler.state_dict()
    assert set(state) == {"format", "format_version", "dataset", "seed",
                          "shuffle", "batch_size", "drop_last", "epoch",
                          "cursor"}
    assert state["format"] == "tensorforge.native_sampler"
    assert state["format_version"] == 1
    assert set(state["dataset"]) == {"samples", "feature_shape",
                                     "feature_dtype", "fingerprint"}
    assert sampler_module._FORMAT == "tensorforge.native_sampler"
    assert sampler_module._FORMAT_VERSION == 1
    assert sampler_module._SUPPORTED_FORMAT_VERSIONS == (1,)


def test_every_state_field_has_the_contracted_type():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=20240612)
    state = sampler.state_dict()
    assert type(state) is dict
    assert type(state["format"]) is str
    assert type(state["format_version"]) is int
    assert type(state["dataset"]) is dict
    assert type(state["seed"]) is int
    assert type(state["shuffle"]) is bool
    assert type(state["batch_size"]) is int
    assert type(state["drop_last"]) is bool
    assert type(state["epoch"]) is int
    assert type(state["cursor"]) is int
    identity = state["dataset"]
    assert type(identity["samples"]) is int
    # Emitted as a **list**, matching what a JSON round trip returns, so a
    # saved-and-reloaded state compares equal without normalization.
    assert type(identity["feature_shape"]) is list
    assert all(type(dim) is int for dim in identity["feature_shape"])
    assert type(identity["feature_dtype"]) is str
    assert type(identity["fingerprint"]) is str
    assert len(identity["fingerprint"]) == 64
    assert identity["fingerprint"] == identity["fingerprint"].lower()
    assert set(identity["fingerprint"]) <= set("0123456789abcdef")


def test_the_state_is_a_fresh_structure_sharing_nothing():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    first = sampler.state_dict()
    second = sampler.state_dict()
    assert first == second
    assert first is not second
    assert first["dataset"] is not second["dataset"]
    assert first["dataset"]["feature_shape"] is not \
        second["dataset"]["feature_shape"]
    # Editing what a caller was given reaches nothing.
    first["seed"] = 99
    first["cursor"] = 2
    first["dataset"]["samples"] = 0
    first["dataset"]["feature_shape"].append(7)
    assert sampler.seed == 7
    assert sampler.cursor == 0
    assert sampler.state_dict() == second


def test_the_state_survives_a_json_round_trip_without_normalization():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=UINT64_MAX,
                           drop_last=True)
    state = sampler.state_dict()
    restored = json.loads(json.dumps(state))
    assert restored == state
    assert type(restored["shuffle"]) is bool
    assert type(restored["drop_last"]) is bool
    assert restored["seed"] == UINT64_MAX    # arbitrary precision, exact
    sampler.load_state_dict(restored)
    assert sampler.state_dict() == state


def test_the_state_passes_the_checkpoint_metadata_validator_unchanged():
    """§2.6/§11.1: every field is JSON-native, which is why Phase J needs
    no checkpoint schema change, no root field, and no version 4."""
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=20240612)
    state = sampler.state_dict()
    payload = {"training": {"next_step": 42, "data_loader": state}}
    validated = native_checkpoint_module._validated_metadata(
        payload, "metadata", set())
    assert validated == payload
    assert validated is not payload
    assert validated["training"]["data_loader"] == state


def test_a_scalar_sample_dataset_emits_an_empty_feature_shape():
    features = np.arange(6, dtype=np.float64)
    targets = np.arange(6, dtype=np.int64) % 2
    dataset = NativeTensorDataset(features, targets)
    assert dataset.feature_shape == ()
    sampler = NativeBatchSampler(dataset, batch_size=2)
    identity = sampler.state_dict()["dataset"]
    assert identity["feature_shape"] == []
    assert type(identity["feature_shape"]) is list
    sampler.load_state_dict(sampler.state_dict())
    assert sampler.cursor == 0


def test_the_state_carries_no_payload_permutation_or_process_local_value():
    """§11.4: no dataset content, no NumPy object, no bytes, no id, no
    address, no callable — and **no permutation**, which is derivable."""
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    state = sampler.state_dict()
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

    walk(state)
    text = json.dumps(state)
    assert str(list(order)) not in text
    assert str(order) not in text
    for forbidden in ("permutation", "cache", "0x", "object at",
                      "ndarray", "NativeTensor"):
        assert forbidden not in text, forbidden
    assert str(id(sampler)) not in text
    assert str(id(sampler.dataset)) not in text


def test_the_state_describes_the_exact_next_batch():
    for cursor in range(3):
        sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
        state = sampler.state_dict()
        state["cursor"] = cursor
        sampler.load_state_dict(state)
        fresh = NativeBatchSampler(make_dataset(8), batch_size=1)
        fresh.load_state_dict(sampler.state_dict())
        assert fresh.next_batch_indices() == sampler.next_batch_indices()
        assert fresh.next_batch_indices() == sampler.plan()[cursor]


def test_state_dict_works_with_a_closed_dataset(live_storages):
    baseline = settled(live_storages)
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True, seed=7)
    before = sampler.state_dict()
    dataset.close()
    assert sampler.state_dict() == before
    sampler.load_state_dict(before)
    assert sampler.state_dict() == before
    assert settled(live_storages) == baseline


def test_state_dict_is_pure():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    before = observable(sampler)
    for _ in range(4):
        sampler.state_dict()
    assert observable(sampler) == before
    assert (sampler.epoch, sampler.cursor) == (0, 0)


# ===========================================================================
# 9. State loading (§12.4) — transactional, validated, and never coercive
# ===========================================================================

def test_a_state_round_trip_restores_everything_and_returns_none():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=20240612,
                           drop_last=False)
    state = sampler.state_dict()
    state["epoch"], state["cursor"] = 4, 2
    assert sampler.load_state_dict(state) is None
    assert sampler.epoch == 4
    assert sampler.cursor == 2
    assert sampler.remaining == 1
    assert sampler.state_dict() == state


def test_configuration_is_adopted_from_the_state():
    """§12.4: structural facts about live objects are validated and never
    adopted; configuration the state carries **is** adopted — the
    NativeAdam rule applied unchanged. A restored sampler may legitimately
    report a different ``batch_size`` than its constructor was given."""
    source = NativeBatchSampler(make_dataset(12), batch_size=5, shuffle=True,
                                seed=0xFEDCBA9876543210, drop_last=True)
    state = source.state_dict()
    state["epoch"], state["cursor"] = 3, 1
    target = NativeBatchSampler(make_dataset(12), batch_size=1, shuffle=False,
                                seed=0, drop_last=False)
    target.load_state_dict(state)
    assert target.batch_size == 5
    assert target.shuffle is True
    assert target.seed == 0xFEDCBA9876543210
    assert target.drop_last is True
    assert target.epoch == 3
    assert target.cursor == 1
    assert target.state_dict() == state


def test_object_and_dataset_identity_are_preserved_absolutely():
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3)
    other = make_dataset(8)
    identity_before = id(sampler)
    state = sampler.state_dict()
    # A state produced from an *equal but distinct* dataset still loads,
    # because the four identity fields agree — and the sampler keeps its
    # own object rather than adopting the other one.
    twin = NativeBatchSampler(other, batch_size=2, shuffle=True, seed=5)
    sampler.load_state_dict(twin.state_dict())
    assert sampler.dataset is dataset
    assert sampler.dataset is not other
    assert id(sampler) == identity_before
    assert sampler.seed == 5


@pytest.mark.parametrize("bad", [
    None, [], (), "state", 7, object(),
])
def test_the_state_container_must_be_a_dict(bad):
    sampler = make_sampler(8)
    with pytest.raises(TypeError):
        sampler.load_state_dict(bad)


def test_a_mapping_that_is_not_a_dict_is_rejected():
    """Exact-type discipline: a ``dict`` subclass or a ``Mapping`` is not
    accepted by coercion, matching §11.5's no-normalization rule."""
    import collections

    sampler = make_sampler(8)
    state = sampler.state_dict()
    with pytest.raises(TypeError):
        sampler.load_state_dict(collections.OrderedDict(state))


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda s: s.pop("cursor"), id="missing-cursor"),
    pytest.param(lambda s: s.pop("dataset"), id="missing-dataset"),
    pytest.param(lambda s: s.pop("format"), id="missing-format"),
    pytest.param(lambda s: s.update(extra=1), id="unexpected-key"),
    pytest.param(lambda s: s.update(strict=False), id="no-strict-flag"),
    pytest.param(lambda s: s.update(permutation=[0, 1]), id="no-permutation"),
])
def test_the_root_key_set_is_exact_in_both_directions(mutate):
    sampler = make_sampler(8)
    state = sampler.state_dict()
    mutate(state)
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda s: s["dataset"].pop("fingerprint"), id="missing"),
    pytest.param(lambda s: s["dataset"].update(extra=1), id="unexpected"),
])
def test_the_dataset_key_set_is_exact_in_both_directions(mutate):
    sampler = make_sampler(8)
    state = sampler.state_dict()
    mutate(state)
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)


@pytest.mark.parametrize("value,expected", [
    (7, TypeError), (None, TypeError), (b"x", TypeError),
    ("tensorforge.native_data_loader", ValueError),
    ("tensorforge.native_checkpoint", ValueError),
    ("", ValueError), ("TENSORFORGE.NATIVE_SAMPLER", ValueError),
])
def test_the_format_tag_is_validated_by_type_then_by_value(value, expected):
    sampler = make_sampler(8)
    state = sampler.state_dict()
    state["format"] = value
    with pytest.raises(expected):
        sampler.load_state_dict(state)


@pytest.mark.parametrize("value,expected", [
    (True, TypeError), (1.0, TypeError), ("1", TypeError),
    (np.int64(1), TypeError),
    (0, ValueError), (2, ValueError), (-1, ValueError), (4, ValueError),
])
def test_the_format_version_is_validated_by_type_then_by_value(value,
                                                               expected):
    sampler = make_sampler(8)
    state = sampler.state_dict()
    state["format_version"] = value
    with pytest.raises(expected):
        sampler.load_state_dict(state)


@pytest.mark.parametrize("value", [None, [], "x", 7, ((1, 2),)])
def test_the_dataset_block_must_be_a_dict(value):
    sampler = make_sampler(8)
    state = sampler.state_dict()
    state["dataset"] = value
    with pytest.raises(TypeError):
        sampler.load_state_dict(state)


@pytest.mark.parametrize("field,value,expected", [
    ("samples", True, TypeError), ("samples", 8.0, TypeError),
    ("samples", np.int64(8), TypeError), ("samples", "8", TypeError),
    ("samples", 0, ValueError), ("samples", -1, ValueError),
    ("feature_shape", 2, TypeError), ("feature_shape", None, TypeError),
    ("feature_shape", "2", TypeError),
    ("feature_shape", [True], TypeError),
    ("feature_shape", [np.int64(2)], TypeError),
    ("feature_shape", [2.0], TypeError),
    ("feature_shape", [0], ValueError), ("feature_shape", [-2], ValueError),
    ("feature_dtype", 7, TypeError), ("feature_dtype", None, TypeError),
    ("feature_dtype", "float16", ValueError),
    ("feature_dtype", "Float64", ValueError),
    ("feature_dtype", "f8", ValueError),
    ("fingerprint", 7, TypeError), ("fingerprint", None, TypeError),
    ("fingerprint", b"a" * 64, TypeError),
    ("fingerprint", "a" * 63, ValueError),
    ("fingerprint", "a" * 65, ValueError),
    ("fingerprint", "A" * 64, ValueError),
    ("fingerprint", "g" * 64, ValueError),
    ("fingerprint", "0x" + "a" * 62, ValueError),
])
def test_each_dataset_identity_field_is_validated(field, value, expected):
    sampler = make_sampler(8)
    state = sampler.state_dict()
    state["dataset"][field] = value
    with pytest.raises(expected):
        sampler.load_state_dict(state)


def test_a_tuple_feature_shape_is_accepted_on_load():
    """§11.2: ``load_state_dict`` accepts a tuple there as well, following
    the v3.13 precedent that a caller may have rebuilt the container —
    while ``state_dict()`` always emits a list."""
    dataset = make_dataset(8, width=3)
    sampler = NativeBatchSampler(dataset, batch_size=3)
    state = sampler.state_dict()
    state["dataset"]["feature_shape"] = tuple(
        state["dataset"]["feature_shape"])
    sampler.load_state_dict(state)
    assert type(sampler.state_dict()["dataset"]["feature_shape"]) is list


def test_each_dataset_mismatch_is_rejected_independently():
    """§12.4 step 8: the four identity fields, each on its own, so a
    mismatch names one understandable difference."""
    sampler = NativeBatchSampler(make_dataset(8, width=2), batch_size=3)
    base = sampler.state_dict()

    # samples
    state = sampler.state_dict()
    state["dataset"]["samples"] = 9
    with pytest.raises(ValueError) as samples_error:
        sampler.load_state_dict(state)
    assert "9" in str(samples_error.value)

    # feature shape
    state = sampler.state_dict()
    state["dataset"]["feature_shape"] = [3]
    with pytest.raises(ValueError) as shape_error:
        sampler.load_state_dict(state)
    assert "shape" in str(shape_error.value)

    # feature dtype
    state = sampler.state_dict()
    state["dataset"]["feature_dtype"] = "float32"
    with pytest.raises(ValueError) as dtype_error:
        sampler.load_state_dict(state)
    assert "dtype" in str(dtype_error.value)

    # fingerprint — the one that catches "same shape, same dtype,
    # different data", which structural fields cannot.
    other = make_dataset(8, width=2)
    other_state = NativeBatchSampler(other, batch_size=3).state_dict()
    assert other_state["dataset"] == base["dataset"]      # equal content
    different = NativeTensorDataset(
        np.arange(16, dtype=np.float64).reshape(8, 2) + 1.0,
        np.arange(8, dtype=np.int64) % 3)
    state = NativeBatchSampler(different, batch_size=3).state_dict()
    assert state["dataset"]["samples"] == base["dataset"]["samples"]
    assert state["dataset"]["feature_shape"] == base["dataset"]["feature_shape"]
    assert state["dataset"]["feature_dtype"] == base["dataset"]["feature_dtype"]
    assert state["dataset"]["fingerprint"] != base["dataset"]["fingerprint"]
    with pytest.raises(ValueError) as fingerprint_error:
        sampler.load_state_dict(state)
    assert "fingerprint" in str(fingerprint_error.value)
    assert sampler.state_dict() == base


def test_a_structural_mismatch_is_reported_before_the_fingerprint():
    """§12.4: structural first, digest last — "the fingerprints differ" is
    only useful once the shapes agree."""
    sampler = NativeBatchSampler(make_dataset(8, width=2), batch_size=3)
    state = sampler.state_dict()
    state["dataset"]["samples"] = 5
    state["dataset"]["feature_shape"] = [9]
    state["dataset"]["feature_dtype"] = "float32"
    state["dataset"]["fingerprint"] = "b" * 64
    with pytest.raises(ValueError) as excinfo:
        sampler.load_state_dict(state)
    assert "samples" in str(excinfo.value)
    assert "fingerprint" not in str(excinfo.value)
    # ...and with samples agreeing, the shape is next.
    state["dataset"]["samples"] = 8
    with pytest.raises(ValueError) as shape_first:
        sampler.load_state_dict(state)
    assert "shape" in str(shape_first.value)
    # ...then the dtype, and only then the digest.
    state["dataset"]["feature_shape"] = [2]
    with pytest.raises(ValueError) as dtype_first:
        sampler.load_state_dict(state)
    assert "dtype" in str(dtype_first.value)
    state["dataset"]["feature_dtype"] = "float64"
    with pytest.raises(ValueError) as digest_last:
        sampler.load_state_dict(state)
    assert "fingerprint" in str(digest_last.value)


@pytest.mark.parametrize("field,value,expected", [
    ("seed", True, TypeError), ("seed", 1.0, TypeError),
    ("seed", "7", TypeError), ("seed", np.uint64(7), TypeError),
    ("seed", -1, ValueError), ("seed", 2**64, ValueError),
    ("shuffle", 1, TypeError), ("shuffle", 0, TypeError),
    ("shuffle", None, TypeError), ("shuffle", np.bool_(True), TypeError),
    ("shuffle", "true", TypeError),
    ("batch_size", True, TypeError), ("batch_size", 3.0, TypeError),
    ("batch_size", np.int64(3), TypeError),
    ("batch_size", 0, ValueError), ("batch_size", -3, ValueError),
    ("drop_last", 1, TypeError), ("drop_last", 0, TypeError),
    ("drop_last", None, TypeError), ("drop_last", np.bool_(False), TypeError),
    ("epoch", True, TypeError), ("epoch", 1.0, TypeError),
    ("epoch", np.int64(1), TypeError),
    ("epoch", -1, ValueError), ("epoch", 2**64, ValueError),
    ("cursor", True, TypeError), ("cursor", 0.0, TypeError),
    ("cursor", np.int64(0), TypeError), ("cursor", "0", TypeError),
    ("cursor", -1, ValueError),
])
def test_each_configuration_field_is_validated(field, value, expected):
    sampler = make_sampler(8, batch_size=3)
    state = sampler.state_dict()
    state[field] = value
    with pytest.raises(expected):
        sampler.load_state_dict(state)


@pytest.mark.parametrize("good", [0, 1, UINT64_MAX])
def test_the_state_seed_and_epoch_boundaries_are_accepted(good):
    sampler = make_sampler(8, batch_size=3, shuffle=True)
    state = sampler.state_dict()
    state["seed"] = good
    state["epoch"] = good
    sampler.load_state_dict(state)
    assert sampler.seed == good
    assert sampler.epoch == good
    assert len(sampler.epoch_permutation()) == 8


def test_a_zero_batch_state_is_rejected_by_the_same_joint_rule():
    """§7.5/§12.4 step 11: a state whose ``batch_size`` and ``drop_last``
    would produce zero batches is refused before any mutation."""
    sampler = make_sampler(8, batch_size=3)
    state = sampler.state_dict()
    state["batch_size"] = 9
    state["drop_last"] = True
    state["cursor"] = 0
    with pytest.raises(ValueError) as excinfo:
        sampler.load_state_dict(state)
    assert "9" in str(excinfo.value)
    assert sampler.batch_size == 3
    # ...and the same batch size with the tail kept is fine.
    state["drop_last"] = False
    sampler.load_state_dict(state)
    assert sampler.batch_size == 9
    assert sampler.batches_per_epoch == 1


@pytest.mark.parametrize("cursor,batch_size,drop_last,ok", [
    (0, 3, False, True), (2, 3, False, True), (3, 3, False, False),
    (4, 3, False, False), (-1, 3, False, False),
    (0, 3, True, True), (1, 3, True, True), (2, 3, True, False),
    (0, 8, False, True), (1, 8, False, False),
])
def test_the_cursor_is_checked_against_the_candidate_batch_count(
        cursor, batch_size, drop_last, ok):
    """§12.4 step 12: last, because it is the only rule depending on
    several other fields being valid first — and it is never clamped."""
    sampler = make_sampler(8, batch_size=3)
    state = sampler.state_dict()
    state["batch_size"] = batch_size
    state["drop_last"] = drop_last
    state["cursor"] = cursor
    if ok:
        sampler.load_state_dict(state)
        assert sampler.cursor == cursor
        assert sampler.remaining >= 1
    else:
        with pytest.raises(ValueError):
            sampler.load_state_dict(state)
        assert sampler.cursor == 0


@pytest.mark.parametrize("mutate,expected,reason", [
    pytest.param(lambda s: [s.update(extra=1), s.__setitem__("format", 7)],
                 ValueError, "key set before format type",
                 id="keys-before-format"),
    pytest.param(lambda s: [s.__setitem__("format", "nope"),
                            s.__setitem__("format_version", "1")],
                 ValueError, "format value before version type",
                 id="format-before-version"),
    pytest.param(lambda s: [s.__setitem__("format_version", 9),
                            s.__setitem__("dataset", None)],
                 ValueError, "version value before dataset type",
                 id="version-before-dataset"),
    pytest.param(lambda s: [s["dataset"].pop("samples"),
                            s.__setitem__("seed", -1)],
                 ValueError, "dataset keys before configuration",
                 id="dataset-keys-before-config"),
    pytest.param(lambda s: [s["dataset"].__setitem__("samples", "8"),
                            s.__setitem__("seed", -1)],
                 TypeError, "dataset field type before configuration range",
                 id="dataset-type-before-config-range"),
    pytest.param(lambda s: [s["dataset"].__setitem__("samples", 99),
                            s.__setitem__("seed", "x")],
                 ValueError, "dataset compatibility before configuration type",
                 id="compat-before-config-type"),
    pytest.param(lambda s: [s.__setitem__("seed", "x"),
                            s.__setitem__("cursor", 99)],
                 TypeError, "configuration type before cursor range",
                 id="config-type-before-cursor"),
    pytest.param(lambda s: [s.__setitem__("seed", -1),
                            s.__setitem__("cursor", 99)],
                 ValueError, "seed range before cursor range",
                 id="seed-range-before-cursor"),
    pytest.param(lambda s: [s.__setitem__("epoch", 1.0),
                            s.__setitem__("cursor", 99)],
                 TypeError, "epoch type before cursor range",
                 id="epoch-type-before-cursor"),
])
def test_the_load_precedence_is_exactly_the_contracted_order(mutate, expected,
                                                             reason):
    """§12.4, probed by exception *type* so no message is a contract."""
    sampler = make_sampler(8, batch_size=3)
    state = sampler.state_dict()
    mutate(state)
    with pytest.raises(expected):
        sampler.load_state_dict(state)


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda s: s.__setitem__("seed", -1), id="seed-range"),
    pytest.param(lambda s: s.__setitem__("shuffle", 1), id="shuffle-type"),
    pytest.param(lambda s: s.__setitem__("cursor", 99), id="cursor-range"),
    pytest.param(lambda s: s.__setitem__("format", "nope"), id="format"),
    pytest.param(lambda s: s.__setitem__("format_version", 2), id="version"),
    pytest.param(lambda s: s.pop("epoch"), id="missing-key"),
    pytest.param(lambda s: s.update(extra=1), id="extra-key"),
    pytest.param(lambda s: s["dataset"].__setitem__("samples", 99),
                 id="dataset-samples"),
    pytest.param(lambda s: s["dataset"].__setitem__("fingerprint", "c" * 64),
                 id="dataset-fingerprint"),
    pytest.param(lambda s: s["dataset"].__setitem__("feature_dtype",
                                                    "float32"),
                 id="dataset-dtype"),
    pytest.param(lambda s: [s.__setitem__("batch_size", 99),
                            s.__setitem__("drop_last", True)],
                 id="zero-batch"),
])
def test_a_rejected_load_leaves_the_whole_observable_state_unchanged(mutate):
    """§12.7/§17.5: nothing is mutated. Every field, the position, the
    configuration, the planning results, and the cache behavior are
    exactly as they were — the complete before/after fingerprint."""
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=20240612)
    # Start from a non-origin position, so a partial commit would show.
    moved = sampler.state_dict()
    moved["epoch"], moved["cursor"] = 5, 2
    sampler.load_state_dict(moved)
    sampler.epoch_permutation()          # populate the cache
    before = observable(sampler)

    state = sampler.state_dict()
    mutate(state)
    with pytest.raises((TypeError, ValueError)):
        sampler.load_state_dict(state)
    assert observable(sampler) == before
    assert sampler.state_dict() == moved


def test_nothing_is_normalized_coerced_or_repaired():
    """§11.5: ``1`` is not ``True``, ``True`` is not ``1``, ``1.0`` is not
    ``1``, ``"1"`` is not ``1``, a NumPy ``int64`` is not an ``int``, a
    missing key is not a default, an unknown key is not ignored, and a
    cursor past the end is not clamped."""
    sampler = make_sampler(8, batch_size=3)
    cases = [
        ("shuffle", 1), ("shuffle", 0), ("drop_last", 1),
        ("batch_size", True), ("seed", True), ("epoch", True),
        ("cursor", True), ("batch_size", 3.0), ("seed", "0"),
        ("format_version", 1.0), ("cursor", np.int64(0)),
        ("batch_size", np.int64(3)),
    ]
    for field, value in cases:
        state = sampler.state_dict()
        state[field] = value
        with pytest.raises(TypeError):
            sampler.load_state_dict(state)
    # A cursor past the end is rejected, never clamped to a valid one.
    state = sampler.state_dict()
    state["cursor"] = 3
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)
    assert sampler.cursor == 0


def test_the_commit_is_six_non_failing_assignments():
    """§12.4 Phase 2 / §17.5: the commit consists only of assignments of
    already-validated ints and bools, which is what makes the transaction
    exact without a rollback path — there is no failure to roll back from.
    Asserted structurally, as the design says J4 will."""
    source = inspect.getsource(NativeBatchSampler._assign_state)
    body = ast.parse(source.lstrip()).body[0].body
    statements = [node for node in body if not isinstance(node, ast.Expr)]
    assert all(isinstance(node, ast.Assign) for node in statements), statements
    # Six state writes plus the two cache-invalidation writes, and no call,
    # comparison, loop, branch, or raise anywhere in the seam.
    targets = [node.targets[0].attr for node in statements]
    assert targets == ["_seed", "_shuffle", "_batch_size", "_drop_last",
                       "_epoch", "_cursor", "_cache_key", "_cache_order"]
    for node in ast.walk(ast.parse(source.lstrip())):
        assert not isinstance(node, (ast.Call, ast.Raise, ast.If, ast.For,
                                     ast.While, ast.Try, ast.Compare))


def test_the_private_validate_and_assign_split_mutates_nothing_on_its_own():
    """The private helpers J3 needs: validation that touches nothing, and
    a snapshot that reproduces the current position exactly."""
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=7)
    before = observable(sampler)
    state = sampler.state_dict()
    state["epoch"], state["cursor"], state["seed"] = 9, 1, 4242
    values = sampler._validate_state(state)
    assert observable(sampler) == before          # validation alone moved none
    assert values == (4242, True, 3, False, 9, 1)
    snapshot = sampler._snapshot_state()
    assert snapshot == (7, True, 3, False, 0, 0)
    sampler._assign_state(*values)
    assert (sampler.seed, sampler.epoch, sampler.cursor) == (4242, 9, 1)
    # ...and the snapshot restores the pre-load position exactly, which is
    # the property J3's rollback will rest on.
    sampler._assign_state(*snapshot)
    assert observable(sampler) == before


def test_the_canonical_epoch_boundary_is_encoded_but_not_public():
    """§7.4: the transition J3 applies after a successful delivery. It is
    computed here without mutating anything, and J2 exposes no way to
    apply it — nothing in this milestone delivers a batch."""
    sampler = make_sampler(8, batch_size=3)      # 3 batches per epoch
    assert sampler._next_position(0, 0) == (0, 1)
    assert sampler._next_position(0, 1) == (0, 2)
    # The boundary is canonicalized immediately, so there is exactly one
    # representation of "end of epoch 0" and it is "start of epoch 1".
    assert sampler._next_position(0, 2) == (1, 0)
    assert sampler._next_position(41, 2) == (42, 0)
    before = observable(sampler)
    for epoch in range(3):
        for cursor in range(3):
            sampler._next_position(epoch, cursor)
    assert observable(sampler) == before
    # No public surface applies it.
    assert not hasattr(sampler, "next_position")
    assert not hasattr(sampler, "advance")


def test_the_epoch_refuses_to_advance_past_the_uint64_domain():
    """§7.4: unreachable in practice, and specified so that it is not
    undefined. It raises and moves nothing, exactly as NativeGenerator
    refuses at an exhausted counter."""
    sampler = make_sampler(8, batch_size=3)
    state = sampler.state_dict()
    state["epoch"], state["cursor"] = UINT64_MAX, 2
    sampler.load_state_dict(state)
    before = observable(sampler)
    with pytest.raises(RuntimeError):
        sampler._next_position(sampler.epoch, sampler.cursor)
    assert observable(sampler) == before
    assert sampler.epoch == UINT64_MAX
    # One batch earlier in the same epoch is still fine.
    assert sampler._next_position(UINT64_MAX, 0) == (UINT64_MAX, 1)


def test_load_state_dict_works_with_a_closed_dataset():
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True, seed=7)
    state = sampler.state_dict()
    state["epoch"], state["cursor"] = 2, 1
    dataset.close()
    sampler.load_state_dict(state)
    assert (sampler.epoch, sampler.cursor) == (2, 1)
    assert sampler.next_batch_indices() == sampler.plan()[1]


def test_a_state_from_a_json_file_round_trip_restores_the_same_batches():
    """The shape a caller's checkpoint-metadata workflow will take at J5,
    proved here without any checkpoint coupling."""
    original = NativeBatchSampler(make_dataset(13), batch_size=4,
                                  shuffle=True, seed=0xFEDCBA9876543210)
    state = original.state_dict()
    state["epoch"], state["cursor"] = 3, 2
    original.load_state_dict(state)
    carried = json.loads(json.dumps(original.state_dict()))

    fresh = NativeBatchSampler(make_dataset(13), batch_size=1, shuffle=False,
                               seed=1)
    fresh.load_state_dict(carried)
    assert fresh.state_dict() == original.state_dict()
    assert fresh.next_batch_indices() == original.next_batch_indices()
    for epoch in range(3, 9):
        assert fresh.plan(epoch) == original.plan(epoch)


# ===========================================================================
# 10. Representation
# ===========================================================================

def test_the_repr_shows_configuration_and_position_and_nothing_else():
    sampler = make_sampler(8, batch_size=3, shuffle=True, seed=20240612,
                           drop_last=True)
    state = sampler.state_dict()
    state["epoch"], state["cursor"] = 4, 1
    sampler.load_state_dict(state)
    text = repr(sampler)
    assert text.startswith("NativeBatchSampler(")
    for fragment in ("samples=8", "batch_size=3", "shuffle=True",
                     "seed=20240612", "drop_last=True", "epoch=4",
                     "cursor=1", "batches_per_epoch=2", "remaining=1"):
        assert fragment in text, fragment


def test_the_repr_carries_no_data_fingerprint_permutation_or_address():
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True, seed=7)
    order = sampler.epoch_permutation()
    text = repr(sampler)
    assert dataset.fingerprint not in text
    assert dataset.fingerprint[:12] not in text
    assert str(order) not in text
    assert str(list(order)) not in text
    assert "0x" not in text
    assert str(id(sampler)) not in text
    assert str(id(dataset)) not in text
    assert "object at" not in text
    assert "ndarray" not in text
    assert "array(" not in text


def test_the_repr_works_when_the_dataset_is_closed():
    dataset = make_dataset(8)
    sampler = NativeBatchSampler(dataset, batch_size=3)
    before = repr(sampler)
    dataset.close()
    assert repr(sampler) == before


# ===========================================================================
# 11. J2 non-goals — what this milestone must not have added
# ===========================================================================

def test_no_loader_iterator_or_delivery_runtime_exists_anywhere():
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    assert not (package / "native_data_loader.py").exists()
    definitions = set()
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in ("NativeDataLoader", "_NativeBatchIterator"):
            if re.search(rf"^\s*class {name}\b", text, re.M):
                definitions.add(name)
        assert not re.search(r"^\s*def _deliver_batch\b", text, re.M), path.name
    assert definitions == set()
    # Negative control: the scanner really does find a class when one is
    # present, so "none found" is evidence rather than a dead regex.
    planted = "class _NativeBatchIterator:\n    pass\n"
    assert re.search(r"^\s*class _NativeBatchIterator\b", planted, re.M)
    assert re.search(r"^\s*def _deliver_batch\b", "def _deliver_batch(r):\n",
                     re.M)


def test_the_sampler_module_holds_no_generator_thread_lock_or_worker():
    """§8.3, §16.1: no NativeGenerator is accepted, held, or created; and
    Phase J adds no worker, thread, pool, prefetch, queue, future, async
    iteration, or lock."""
    tree = ast.parse((REPO_ROOT / SAMPLER_SOURCE).read_text(encoding="utf-8"))
    names = code_identifiers(SAMPLER_SOURCE)
    for forbidden in ("threading", "Lock", "RLock", "queue", "Queue",
                      "concurrent", "asyncio", "multiprocessing",
                      "state_transaction", "_reserve_call", "_commit_call",
                      "_abandon_call", "NativeGenerator", "random",
                      "secrets", "time", "environ", "ctypes", "Thread",
                      "Pool", "Future", "prefetch"):
        assert forbidden not in names, forbidden
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.AsyncFunctionDef, ast.Await,
                                     ast.AsyncFor, ast.AsyncWith)), node
    # The one thing it *does* take from the generator module is the shared
    # uint64 validator — a rule, not a generator (§8.3).
    imported = set()
    for node in ast.walk(tree):
        assert not isinstance(node, ast.Import), ast.dump(node)
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add((node.module, alias.name))
    assert imported == {
        (None, "_native_permutation"),
        ("native_generator", "UINT64_MAX"),
        ("native_generator", "_validate_uint64"),
        ("native_dataset", "NativeTensorDataset"),
    }
    # No NativeGenerator instance is reachable from a constructed sampler.
    sampler = make_sampler(8)
    for slot in NativeBatchSampler.__slots__:
        value = getattr(sampler, slot)
        assert not isinstance(value,
                              native_generator_module.NativeGenerator), slot


def test_the_sampler_constructs_no_native_tensor_and_materializes_nothing():
    """J2's invariant: no ``NativeTensor`` allocation anywhere in the
    milestone, and no batch materialization."""
    names = code_identifiers(SAMPLER_SOURCE) | code_identifiers(
        PERMUTATION_SOURCE)
    for forbidden in ("NativeTensor", "NativeTensorCore", "from_array",
                      "feature_batch", "target_batch", "to_numpy", "numpy",
                      "np", "zeros", "storage", "NativeStorage", "close"):
        assert forbidden not in names, forbidden
    sampler = make_sampler(8)
    for name in ("feature_batch", "target_batch", "batch", "materialize",
                 "collate", "transform", "prefetch", "workers",
                 "num_workers", "pin_memory"):
        assert not hasattr(sampler, name), name


def test_this_milestone_touched_no_cpp_cmake_or_abi_surface():
    """§22.2: no kernel, header, CTest, translation unit, build option, or
    C ABI symbol. Asserted against the live inventories."""
    exports = set()
    for path in (REPO_ROOT / "cpp" / "src").glob("*.cpp"):
        text = path.read_text(encoding="utf-8")
        exports.update(re.findall(r"TF_EXPORT[^(]*?\b(tf_[a-z0-9_]+)\s*\(",
                                  text))
    assert len(exports) == 54, sorted(exports)
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert cmake.count("add_test") == 24
    assert len(list((REPO_ROOT / "examples").glob("*.py"))) == 15
    assert len(list((REPO_ROOT / "benchmarks").glob("*.py"))) == 8
    for module in (SAMPLER_SOURCE, PERMUTATION_SOURCE):
        names = code_identifiers(module)
        assert not any(name.startswith("tf_") for name in names), module
        assert "argtypes" not in names, module
        assert "restype" not in names, module


def test_no_capability_registry_checkpoint_or_optimizer_version_moved():
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    assert cpp.backend_info()["dtype"] == "float64"
    assert cpp.backend_info()["stable_framework_integration"] is False
    assert native_checkpoint_module._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint_module._FORMAT_VERSION == 3
    assert native_checkpoint_module._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    from tensorforge.experimental import native_optimizer_state
    assert native_optimizer_state.FORMAT_VERSION == 1


def test_the_checkpoint_modules_know_nothing_about_the_sampler():
    """§13.6: no import in either direction between the checkpoint module
    and the data-pipeline modules, and no automatic discovery."""
    checkpoint = code_identifiers(
        "src/tensorforge/experimental/native_checkpoint.py")
    for forbidden in ("native_sampler", "NativeBatchSampler",
                      "_native_permutation", "native_dataset",
                      "NativeTensorDataset", "sampler", "loader",
                      "epoch", "cursor", "batch_size", "drop_last"):
        assert forbidden not in checkpoint, forbidden
    sampler = code_identifiers(SAMPLER_SOURCE)
    for forbidden in ("native_checkpoint", "save_native_checkpoint",
                      "load_native_checkpoint", "_validated_metadata"):
        assert forbidden not in sampler, forbidden


def test_the_sampler_owns_no_dtype_or_device_authority():
    """§19: the sampler owns no dtype-bearing numeric state, so it takes
    neither a ``dtype`` nor a ``device`` argument and reports neither."""
    parameters = inspect.signature(NativeBatchSampler).parameters
    assert "dtype" not in parameters
    assert "device" not in parameters
    sampler = make_sampler(8)
    assert not hasattr(sampler, "dtype")
    assert not hasattr(sampler, "device")
    # The dataset remains the single authority, and the state reports it
    # rather than restating it.
    assert sampler.state_dict()["dataset"]["feature_dtype"] == \
        sampler.dataset.dtype
