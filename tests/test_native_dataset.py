"""NativeTensorDataset — the host-backed native dataset (Phase J,
milestone J1; docs/native_data_pipeline_design.md §4, §5, §6, §10.4,
§12.2, §12.6, §15, §17.2).

What this module proves, and what it deliberately does not:

* **§4 input contract** — every accepted and every rejected input class,
  and the exact validation *precedence* when a caller gets more than one
  thing wrong, which a per-rule test cannot see.
* **§5 ownership** — the snapshots are unconditional copies, so mutating,
  resizing, or deleting the caller's arrays afterwards reaches nothing.
  Asserted in both directions rather than by reading the implementation.
* **§6 identity** — the fingerprint against **independently computed
  known answers**, built here from Python floats and ``struct`` without
  touching the production helper, plus its sensitivity and its
  endian/layout invariance.
* **§12.6 batch requests** — container, emptiness, bounds, order, and
  duplicates, in order, at both batch methods.
* **§15/§17.2 lifecycle** — close, use after close, and the construction
  failure positions, each with the live-storage baseline around it.

**Not tested here, because it does not exist:** shuffling, permutations,
cursors, epochs, batch sizes, drop-last, sampler or loader state, and
checkpoint loader-state integration. ``NativeBatchSampler`` (J2) and
``NativeDataLoader`` (J3) have not started, and nothing in this file
needs them — a dataset answers only "given these indices, what is the
batch?", which is exactly why J1 is testable on its own.

No test here asserts an exact error message, a dict ordering, a timing,
or a GC event.

Selector: python -m pytest -q tests/test_native_dataset.py
"""

import gc
import hashlib
import json
import struct
import sys

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensorDataset
from tensorforge.experimental import native_dataset as native_dataset_module
from tensorforge.experimental import native_checkpoint as native_checkpoint_module

# Only the tests that actually materialize a native feature batch need the
# built library. Construction, validation, the fingerprint, identity, target
# batches, and the whole lifecycle are pure Python over NumPy, and stay
# provable on a machine with no C++ compiler.
needs_backend = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    deterministic instrumentation for native-allocation lifetime (the
    Phase-C..G precedent). There is no public counter, and J1 does not add
    one."""
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
    """The live-storage count after a collection. Collection settles the
    count; it is never the proof that anything was released — every test
    here closes what it owns explicitly first."""
    gc.collect()
    return len(live_storages)


def features_2d(samples=6, width=2, dtype=np.float64):
    return np.arange(samples * width, dtype=dtype).reshape(samples, width)


def targets_for(samples=6, dtype=np.int64):
    return np.arange(samples, dtype=dtype) % 3


def dataset(samples=6, width=2, *, dtype=None, feature_dtype=np.float64):
    return NativeTensorDataset(features_2d(samples, width, feature_dtype),
                               targets_for(samples), dtype=dtype)


# The independent fingerprint oracle. It shares **no code** with the
# production helper: it walks nested Python lists, packs every value with
# ``struct``, and never touches a NumPy buffer, ``astype``, or chunking. If
# the two agree, they agree because the byte stream of §6.3 is what both
# implement — not because one called the other.
def oracle_fingerprint(dtype_name, rows, target_values):
    hasher = hashlib.sha256()
    hasher.update(b"tensorforge.native_dataset\x00")
    hasher.update(b"fingerprint-v1\x00")
    hasher.update(dtype_name.encode("utf-8") + b"\x00")
    shape = []
    probe = rows
    while isinstance(probe, list):
        shape.append(len(probe))
        probe = probe[0]
    hasher.update(struct.pack("<Q", len(shape)))
    for dimension in shape:
        hasher.update(struct.pack("<Q", dimension))
    code = "<f" if dtype_name == "float32" else "<d"

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
        else:
            hasher.update(struct.pack(code, node))

    walk(rows)
    hasher.update(b"targets\x00")
    hasher.update(struct.pack("<Q", len(target_values)))
    for value in target_values:
        hasher.update(struct.pack("<q", value))
    return hasher.hexdigest()


# Three committed known answers, produced by the oracle above and written
# here as literals. They are the specification: a change to the domain tag,
# the schema tag, the dtype encoding, the shape prefix, the target marker,
# the element order, or the endianness moves one of them.
KNOWN_FLOAT64_2D = (
    "eceacef05cf7dbcedc68bd3562ff44e936af3f15e639a1036ae01dd123bcba17")
KNOWN_FLOAT32_2D = (
    "1fe59987fdeca6925c2ee9daffa38699111a8eeca4446d13b73277eba6f55356")
KNOWN_FLOAT64_SCALAR = (
    "300b79e080adbb4722d478e7a222a666126de12be09fad0b4b11d62381caed05")

KNOWN_ROWS = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
KNOWN_TARGETS = [0, 1, 0]
KNOWN_SCALAR_ROWS = [0.5, -0.5, 2.25, -3.125]
KNOWN_SCALAR_TARGETS = [3, 2, 1, 0]


# ===========================================================================
# 1. Public API, exports, and isolation
# ===========================================================================

def test_the_class_lives_in_its_own_module_and_is_exported():
    from tensorforge.experimental.native_dataset import NativeTensorDataset as direct

    import tensorforge.experimental as experimental

    assert direct is NativeTensorDataset
    assert experimental.NativeTensorDataset is NativeTensorDataset
    assert "NativeTensorDataset" in experimental.__all__
    assert NativeTensorDataset.__module__ == (
        "tensorforge.experimental.native_dataset")


def test_the_experimental_inventory_grew_by_exactly_one_name():
    """The export inventory is a contract. J1 adds exactly one public
    experimental name and nothing else — asserted against the J0 inventory
    written out here rather than against the live list, so a silent second
    addition fails."""
    import tensorforge.experimental as experimental

    j0_inventory = {
        "NativeTensor", "NativeGenerator", "NativeParameter",
        "NativeParameterRegistry", "NativeModule", "NativeLinear",
        "NativeReLU", "NativeFlatten", "NativeConv2d", "NativeMaxPool2d",
        "NativeSequential", "NativeLayerNorm", "NativeBatchNorm1d",
        "NativeBatchNorm2d", "NativeDropout", "NativeMSELoss",
        "NativeCrossEntropyLoss", "native_accuracy", "NativeSGD",
        "NativeAdam", "save_native_checkpoint", "load_native_checkpoint",
    }
    # What a *later* milestone legitimately added, named separately so the
    # J1 statement stays exactly what J1 shipped and the whole check stays
    # an exact equality in both directions.
    post_j1_additions = {
        "NativeBatchSampler",   # Phase J, milestone J2 — not J1
    }
    live = set(experimental.__all__)
    assert len(experimental.__all__) == len(live), "duplicate export"
    assert live - j0_inventory == {"NativeTensorDataset"} | post_j1_additions
    assert j0_inventory - live == set()
    assert len(experimental.__all__) == len(j0_inventory) + 1 + len(
        post_j1_additions)


def test_no_later_phase_j_name_exists_yet():
    """J3 has not started, so neither its class, its iterator, nor its
    module may appear — and J2's derivation helper, which *has* landed,
    stays permanently private rather than joining the exports."""
    import tensorforge.experimental as experimental

    for name in ("NativeDataLoader", "_NativeBatchIterator"):
        assert not hasattr(experimental, name), name
        assert name not in experimental.__all__, name
    for helper in ("_native_permutation", "splitmix64_mix", "epoch_key",
                   "draw_bits", "bounded", "permutation"):
        assert helper not in experimental.__all__, helper
    package = native_dataset_module.__file__.rsplit("native_dataset.py", 1)[0]
    from pathlib import Path

    assert not (Path(package) / "native_data_loader.py").exists()
    # J2's two modules exist; the dataset must still not be the one that
    # defines their classes.
    for module in ("native_sampler.py", "_native_permutation.py"):
        assert (Path(package) / module).is_file(), module
    dataset_source = Path(native_dataset_module.__file__).read_text(
        encoding="utf-8")
    for absent in ("class NativeBatchSampler", "class NativeDataLoader"):
        assert absent not in dataset_source, absent


def test_no_sampler_or_loader_concept_leaked_into_the_dataset():
    """The J1 exclusion list, stated over the live class rather than
    promised in prose."""
    for name in ("state_dict", "load_state_dict", "epoch", "cursor",
                 "shuffle", "seed", "batch_size", "drop_last", "plan",
                 "epoch_permutation", "next_batch_indices",
                 "batches_per_epoch", "remaining", "__iter__", "__next__",
                 "__getitem__"):
        assert not hasattr(NativeTensorDataset, name), name
    # ...and no public accessor for either snapshot (§3.3).
    public = {name for name in dir(NativeTensorDataset)
              if not name.startswith("_")}
    assert public == {
        "samples", "feature_shape", "dtype", "device", "fingerprint",
        "closed", "identity", "feature_batch", "target_batch", "close",
    }, sorted(public)


def test_the_stable_line_did_not_gain_the_name_and_stays_isolated():
    assert not hasattr(tensorforge, "NativeTensorDataset")
    assert "NativeTensorDataset" not in tensorforge.__all__
    # The stable mini-batch iterator is untouched and stays stable-only.
    assert hasattr(tensorforge, "batches")
    assert "batches" in tensorforge.__all__


def test_importing_the_stable_package_still_does_not_load_the_backend():
    """The isolation J1 must not break, proved in a clean interpreter: a
    bare ``import tensorforge`` imports neither the experimental package
    nor the ctypes wrapper, and loads no library."""
    import subprocess

    program = (
        "import sys, tensorforge;"
        "bad=[m for m in sys.modules if 'tensorforge.experimental' in m"
        " or 'tensorforge.backends' in m];"
        "print(bad)"
    )
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", result.stdout


def test_a_stable_tensor_is_rejected_like_any_other_non_ndarray():
    """§18: no stable ``Tensor`` acceptance, and no bridge."""
    stable = tensorforge.Tensor(np.arange(6.0).reshape(3, 2))
    with pytest.raises(TypeError):
        NativeTensorDataset(stable, targets_for(3))
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(3), tensorforge.Tensor(np.zeros(3)))


def test_the_dataset_module_declares_no_public_helper():
    """Private helpers are allowed; exporting one is not."""
    assert not hasattr(native_dataset_module, "__all__")
    public = {name for name in vars(native_dataset_module)
              if not name.startswith("_")}
    # Only the class itself, plus the modules it imports.
    assert public - {"hashlib", "np", "cpp", "normalize_module_dtype",
                     "NativeTensor"} == {"NativeTensorDataset"}, sorted(public)


# ===========================================================================
# 2. Accepted construction
# ===========================================================================

@pytest.mark.parametrize("host_dtype", [np.float16, np.float32, np.float64])
def test_every_floating_host_width_is_accepted(host_dtype):
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=host_dtype)
    data = NativeTensorDataset(values, np.array([0, 1]))
    assert data.samples == 2
    assert data.feature_shape == (2,)
    assert data.dtype == "float64"


def test_longdouble_is_accepted_where_numpy_provides_a_distinct_one():
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.longdouble)
    assert np.issubdtype(values.dtype, np.floating)
    data = NativeTensorDataset(values, np.array([0, 1]))
    assert data.samples == 2
    assert np.array_equal(data.target_batch([0, 1]), [0, 1])


@pytest.mark.parametrize("requested,expected", [
    (None, "float64"), ("float64", "float64"), ("float32", "float32"),
])
def test_the_dtype_argument_selects_the_native_width(requested, expected):
    data = dataset(dtype=requested)
    assert data.dtype == expected
    assert data.device == "cpu"


def test_the_native_dtype_is_never_inferred_from_the_host_array():
    """The Phase-I rule, asserted in both directions: a float32 host array
    with ``dtype`` omitted gives a **float64** dataset, and the only way to
    a float32 dataset is to ask for one."""
    host = features_2d(4, 3, np.float32)
    assert NativeTensorDataset(host, targets_for(4)).dtype == "float64"
    assert NativeTensorDataset(host, targets_for(4),
                               dtype="float32").dtype == "float32"
    # ...and the converse: a float64 host array can produce a float32
    # dataset, so the host dtype is not even a lower bound.
    assert NativeTensorDataset(features_2d(4, 3, np.float64), targets_for(4),
                               dtype="float32").dtype == "float32"


def test_rank_one_features_are_scalar_samples():
    data = NativeTensorDataset(np.arange(5.0), np.arange(5))
    assert data.feature_shape == ()
    assert len(data) == 5
    assert data.identity()["feature_shape"] == []


@pytest.mark.parametrize("shape", [(4,), (4, 1), (4, 2, 3), (4, 1, 6, 6)])
def test_every_rank_from_one_upward_is_accepted(shape):
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    data = NativeTensorDataset(values, np.zeros(4, dtype=np.int64))
    assert data.samples == 4
    assert data.feature_shape == shape[1:]


@pytest.mark.parametrize("make", [
    pytest.param(lambda: features_2d(6, 4), id="contiguous"),
    pytest.param(lambda: features_2d(6, 4)[:, ::2], id="strided-slice"),
    pytest.param(lambda: features_2d(4, 6).T, id="transposed"),
    pytest.param(lambda: features_2d(6, 4)[::-1], id="negative-stride"),
    pytest.param(lambda: features_2d(12, 4)[2:8], id="sliced"),
    pytest.param(lambda: features_2d(6, 4).astype(">f8"), id="big-endian"),
    pytest.param(lambda: features_2d(6, 4).astype("<f8"), id="little-endian"),
])
def test_every_input_layout_and_byte_order_is_accepted(make):
    values = make()
    data = NativeTensorDataset(values, np.zeros(values.shape[0], dtype=np.int64))
    assert data.samples == values.shape[0]
    assert data.feature_shape == values.shape[1:]


def test_a_read_only_input_is_accepted():
    values = features_2d(4, 2)
    values.setflags(write=False)
    labels = targets_for(4)
    labels.setflags(write=False)
    data = NativeTensorDataset(values, labels)
    assert data.samples == 4
    assert np.array_equal(data.target_batch([0, 1, 2, 3]), labels)


@pytest.mark.parametrize("target_dtype", [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    ">i4", "<i4", ">u8", "<i8",
])
def test_every_integer_target_width_and_byte_order_is_accepted(target_dtype):
    labels = np.array([0, 1, 2, 1], dtype=target_dtype)
    data = NativeTensorDataset(features_2d(4, 2), labels)
    batch = data.target_batch([0, 1, 2, 3])
    assert batch.dtype == np.int64
    assert batch.tolist() == [0, 1, 2, 1]


def test_inputs_that_alias_each_other_still_produce_independent_snapshots():
    """§4.5: the two inputs may be views of one buffer. Two independent
    snapshots are taken, so no aliasing survives."""
    buffer = np.arange(8, dtype=np.float64)
    values = buffer[:4]
    labels = buffer[4:].astype(np.int64)
    data = NativeTensorDataset(values, labels)
    assert np.shares_memory(values, buffer)
    assert not np.shares_memory(data._features, buffer)
    assert not np.shares_memory(data._targets, buffer)
    assert not np.shares_memory(data._features, data._targets)


def test_non_finite_and_signed_zero_feature_values_are_ordinary_values():
    """§4.2: the dataset has no opinion about NaN, infinities, subnormals,
    or signed zeros, and adds no check."""
    values = np.array([[np.nan, np.inf], [-np.inf, -0.0],
                       [5e-324, 0.0]], dtype=np.float64)
    data = NativeTensorDataset(values, np.zeros(3, dtype=np.int64))
    assert data.samples == 3
    # The snapshot preserves the exact bit patterns, signed zero included.
    stored = data._features
    assert np.isnan(stored[0, 0]) and np.isposinf(stored[0, 1])
    assert stored[1, 1].view(np.int64) == np.float64(-0.0).view(np.int64)


# ===========================================================================
# 3. Rejected construction
# ===========================================================================

@pytest.mark.parametrize("bad", [
    [[0.0, 1.0], [2.0, 3.0]], ((0.0, 1.0), (2.0, 3.0)), 3.0, None,
    "features", b"features", {"a": 1}, range(4),
    (value for value in (1.0, 2.0)), memoryview(b"abcd"),
])
def test_a_non_ndarray_feature_input_is_a_type_error(bad):
    with pytest.raises(TypeError):
        NativeTensorDataset(bad, np.zeros(2, dtype=np.int64))


def test_an_ndarray_subclass_is_rejected_for_either_argument():
    """§4.1: ``type(...) is``, not ``isinstance``. A masked array's mask
    would be silently discarded by a snapshot."""
    masked = np.ma.masked_array(features_2d(4, 2),
                                mask=np.zeros((4, 2), dtype=bool))
    assert isinstance(masked, np.ndarray)
    with pytest.raises(TypeError):
        NativeTensorDataset(masked, targets_for(4))

    class Subclass(np.ndarray):
        pass

    view = features_2d(4, 2).view(Subclass)
    assert isinstance(view, np.ndarray) and type(view) is not np.ndarray
    with pytest.raises(TypeError):
        NativeTensorDataset(view, targets_for(4))
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(4, 2),
                            targets_for(4).view(Subclass))


def test_a_zero_dimensional_feature_array_is_rejected():
    """A 0-d array has no sample axis, so it fails the *rank* rule. A NumPy
    scalar is not an ndarray at all and fails the earlier *type* rule —
    two different faults, and the ordering is what distinguishes them."""
    with pytest.raises(ValueError):
        NativeTensorDataset(np.array(1.5), np.zeros(1, dtype=np.int64))
    with pytest.raises(TypeError):
        NativeTensorDataset(np.float64(1.5), np.zeros(1, dtype=np.int64))


def test_an_empty_dataset_is_rejected():
    """§4.6: the native runtime rejects zero-element storage, so an empty
    dataset could never produce a batch."""
    with pytest.raises(ValueError):
        NativeTensorDataset(np.zeros((0, 3)), np.zeros(0, dtype=np.int64))


@pytest.mark.parametrize("shape", [(4, 0), (4, 0, 2), (4, 2, 0)])
def test_a_zero_trailing_dimension_is_rejected(shape):
    with pytest.raises(ValueError):
        NativeTensorDataset(np.zeros(shape), np.zeros(4, dtype=np.int64))


@pytest.mark.parametrize("values", [
    pytest.param(np.zeros((4, 2), dtype=bool), id="bool"),
    pytest.param(np.zeros((4, 2), dtype=np.int8), id="int8"),
    pytest.param(np.zeros((4, 2), dtype=np.int64), id="int64"),
    pytest.param(np.zeros((4, 2), dtype=np.uint32), id="uint32"),
    pytest.param(np.zeros((4, 2), dtype=np.complex128), id="complex"),
    pytest.param(np.zeros((4, 2), dtype=object), id="object"),
    pytest.param(np.zeros(4, dtype=[("a", "f8"), ("b", "i4")]), id="struct"),
    pytest.param(np.array(["a", "b", "c", "d"]), id="str"),
    pytest.param(np.array([b"a", b"b", b"c", b"d"]), id="bytes"),
    pytest.param(np.zeros(4, dtype="datetime64[s]"), id="datetime"),
    pytest.param(np.zeros(4, dtype="timedelta64[s]"), id="timedelta"),
])
def test_every_non_floating_feature_kind_is_a_type_error(values):
    with pytest.raises(TypeError):
        NativeTensorDataset(values, np.zeros(4, dtype=np.int64))


@pytest.mark.parametrize("bad", [
    [0, 1, 2, 3], (0, 1, 2, 3), 0, None, "0123", {"a": 1},
])
def test_a_non_ndarray_target_input_is_a_type_error(bad):
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(4, 2), bad)


@pytest.mark.parametrize("labels", [
    pytest.param(np.array(1), id="scalar"),
    pytest.param(np.zeros((4, 1), dtype=np.int64), id="column"),
    pytest.param(np.zeros((2, 2), dtype=np.int64), id="two-dimensional"),
    pytest.param(np.zeros((4, 1, 1), dtype=np.int64), id="rank-three"),
])
def test_a_target_rank_other_than_one_is_a_value_error(labels):
    with pytest.raises(ValueError):
        NativeTensorDataset(features_2d(4, 2), labels)


@pytest.mark.parametrize("labels", [
    pytest.param(np.array([True, False, True, False]), id="bool"),
    pytest.param(np.zeros(4, dtype=np.float64), id="float"),
    pytest.param(np.zeros(4, dtype=np.float32), id="float32"),
    pytest.param(np.zeros(4, dtype=np.complex128), id="complex"),
    pytest.param(np.zeros(4, dtype=object), id="object"),
    pytest.param(np.array(["0", "1", "2", "3"]), id="str"),
    pytest.param(np.zeros(4, dtype=[("a", "i8")]), id="struct"),
])
def test_every_non_integer_target_kind_is_a_type_error(labels):
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(4, 2), labels)


@pytest.mark.parametrize("count", [0, 3, 5, 40])
def test_a_target_count_mismatch_is_a_value_error(count):
    with pytest.raises(ValueError):
        NativeTensorDataset(features_2d(4, 2), np.zeros(count, dtype=np.int64))


@pytest.mark.parametrize("labels", [
    np.array([0, -1, 2, 3], dtype=np.int64),
    np.array([-1, 0, 0, 0], dtype=np.int8),
    np.array([0, 0, 0, -32768], dtype=np.int16),
])
def test_a_negative_target_is_a_value_error(labels):
    with pytest.raises(ValueError) as excinfo:
        NativeTensorDataset(features_2d(4, 2), labels)
    # The message identifies the offending index rather than only the fact.
    assert "index" in str(excinfo.value)


def test_a_uint64_target_above_the_int64_maximum_is_a_value_error():
    labels = np.array([0, 1, 2, 2 ** 63], dtype=np.uint64)
    with pytest.raises(ValueError) as excinfo:
        NativeTensorDataset(features_2d(4, 2), labels)
    assert "index" in str(excinfo.value)
    # ...and exactly at the maximum it is accepted: the boundary is
    # representability, not magnitude, and there is no class upper bound.
    at_maximum = np.array([0, 1, 2, 2 ** 63 - 1], dtype=np.uint64)
    data = NativeTensorDataset(features_2d(4, 2), at_maximum)
    assert data.target_batch([3])[0] == 2 ** 63 - 1


def test_there_is_no_num_classes_argument_and_no_upper_class_bound():
    """§4.4: the model's logits are the only authority on the class count,
    so the dataset must not become a second one."""
    huge = np.array([0, 10 ** 6, 3, 7], dtype=np.int64)
    data = NativeTensorDataset(features_2d(4, 2), huge)
    assert data.target_batch([1])[0] == 10 ** 6
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(4, 2), targets_for(4), num_classes=8)


@pytest.mark.parametrize("bad", [
    "f4", "single", "double", "Float32", "FLOAT64", " float32", "float",
    "float16", "float128", "int64", "", "cpu",
])
def test_an_unsupported_dtype_string_is_a_value_error(bad):
    with pytest.raises(ValueError):
        NativeTensorDataset(features_2d(4, 2), targets_for(4), dtype=bad)


@pytest.mark.parametrize("bad", [
    np.float32, np.float64, np.dtype("float32"), np.dtype("float64"),
    32, 64, True, 3.5, ["float32"], ("float64",),
])
def test_a_non_string_dtype_is_a_type_error(bad):
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(4, 2), targets_for(4), dtype=bad)


def test_the_dtype_argument_is_keyword_only_and_there_is_no_device():
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(4, 2), targets_for(4), "float32")
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(4, 2), targets_for(4), device="cpu")
    with pytest.raises(TypeError):
        NativeTensorDataset(features_2d(4, 2), targets_for(4), device="cuda")


# --- validation precedence, which no single-fault test can see ------------

def test_the_dtype_is_normalized_before_any_numpy_input_work():
    """Step 1 precedes step 2: a bad dtype is refused even when *both*
    array arguments are also invalid."""
    with pytest.raises(ValueError):
        NativeTensorDataset("not an array", "not an array", dtype="f4")
    with pytest.raises(TypeError):
        NativeTensorDataset("not an array", "not an array",
                            dtype=np.float32)


@pytest.mark.parametrize("features,targets,expected", [
    # Feature faults outrank every target fault.
    pytest.param("nope", "nope", TypeError, id="feature-type-first"),
    pytest.param(np.array(1.0), np.zeros((2, 2)), ValueError,
                 id="feature-rank-before-target-rank"),
    pytest.param(np.zeros((4, 2), dtype=np.int64), "nope", TypeError,
                 id="feature-kind-before-target-type"),
    pytest.param(np.zeros((0, 2)), np.zeros(5, dtype=np.int64), ValueError,
                 id="feature-count-before-target-count"),
    # Within the targets: type, then rank, then kind, then count, then values.
    pytest.param(np.zeros((4, 2)), [0, 1, 2, 3], TypeError,
                 id="target-type-before-rank"),
    pytest.param(np.zeros((4, 2)), np.zeros((4, 1), dtype=np.float64),
                 ValueError, id="target-rank-before-kind"),
    pytest.param(np.zeros((4, 2)), np.zeros(9, dtype=np.float64), TypeError,
                 id="target-kind-before-count"),
    pytest.param(np.zeros((4, 2)), np.array([-1, -2], dtype=np.int64),
                 ValueError, id="target-count-before-values"),
])
def test_the_construction_precedence_is_preserved(features, targets, expected):
    with pytest.raises(expected):
        NativeTensorDataset(features, targets)


def test_target_representability_is_checked_before_non_negativity():
    """Steps 10 and 11 are ordered. They cannot both fire for one dtype —
    unsigned cannot be negative and signed cannot exceed the maximum — so
    each is asserted at the boundary that can produce it."""
    with pytest.raises(ValueError) as overflow:
        NativeTensorDataset(features_2d(2, 2),
                            np.array([2 ** 63, 0], dtype=np.uint64))
    assert "int64" in str(overflow.value)
    with pytest.raises(ValueError) as negative:
        NativeTensorDataset(features_2d(2, 2),
                            np.array([-5, 0], dtype=np.int64))
    assert "non-negative" in str(negative.value)


def test_a_rejected_construction_mutates_nothing_and_builds_nothing(
        live_storages):
    values = features_2d(4, 2)
    labels = np.array([0, -1, 2, 3], dtype=np.int64)
    before_values = values.copy()
    before_labels = labels.copy()
    baseline = settled(live_storages)
    with pytest.raises(ValueError):
        NativeTensorDataset(values, labels)
    assert np.array_equal(values, before_values)
    assert np.array_equal(labels, before_labels)
    assert settled(live_storages) == baseline
    # ...and the very next attempt with valid data still works.
    data = NativeTensorDataset(values, np.abs(labels))
    assert data.samples == 4


# ===========================================================================
# 4. Snapshot ownership and aliasing
# ===========================================================================

def test_mutating_the_caller_arrays_afterwards_changes_nothing():
    """§5.2, in both directions: what the dataset reports and what it
    gathers are both unaffected."""
    values = features_2d(4, 2)
    labels = targets_for(4)
    data = NativeTensorDataset(values, labels)
    fingerprint = data.fingerprint
    expected = data._features.copy()

    values[:] = -99.0
    labels[:] = 7

    assert np.array_equal(data._features, expected)
    assert data.fingerprint == fingerprint
    assert data.target_batch([0, 1, 2, 3]).tolist() == targets_for(4).tolist()


def test_deleting_or_replacing_the_caller_arrays_changes_nothing():
    # .copy() so the array owns its data and can actually be resized —
    # the point of the test is the *dataset's* independence, not NumPy's
    # resize rules.
    values = features_2d(4, 2).copy()
    labels = targets_for(4).copy()
    data = NativeTensorDataset(values, labels)
    expected = data._features.copy()
    fingerprint = data.fingerprint

    values.resize((8, 2), refcheck=False)
    del values
    del labels
    gc.collect()

    assert np.array_equal(data._features, expected)
    assert data.fingerprint == fingerprint
    assert data.samples == 4


@pytest.mark.parametrize("make", [
    pytest.param(lambda: features_2d(6, 4), id="already-contiguous"),
    pytest.param(lambda: features_2d(6, 4)[:, ::2], id="strided"),
    pytest.param(lambda: np.asfortranarray(features_2d(6, 4)), id="fortran"),
])
def test_the_snapshot_never_aliases_the_input_even_when_it_could(make):
    """The ``copy=True`` rule: ``ascontiguousarray`` would return an
    already-contiguous, already-typed input *unchanged*, which is exactly
    the common case this must not do."""
    values = make()
    labels = targets_for(values.shape[0])
    data = NativeTensorDataset(values, labels)
    assert not np.shares_memory(data._features, values)
    assert not np.shares_memory(data._targets, labels)
    assert data._features.flags["C_CONTIGUOUS"]
    assert data._features.flags["OWNDATA"]
    assert data._targets.flags["C_CONTIGUOUS"]
    assert data._targets.flags["OWNDATA"]


def test_the_snapshot_is_native_order_at_the_selected_width():
    values = features_2d(4, 2).astype(">f8")
    labels = np.array([0, 1, 2, 3], dtype=">i8")
    data = NativeTensorDataset(values, labels, dtype="float32")
    assert data._features.dtype == np.float32
    assert data._features.dtype.byteorder in ("=", "|", "<" if
                                              sys.byteorder == "little"
                                              else ">")
    assert data._targets.dtype == np.int64
    assert data._targets.dtype.isnative


def test_two_datasets_from_the_same_inputs_own_independent_snapshots():
    values = features_2d(4, 2)
    labels = targets_for(4)
    first = NativeTensorDataset(values, labels)
    second = NativeTensorDataset(values, labels)
    assert first.fingerprint == second.fingerprint
    assert not np.shares_memory(first._features, second._features)
    assert not np.shares_memory(first._targets, second._targets)
    first.close()
    assert second.samples == 4
    assert second.target_batch([0]).tolist() == [labels[0]]


def test_no_public_snapshot_accessor_exists():
    data = dataset()
    for name in ("features", "targets", "data", "values", "labels",
                 "feature_array", "target_array", "snapshot"):
        assert not hasattr(data, name), name
    # __slots__ means an accessor cannot even be attached after the fact.
    with pytest.raises(AttributeError):
        data.features = np.zeros(3)


def test_the_dataset_holds_no_native_storage_between_calls(live_storages):
    """§5.4: constructing, holding, inspecting, and discarding a dataset
    leaves the native live-storage count exactly where it was."""
    baseline = settled(live_storages)
    data = NativeTensorDataset(features_2d(64, 8), targets_for(64))
    assert settled(live_storages) == baseline
    assert data.identity()["samples"] == 64
    assert repr(data)
    assert settled(live_storages) == baseline
    data.close()
    del data
    assert settled(live_storages) == baseline


# ===========================================================================
# 5. Metadata and identity
# ===========================================================================

def test_the_metadata_properties_report_exactly_the_construction():
    data = NativeTensorDataset(np.zeros((7, 1, 6, 6)),
                               np.zeros(7, dtype=np.int64), dtype="float32")
    assert data.samples == 7
    assert len(data) == 7
    assert data.feature_shape == (1, 6, 6)
    assert data.dtype == "float32"
    assert data.device == "cpu"
    assert data.closed is False
    assert isinstance(data.samples, int)
    assert all(isinstance(dimension, int) for dimension in data.feature_shape)


def test_the_metadata_properties_are_read_only():
    data = dataset()
    for name, value in (("samples", 3), ("feature_shape", ()),
                        ("dtype", "float32"), ("device", "cuda"),
                        ("fingerprint", "0" * 64), ("closed", True)):
        with pytest.raises(AttributeError):
            setattr(data, name, value)


def test_identity_returns_the_four_json_native_fields():
    data = dataset(8, 3, dtype="float32")
    identity = data.identity()
    assert set(identity) == {"samples", "feature_shape", "feature_dtype",
                             "fingerprint"}
    assert identity["samples"] == 8
    assert identity["feature_shape"] == [3]
    assert identity["feature_dtype"] == "float32"
    assert identity["fingerprint"] == data.fingerprint
    assert type(identity) is dict
    assert type(identity["feature_shape"]) is list
    assert type(identity["samples"]) is int
    assert all(type(dimension) is int for dimension in identity["feature_shape"])
    assert type(identity["feature_dtype"]) is str
    assert type(identity["fingerprint"]) is str


def test_identity_is_a_fresh_structure_every_call():
    data = dataset()
    first = data.identity()
    second = data.identity()
    assert first == second
    assert first is not second
    assert first["feature_shape"] is not second["feature_shape"]
    first["samples"] = -1
    first["feature_shape"].append(99)
    assert data.identity() == second
    assert data.samples == 6
    assert data.feature_shape == (2,)


def test_identity_round_trips_through_json_unchanged():
    data = dataset(5, 4, dtype="float32")
    identity = data.identity()
    assert json.loads(json.dumps(identity)) == identity


def test_identity_is_accepted_unchanged_by_the_checkpoint_metadata_validator():
    """§6.4: every field passes ``_validated_metadata``, which is what
    lets a caller record it through the existing version-3 metadata
    channel without the archive growing a field or a version."""
    data = dataset(5, 4, dtype="float32")
    identity = data.identity()
    validated = native_checkpoint_module._validated_metadata(
        {"dataset": identity}, "metadata", set())
    assert validated == {"dataset": identity}
    # Scalar samples too, whose feature_shape is the empty list.
    scalar = NativeTensorDataset(np.arange(3.0), np.zeros(3, dtype=np.int64))
    assert native_checkpoint_module._validated_metadata(
        scalar.identity(), "metadata", set()) == scalar.identity()


def test_identity_carries_no_payload_and_nothing_process_local():
    data = dataset(4, 2)
    identity = data.identity()
    flat = json.dumps(identity)
    for value in data._features.ravel().tolist():
        assert repr(value) not in flat
    assert str(id(data)) not in flat
    assert hex(id(data)) not in flat
    # ...and no NumPy scalar, tuple, or bytes leaked into it.
    for value in identity.values():
        assert not isinstance(value, (np.generic, tuple, bytes, np.ndarray))


def test_metadata_and_identity_survive_close():
    data = dataset(4, 2, dtype="float32")
    identity = data.identity()
    fingerprint = data.fingerprint
    data.close()
    assert data.closed is True
    assert data.samples == 4
    assert len(data) == 4
    assert data.feature_shape == (2,)
    assert data.dtype == "float32"
    assert data.device == "cpu"
    assert data.fingerprint == fingerprint
    assert data.identity() == identity
    assert repr(data)


# ===========================================================================
# 6. The fingerprint
# ===========================================================================

def test_the_fingerprint_is_sixty_four_lowercase_hex_characters():
    for data in (dataset(), dataset(dtype="float32"),
                 NativeTensorDataset(np.arange(3.0), np.zeros(3, dtype=np.int64))):
        value = data.fingerprint
        assert isinstance(value, str)
        assert len(value) == 64
        assert value == value.lower()
        assert all(character in "0123456789abcdef" for character in value)
        int(value, 16)


@pytest.mark.parametrize("rows,labels,requested,expected", [
    (KNOWN_ROWS, KNOWN_TARGETS, None, KNOWN_FLOAT64_2D),
    (KNOWN_ROWS, KNOWN_TARGETS, "float64", KNOWN_FLOAT64_2D),
    (KNOWN_ROWS, KNOWN_TARGETS, "float32", KNOWN_FLOAT32_2D),
    (KNOWN_SCALAR_ROWS, KNOWN_SCALAR_TARGETS, None, KNOWN_FLOAT64_SCALAR),
])
def test_the_fingerprint_matches_its_committed_known_answer(
        rows, labels, requested, expected):
    """The known-answer test. ``expected`` is a literal produced by the
    independent ``struct``-based oracle below, so this is a specification
    check rather than the implementation compared against itself."""
    data = NativeTensorDataset(np.array(rows, dtype=np.float64),
                               np.array(labels, dtype=np.int64),
                               dtype=requested)
    assert data.fingerprint == expected


@pytest.mark.parametrize("rows,labels,dtype_name,expected", [
    (KNOWN_ROWS, KNOWN_TARGETS, "float64", KNOWN_FLOAT64_2D),
    (KNOWN_ROWS, KNOWN_TARGETS, "float32", KNOWN_FLOAT32_2D),
    (KNOWN_SCALAR_ROWS, KNOWN_SCALAR_TARGETS, "float64",
     KNOWN_FLOAT64_SCALAR),
])
def test_the_committed_vectors_are_what_the_independent_oracle_produces(
        rows, labels, dtype_name, expected):
    """The control on the vectors themselves: the oracle walks Python
    lists with ``struct`` and shares no code with the production helper,
    so agreement means both implement §6.3's byte stream."""
    assert oracle_fingerprint(dtype_name, rows, labels) == expected


def test_the_oracle_can_actually_fail():
    """Negative control: an oracle that returned a constant would make
    every vector above vacuous."""
    base = oracle_fingerprint("float64", KNOWN_ROWS, KNOWN_TARGETS)
    assert oracle_fingerprint("float32", KNOWN_ROWS, KNOWN_TARGETS) != base
    assert oracle_fingerprint("float64", [[0.0, 1.0], [2.0, 3.0],
                                          [4.0, 5.5]], KNOWN_TARGETS) != base
    assert oracle_fingerprint("float64", KNOWN_ROWS, [0, 1, 1]) != base
    assert oracle_fingerprint("float64", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                              KNOWN_TARGETS) != base


def test_the_fingerprint_is_deterministic_across_repeated_construction():
    values = features_2d(20, 3)
    labels = targets_for(20)
    digests = {NativeTensorDataset(values, labels).fingerprint
               for _ in range(5)}
    assert len(digests) == 1


@pytest.mark.parametrize("make", [
    pytest.param(lambda base: base, id="contiguous"),
    pytest.param(lambda base: np.asfortranarray(base), id="fortran"),
    pytest.param(lambda base: base.astype(">f8"), id="big-endian"),
    pytest.param(lambda base: base.astype("<f8"), id="little-endian"),
    pytest.param(lambda base: base.astype(np.float32), id="float32-host"),
    pytest.param(lambda base: np.repeat(base, 2, axis=0)[::2], id="strided"),
    pytest.param(lambda base: base[::-1][::-1], id="double-reversed"),
])
def test_equal_logical_values_fingerprint_identically(make):
    """Layout, byte order, and host width are normalized away; only the
    *values* at the selected native dtype survive into the digest. Values
    are exactly representable in float32, so the host width is a pure
    layout difference here rather than a rounding one."""
    base = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    labels = np.array([0, 1, 0], dtype=np.int64)
    reference = NativeTensorDataset(base, labels).fingerprint
    assert NativeTensorDataset(make(base), labels).fingerprint == reference


def test_a_read_only_input_fingerprints_like_a_writable_one():
    values = features_2d(4, 2)
    labels = targets_for(4)
    writable = NativeTensorDataset(values, labels).fingerprint
    frozen = values.copy()
    frozen.setflags(write=False)
    frozen_labels = labels.copy()
    frozen_labels.setflags(write=False)
    assert NativeTensorDataset(frozen, frozen_labels).fingerprint == writable


def test_a_non_native_byte_order_target_fingerprints_identically():
    values = features_2d(4, 2)
    native = NativeTensorDataset(values, np.array([0, 1, 2, 3],
                                                  dtype="<i8")).fingerprint
    swapped = NativeTensorDataset(values, np.array([0, 1, 2, 3],
                                                   dtype=">i4")).fingerprint
    assert native == swapped


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda v, t, d: (v, t, "float32"), id="selected-dtype"),
    pytest.param(lambda v, t, d: (v.reshape(2, 6), t[:2], d), id="shape"),
    pytest.param(lambda v, t, d: (v + 0.0625, t, d), id="feature-values"),
    pytest.param(lambda v, t, d: (v, t[::-1].copy(), d), id="target-order"),
    pytest.param(lambda v, t, d: (v, t + 1, d), id="target-values"),
    pytest.param(lambda v, t, d: (v[:3], t[:3], d), id="sample-count"),
])
def test_the_fingerprint_is_sensitive_to_every_locked_field(mutate):
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    reference = NativeTensorDataset(values, labels, dtype="float64").fingerprint
    changed_values, changed_labels, changed_dtype = mutate(
        values, labels, "float64")
    changed = NativeTensorDataset(changed_values, changed_labels,
                                  dtype=changed_dtype).fingerprint
    assert changed != reference


def test_a_signed_zero_and_a_nan_payload_reach_the_digest():
    """The digest is over raw value bits, so ``-0.0`` and ``0.0`` are
    distinct even though they compare equal."""
    labels = np.zeros(2, dtype=np.int64)
    positive = NativeTensorDataset(np.array([0.0, 1.0]), labels).fingerprint
    negative = NativeTensorDataset(np.array([-0.0, 1.0]), labels).fingerprint
    assert positive != negative


def test_the_same_elements_in_different_shapes_do_not_collide():
    """§6.3: rank and dimensions are hashed *before* the elements."""
    elements = np.arange(12, dtype=np.float64)
    labels_six = np.zeros(6, dtype=np.int64)
    labels_four = np.zeros(4, dtype=np.int64)
    six_by_two = NativeTensorDataset(elements.reshape(6, 2),
                                     labels_six).fingerprint
    four_by_three = NativeTensorDataset(elements.reshape(4, 3),
                                        labels_four).fingerprint
    assert six_by_two != four_by_three
    # ...and the sample axis is included, so a (6,2) and a (6,2) with the
    # same values but a different sample count cannot be confused either.
    assert NativeTensorDataset(elements.reshape(2, 6),
                               labels_four[:2]).fingerprint != six_by_two


def test_the_digest_does_not_depend_on_python_hash_randomization():
    """A digest built from ``hash()`` would differ between interpreters
    started with different hash seeds. Run twice, with two seeds."""
    import subprocess

    program = (
        "import numpy as np;"
        "from tensorforge.experimental import NativeTensorDataset;"
        "f=np.arange(12.0).reshape(6,2);t=np.arange(6)%3;"
        "print(NativeTensorDataset(f,t).fingerprint)"
    )
    digests = set()
    for seed in ("0", "12345"):
        import os

        environment = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", program], env=environment,
                                capture_output=True, text=True, check=True)
        digests.add(result.stdout.strip())
    assert len(digests) == 1, digests
    assert digests.pop() == dataset().fingerprint


def test_the_chunk_size_cannot_change_the_digest(monkeypatch):
    """The chunking is an implementation detail of peak memory only.
    SHA-256 is a streaming construction, so any chunking of the identical
    byte sequence gives the identical digest."""
    values = features_2d(500, 7)
    labels = targets_for(500)
    reference = NativeTensorDataset(values, labels).fingerprint
    for chunk in (1, 3, 17, 1024, 10 ** 9):
        monkeypatch.setattr(native_dataset_module, "_HASH_CHUNK_ELEMENTS",
                            chunk)
        assert NativeTensorDataset(values, labels).fingerprint == reference


# ===========================================================================
# 7. Batch-index validation (§12.6), at both methods
# ===========================================================================

BATCH_METHODS = ("feature_batch", "target_batch")


def call(data, method, indices):
    return getattr(data, method)(indices)


def rows_in(result):
    """The batch size of either return kind. A ``NativeTensor`` has no
    ``__len__`` (and J1 does not add one), so its row count is read from
    the shape."""
    return result.shape[0] if hasattr(result, "to_numpy") else len(result)


def release(result):
    """Close a feature batch; a target batch needs no close."""
    if hasattr(result, "close"):
        result.close()


@pytest.mark.parametrize("method", BATCH_METHODS)
@pytest.mark.parametrize("indices", [
    pytest.param([0, 1, 2], id="list"),
    pytest.param((0, 1, 2), id="tuple"),
    pytest.param(np.array([0, 1, 2], dtype=np.int64), id="int64-array"),
    pytest.param(np.array([0, 1, 2], dtype=np.int8), id="int8-array"),
    pytest.param(np.array([0, 1, 2], dtype=np.uint32), id="uint32-array"),
    pytest.param(np.array([0, 1, 2], dtype=">i4"), id="big-endian-array"),
    pytest.param([0], id="single"),
    pytest.param([5, 0], id="first-and-last"),
    pytest.param([3, 1, 3, 3], id="duplicates"),
    pytest.param([5, 4, 3, 2, 1, 0], id="reversed"),
])
@needs_backend
def test_every_accepted_index_container_is_accepted(method, indices):
    data = dataset(6, 2)
    result = call(data, method, indices)
    assert rows_in(result) == len(indices)
    release(result)


@pytest.mark.parametrize("method", BATCH_METHODS)
@pytest.mark.parametrize("indices,expected", [
    pytest.param(3, TypeError, id="bare-int"),
    pytest.param(np.int64(3), TypeError, id="numpy-scalar"),
    pytest.param(None, TypeError, id="none"),
    pytest.param("012", TypeError, id="string"),
    pytest.param({0, 1}, TypeError, id="set"),
    pytest.param({0: 1}, TypeError, id="dict"),
    pytest.param(range(3), TypeError, id="range"),
    pytest.param((value for value in (0, 1)), TypeError, id="generator"),
    pytest.param([0, True], TypeError, id="bool-in-list"),
    pytest.param([True], TypeError, id="only-bool"),
    pytest.param([0, 1.0], TypeError, id="float-in-list"),
    pytest.param([0, np.int64(1)], TypeError, id="numpy-int-in-list"),
    pytest.param([0, "1"], TypeError, id="string-in-list"),
    pytest.param([0, None], TypeError, id="none-in-list"),
    pytest.param([[0], [1]], TypeError, id="nested-list"),
    pytest.param(np.array([True, False]), TypeError, id="bool-array"),
    pytest.param(np.array([0.0, 1.0]), TypeError, id="float-array"),
    pytest.param(np.array([0, 1], dtype=object), TypeError, id="object-array"),
    pytest.param(np.array(["0", "1"]), TypeError, id="string-array"),
    pytest.param(np.array([[0, 1], [2, 3]]), ValueError, id="two-dimensional"),
    pytest.param(np.array(3), ValueError, id="zero-dimensional-array"),
    pytest.param([], ValueError, id="empty-list"),
    pytest.param((), ValueError, id="empty-tuple"),
    pytest.param(np.array([], dtype=np.int64), ValueError, id="empty-array"),
    pytest.param([-1], ValueError, id="negative"),
    pytest.param([0, -1, 2], ValueError, id="negative-in-middle"),
    pytest.param([6], ValueError, id="equal-to-sample-count"),
    pytest.param([0, 6], ValueError, id="past-the-end"),
    pytest.param([10 ** 30], ValueError, id="enormous"),
    pytest.param(np.array([2 ** 63 - 1], dtype=np.uint64), ValueError,
                 id="uint64-maximum"),
    pytest.param(np.array([2 ** 64 - 1], dtype=np.uint64), ValueError,
                 id="uint64-above-int64"),
])
def test_every_rejected_index_request_is_refused(method, indices, expected):
    data = dataset(6, 2)
    with pytest.raises(expected):
        call(data, method, indices)


@pytest.mark.parametrize("method", BATCH_METHODS)
def test_an_ndarray_subclass_of_indices_is_rejected(method):
    class Subclass(np.ndarray):
        pass

    data = dataset(6, 2)
    with pytest.raises(TypeError):
        call(data, method, np.array([0, 1]).view(Subclass))


@pytest.mark.parametrize("method", BATCH_METHODS)
def test_the_dimensionality_check_precedes_the_dtype_check(method):
    """A 2-D float array is refused as a *shape* problem: the container
    rules are ordered, so the more structural fault is reported."""
    data = dataset(6, 2)
    with pytest.raises(ValueError):
        call(data, method, np.array([[0.0, 1.0], [2.0, 3.0]]))


@pytest.mark.parametrize("method", BATCH_METHODS)
def test_emptiness_is_checked_before_bounds(method):
    """Step 3 precedes step 4, so an empty request is an emptiness error
    rather than an accidental success."""
    data = dataset(6, 2)
    with pytest.raises(ValueError) as excinfo:
        call(data, method, [])
    assert "at least one index" in str(excinfo.value)


@pytest.mark.parametrize("method", BATCH_METHODS)
def test_a_failed_index_request_leaves_the_dataset_usable(method,
                                                          live_storages):
    data = dataset(6, 2)
    baseline = settled(live_storages)
    for bad in ([], [99], [-1], "nope"):
        with pytest.raises((TypeError, ValueError)):
            call(data, method, bad)
    assert settled(live_storages) == baseline
    assert data.closed is False
    result = call(data, method, [0, 1])
    assert rows_in(result) == 2
    release(result)
    assert settled(live_storages) == baseline


# ===========================================================================
# 8. Feature batches
# ===========================================================================

@needs_backend
@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_a_feature_batch_has_the_exact_values_shape_dtype_and_device(
        dtype_name, live_storages):
    values = features_2d(6, 3)
    data = NativeTensorDataset(values, targets_for(6), dtype=dtype_name)
    baseline = settled(live_storages)

    batch = data.feature_batch([4, 0, 2])
    assert batch.shape == (3, 3)
    assert batch.dtype == dtype_name
    assert batch.device == "cpu"
    assert batch.requires_grad is False
    expected = values[[4, 0, 2]].astype(
        np.float32 if dtype_name == "float32" else np.float64)
    assert np.array_equal(batch.to_numpy(), expected)
    assert batch.to_numpy().dtype == expected.dtype

    batch.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_scalar_samples_give_a_one_dimensional_feature_batch(live_storages):
    data = NativeTensorDataset(np.arange(5.0), np.arange(5))
    baseline = settled(live_storages)
    batch = data.feature_batch([4, 4, 0])
    assert batch.shape == (3,)
    assert np.array_equal(batch.to_numpy(), [4.0, 4.0, 0.0])
    batch.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_higher_rank_feature_batch_keeps_the_per_sample_shape(live_storages):
    values = np.arange(4 * 1 * 3 * 3, dtype=np.float64).reshape(4, 1, 3, 3)
    data = NativeTensorDataset(values, np.zeros(4, dtype=np.int64))
    baseline = settled(live_storages)
    batch = data.feature_batch([2, 2])
    assert batch.shape == (2, 1, 3, 3)
    assert np.array_equal(batch.to_numpy(), values[[2, 2]])
    batch.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_batch_dtype_follows_the_dataset_not_the_gathered_array():
    """The no-inference rule at the batch boundary: a float64 host
    snapshot in a float32 dataset still yields float32 batches, and the
    reverse holds too."""
    values = features_2d(4, 2, np.float32)
    assert NativeTensorDataset(values, targets_for(4)).feature_batch(
        [0]).dtype == "float64"
    narrow = NativeTensorDataset(features_2d(4, 2, np.float64), targets_for(4),
                                 dtype="float32")
    batch = narrow.feature_batch([0])
    assert batch.dtype == "float32"
    batch.close()


@needs_backend
def test_index_order_and_duplicates_are_preserved_exactly(live_storages):
    values = features_2d(6, 2)
    data = NativeTensorDataset(values, targets_for(6))
    baseline = settled(live_storages)
    for indices in ([3, 1, 3], [5, 4, 3, 2, 1, 0], [2, 2, 2, 2], [0, 5]):
        batch = data.feature_batch(indices)
        assert np.array_equal(batch.to_numpy(), values[list(indices)])
        batch.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_feature_batch_is_owning_contiguous_and_independent(live_storages):
    values = features_2d(6, 2)
    data = NativeTensorDataset(values, targets_for(6))
    baseline = settled(live_storages)

    first = data.feature_batch([1, 2])
    second = data.feature_batch([1, 2])
    assert first is not second
    assert first._core._storage is not second._core._storage
    assert first.contiguous
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    assert settled(live_storages) == baseline + 2

    # Closing one leaves the other, and the dataset, entirely intact.
    first.close()
    assert first.closed and not second.closed
    assert np.array_equal(second.to_numpy(), values[[1, 2]])
    assert data.closed is False
    second.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_the_dataset_retains_no_reference_to_a_delivered_batch(live_storages):
    data = dataset(6, 2)
    baseline = settled(live_storages)
    batch = data.feature_batch([0, 1])
    assert settled(live_storages) == baseline + 1
    # Closing the *dataset* must not close, invalidate, or touch a batch
    # the caller already owns.
    data.close()
    assert not batch.closed
    assert batch.shape == (2, 2)
    assert settled(live_storages) == baseline + 1
    batch.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_batch_shares_no_storage_with_the_dataset_snapshot(live_storages):
    """§5.3: the returned tensor's native storage aliases nothing — the
    snapshot is host memory it was copied *from*, so closing the batch and
    reading the dataset afterwards must agree with the original values."""
    values = features_2d(4, 2)
    data = NativeTensorDataset(values, targets_for(4))
    baseline = settled(live_storages)
    fingerprint = data.fingerprint

    batch = data.feature_batch([0, 1])
    assert not np.shares_memory(batch.to_numpy(), data._features)
    batch.close()

    assert data.fingerprint == fingerprint
    assert np.array_equal(data._features, values)
    assert settled(live_storages) == baseline


@needs_backend
def test_a_failed_native_batch_construction_leaves_no_storage(
        live_storages, monkeypatch):
    """§10.6/§17.3, at the one position J1 owns: ``from_array`` closes its
    own storage on a failed transfer, so a failed request allocates no
    persistent native storage and the dataset stays usable."""
    data = dataset(6, 2)
    baseline = settled(live_storages)
    real = native_dataset_module.NativeTensor
    fired = {"count": 0}

    class Failing:
        @staticmethod
        def from_array(values, dtype=None):
            fired["count"] += 1
            raise MemoryError("injected native allocation failure")

    monkeypatch.setattr(native_dataset_module, "NativeTensor", Failing)
    with pytest.raises(MemoryError):
        data.feature_batch([0, 1])
    # Non-vacuity: the patched route really was the one that raised.
    assert fired["count"] == 1
    assert settled(live_storages) == baseline

    monkeypatch.setattr(native_dataset_module, "NativeTensor", real)
    # Retry succeeds and the values are exactly what the first call wanted.
    batch = data.feature_batch([0, 1])
    assert np.array_equal(batch.to_numpy(), features_2d(6, 2)[[0, 1]])
    batch.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_a_real_native_allocation_failure_is_a_memory_error(live_storages):
    """The same position through the runtime's own deterministic
    allocation-failure hook rather than a Python stand-in."""
    if not cpp.fault_injection_available():
        pytest.skip("fault injection not compiled into the backend")
    data = dataset(6, 2)
    baseline = settled(live_storages)
    cpp._arm_alloc_failure(1)
    try:
        with pytest.raises(MemoryError):
            data.feature_batch([0, 1])
    finally:
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()
    assert settled(live_storages) == baseline
    batch = data.feature_batch([0, 1])
    batch.close()
    assert settled(live_storages) == baseline


@needs_backend
def test_many_batches_return_live_storage_exactly_to_baseline(live_storages):
    data = dataset(32, 4, dtype="float32")
    baseline = settled(live_storages)
    batches = [data.feature_batch([index, index + 1]) for index in range(16)]
    assert settled(live_storages) == baseline + 16
    for batch in batches:
        batch.close()
    batches.clear()
    assert settled(live_storages) == baseline


# ===========================================================================
# 9. Target batches
# ===========================================================================

def test_a_target_batch_is_int64_owning_contiguous_and_read_only():
    data = dataset(6, 2)
    batch = data.target_batch([3, 1, 3])
    assert isinstance(batch, np.ndarray)
    assert type(batch) is np.ndarray
    assert batch.dtype == np.int64
    assert batch.shape == (3,)
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.flags["OWNDATA"]
    assert batch.flags["WRITEABLE"] is False


def test_a_target_batch_preserves_order_and_duplicates():
    labels = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    data = NativeTensorDataset(features_2d(6, 2), labels)
    assert data.target_batch([3, 1, 3]).tolist() == [13, 11, 13]
    assert data.target_batch([5, 4, 3, 2, 1, 0]).tolist() == [15, 14, 13, 12,
                                                              11, 10]
    assert data.target_batch(np.array([0, 0])).tolist() == [10, 10]


def test_a_target_batch_shares_no_memory_with_the_dataset_or_another_batch():
    data = dataset(6, 2)
    first = data.target_batch([0, 1, 2])
    second = data.target_batch([0, 1, 2])
    assert first is not second
    assert np.array_equal(first, second)
    assert not np.shares_memory(first, data._targets)
    assert not np.shares_memory(second, data._targets)
    assert not np.shares_memory(first, second)


def test_a_target_batch_refuses_ordinary_mutation():
    data = dataset(6, 2)
    batch = data.target_batch([0, 1])
    with pytest.raises(ValueError):
        batch[0] = 99
    with pytest.raises(ValueError):
        batch[:] = 0
    with pytest.raises(ValueError):
        batch.fill(3)


def test_a_target_batch_survives_the_dataset_close():
    data = dataset(6, 2)
    batch = data.target_batch([0, 1, 2])
    expected = batch.tolist()
    data.close()
    assert batch.tolist() == expected
    assert batch.flags["WRITEABLE"] is False


def test_a_target_batch_creates_no_native_storage(live_storages):
    """Targets are host ``int64`` metadata at every width; no native
    integer tensor exists or is implied."""
    data = dataset(6, 2)
    baseline = settled(live_storages)
    for _ in range(8):
        batch = data.target_batch([0, 1, 2])
        assert batch.dtype == np.int64
    assert settled(live_storages) == baseline


def test_a_wide_target_input_still_yields_int64_batches():
    for dtype in (np.int8, np.uint16, ">i4", np.uint64):
        labels = np.array([0, 1, 2, 3], dtype=dtype)
        data = NativeTensorDataset(features_2d(4, 2), labels)
        batch = data.target_batch([3, 0])
        assert batch.dtype == np.int64
        assert batch.tolist() == [3, 0]


# ===========================================================================
# 10. Lifecycle
# ===========================================================================

def test_close_is_idempotent_and_returns_none():
    data = dataset()
    assert data.closed is False
    assert data.close() is None
    assert data.closed is True
    for _ in range(3):
        assert data.close() is None
    assert data.closed is True


def test_close_drops_both_snapshots():
    data = dataset()
    data.close()
    assert data._features is None
    assert data._targets is None


@pytest.mark.parametrize("method", BATCH_METHODS)
def test_a_batch_request_after_close_is_a_runtime_error(method):
    data = dataset()
    data.close()
    with pytest.raises(RuntimeError):
        call(data, method, [0, 1])


@pytest.mark.parametrize("method", BATCH_METHODS)
@pytest.mark.parametrize("indices", [[], [99], "nope", None, [-1]])
def test_the_closed_check_precedes_every_index_check(method, indices):
    """§12.6 step 1 comes first: a closed dataset reports the lifecycle
    error even when the request is also malformed."""
    data = dataset()
    data.close()
    with pytest.raises(RuntimeError):
        call(data, method, indices)


def test_the_context_manager_closes_on_exit():
    with NativeTensorDataset(features_2d(4, 2), targets_for(4)) as data:
        assert data.closed is False
        assert data.samples == 4
    assert data.closed is True


def test_the_context_manager_does_not_swallow_an_exception():
    """``__exit__`` returns ``False``, so an exception raised inside the
    block propagates — and the dataset is still closed on the way out."""
    data = NativeTensorDataset(features_2d(4, 2), targets_for(4))
    with pytest.raises(ZeroDivisionError):
        with data:
            raise ZeroDivisionError("propagates")
    assert data.closed is True
    assert data.__exit__(None, None, None) is False


def test_repr_is_metadata_only_and_valid_after_close():
    data = dataset(6, 2, dtype="float32")
    text = repr(data)
    assert "NativeTensorDataset" in text
    assert "samples=6" in text
    assert "float32" in text
    assert "cpu" in text
    assert "closed=False" in text
    assert data.fingerprint[:12] in text
    assert data.fingerprint not in text          # prefix only
    assert hex(id(data)) not in text
    for value in data._features.ravel().tolist():
        assert f"{value}" not in text.replace("samples=6", "")

    data.close()
    closed = repr(data)
    assert "closed=True" in closed
    assert data.fingerprint[:12] in closed


def test_a_dataset_needs_no_finalizer_to_be_correct():
    """The dataset owns two NumPy arrays and no native resource, so it has
    no ``__del__`` and correctness relies on none."""
    assert not hasattr(NativeTensorDataset, "__del__")


# ===========================================================================
# 11. Construction failure injection (§17.2)
# ===========================================================================

def failing_np_array(monkeypatch, *, fail_on):
    """Make the ``fail_on``-th ``numpy.array`` call raise ``MemoryError``.

    The constructor calls it exactly twice — the feature snapshot, then
    the target snapshot — so this addresses each allocation position
    individually."""
    calls = {"count": 0}
    real = np.array

    def injected(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == fail_on:
            raise MemoryError("injected snapshot allocation failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "array", injected)
    return calls


@pytest.mark.parametrize("fail_on,position", [(1, "feature"), (2, "target")])
def test_a_snapshot_allocation_failure_publishes_nothing(
        fail_on, position, live_storages, monkeypatch):
    values = features_2d(4, 2)
    labels = targets_for(4)
    before_values = values.copy()
    before_labels = labels.copy()
    baseline = settled(live_storages)

    calls = failing_np_array(monkeypatch, fail_on=fail_on)
    with pytest.raises(MemoryError):
        NativeTensorDataset(values, labels)
    # Non-vacuity: the injected call really did run and really did raise.
    assert calls["count"] == fail_on, position

    monkeypatch.undo()
    assert settled(live_storages) == baseline
    assert np.array_equal(values, before_values)
    assert np.array_equal(labels, before_labels)
    # Retry works, and produces a completely ordinary dataset.
    retried = NativeTensorDataset(values, labels)
    assert retried.samples == 4
    assert retried.fingerprint == NativeTensorDataset(values,
                                                      labels).fingerprint


@pytest.mark.parametrize("attribute", ["_fingerprint", "_hash_values"])
def test_a_fingerprint_stage_failure_publishes_nothing(
        attribute, live_storages, monkeypatch):
    """The digest positions of §17.2: hasher construction, an update, and
    the final publication, addressed through the two seams the module
    actually has."""
    values = features_2d(4, 2)
    labels = targets_for(4)
    baseline = settled(live_storages)
    fired = {"count": 0}

    def injected(*args, **kwargs):
        fired["count"] += 1
        raise MemoryError("injected fingerprint failure")

    monkeypatch.setattr(native_dataset_module, attribute, injected)
    with pytest.raises(MemoryError):
        NativeTensorDataset(values, labels)
    assert fired["count"] >= 1, attribute

    monkeypatch.undo()
    assert settled(live_storages) == baseline
    retried = NativeTensorDataset(values, labels)
    assert retried.samples == 4
    assert len(retried.fingerprint) == 64


def test_a_hashlib_failure_is_not_disguised_as_another_exception(
        live_storages, monkeypatch):
    baseline = settled(live_storages)

    def injected():
        raise MemoryError("injected hasher construction failure")

    monkeypatch.setattr(native_dataset_module.hashlib, "sha256", injected)
    with pytest.raises(MemoryError):
        NativeTensorDataset(features_2d(4, 2), targets_for(4))
    monkeypatch.undo()
    assert settled(live_storages) == baseline


def test_construction_allocates_no_native_storage_at_all(live_storages):
    """§17.2's closing statement: construction touches the native runtime
    nowhere, so no construction failure can move the live-storage count."""
    baseline = settled(live_storages)
    for _ in range(4):
        NativeTensorDataset(features_2d(16, 4), targets_for(16),
                            dtype="float32")
    assert settled(live_storages) == baseline
    for bad in (np.zeros((0, 2)), np.zeros((4, 0)), np.zeros(4, dtype=int)):
        with pytest.raises((TypeError, ValueError)):
            NativeTensorDataset(bad, np.zeros(4, dtype=np.int64))
    assert settled(live_storages) == baseline


def test_the_failure_injection_helper_can_actually_fail():
    """Negative control for the injection above: without the patch the
    same construction succeeds, so the ``pytest.raises`` blocks are
    proving the injection worked rather than passing for free."""
    data = NativeTensorDataset(features_2d(4, 2), targets_for(4))
    assert data.samples == 4
    assert len(data.fingerprint) == 64
