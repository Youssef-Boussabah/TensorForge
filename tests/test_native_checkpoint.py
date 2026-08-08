"""Tests for native checkpointing and deterministic file resume
(Advanced C++ v3.14).

``save_native_checkpoint(path, model, optimizer=None, metadata=None)``
writes one pickle-free NPZ archive — a UTF-8/JSON ``manifest`` entry
(format ``"tensorforge.native_checkpoint"``, format version 1) plus one
float64 array per model parameter (``model::000000`` …) and per Adam
moment (``optimizer::m::000000`` …, ``optimizer::v::000000`` …) —
through a collision-safe temporary file and one atomic ``os.replace``.
``load_native_checkpoint(path, model, optimizer=None)`` validates the
complete archive with ``allow_pickle=False`` before any live mutation
(exact format/fields, strict optimizer presence and type, model keys
and every array's dtype/shape against the live objects, optimizer
scalars through the constructors' own validators), stages independent
NativeTensors, commits through the existing NativeModule/optimizer
loaders only, closes every staged tensor, and returns the checkpoint's
metadata as an independent dict. Model loading keeps its existing
contract (versions +1, retained value-sensitive graphs become stale);
optimizer loading moves no versions. No pickle, no object arrays, no
scheduler/random-state capture, no ``map_location``. See
src/tensorforge/experimental/native_checkpoint.py.

NumPy appears as the explicit serialization boundary (savez /
allow_pickle=False load / to_numpy / from_array) and as the test
oracle; the framework's numerical computation stays native (a tripwire
test proves it).

Selector: python -m pytest -q -k "native_checkpoint"
"""

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

import tensorforge
import tensorforge.serialization as stable_serialization
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint as checkpoint_module

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


P_VALUES = np.array([[1.0, -2.0], [0.5, 3.0]])
G_VALUES = np.array([[0.5, -1.0], [2.0, 0.25]])
X_VALUES = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25], [-1.0, 1.5]])
Y_VALUES = np.array([[1.0], [-0.5], [0.25], [2.0]])


def _small_model(seed=0):
    """The smallest real model: one NativeLinear (keys weight/bias)."""
    return NativeLinear(2, 3, seed=seed)


def _mlp():
    return NativeSequential(
        NativeLinear(2, 8, seed=0),
        NativeReLU(),
        NativeLinear(8, 1, seed=1),
    )


def _set_grads(model, x=X_VALUES):
    """One forward/backward so every parameter has a gradient
    (a scalar sum works for any output width)."""
    model(NativeTensor.from_array(x)).sum().backward()


def _train(model, optimizer, x, y, steps):
    """``steps`` full deterministic iterations; returns loss floats."""
    loss_fn = NativeMSELoss()
    losses = []
    for _ in range(steps):
        prediction = model(x)
        loss = loss_fn(prediction, y)
        losses.append(float(loss.to_numpy()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss.close()
        prediction.close()
    return losses


def _manifest_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def _arrays_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _resave(path, arrays):
    """Rewrite ``path`` as an NPZ holding exactly ``arrays`` — the
    corruption-construction helper (tests only)."""
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def _tamper(source, target, mutate_manifest=None, mutate_arrays=None):
    """Copy the checkpoint at ``source`` to ``target`` with the given
    manifest/array mutations applied."""
    arrays = _arrays_of(source)
    manifest = json.loads(arrays.pop("manifest").tobytes().decode("utf-8"))
    if mutate_manifest is not None:
        result = mutate_manifest(manifest)
        manifest = result if result is not None else manifest
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )
    _resave(target, arrays)
    return target


def _model_fingerprint(model):
    return {
        name: (parameter.to_numpy(), parameter.version, parameter.grad)
        for name, parameter in model.named_parameters()
    }


def _assert_model_untouched(model, fingerprint):
    for name, parameter in model.named_parameters():
        values, version, grad = fingerprint[name]
        assert np.array_equal(parameter.to_numpy(), values), name
        assert parameter.version == version, name
        assert parameter.grad is grad, name


# ======================================================================
# 1. Public API and exports
# ======================================================================


@needs_native
def test_native_checkpoint_exports_and_signatures():
    # Experimental-only exports; nothing leaks into the stable
    # namespace or stable serialization, which is unchanged.
    assert not hasattr(tensorforge, "save_native_checkpoint")
    assert not hasattr(tensorforge, "load_native_checkpoint")
    for absent in ("save_native_checkpoint", "load_native_checkpoint"):
        assert not hasattr(stable_serialization, absent)
    for present in ("save_parameters", "load_parameters",
                    "save_checkpoint", "load_checkpoint"):
        assert hasattr(stable_serialization, present)
    # Narrow signatures: no strict, no map_location, no paths beyond
    # the destination.
    assert list(inspect.signature(save_native_checkpoint).parameters) == [
        "path", "model", "optimizer", "metadata",
    ]
    assert list(inspect.signature(load_native_checkpoint).parameters) == [
        "path", "model", "optimizer",
    ]


@needs_native
def test_native_checkpoint_rejects_bad_arguments(tmp_path):
    model = _small_model()
    path = tmp_path / "checkpoint.npz"
    with pytest.raises(TypeError, match="path must be a str"):
        save_native_checkpoint(42, model)
    with pytest.raises(TypeError, match="model must be a NativeModule"):
        save_native_checkpoint(path, "not a model")
    # Stable-framework objects are rejected by type, never converted.
    stable_model = tensorforge.nn.Sequential(tensorforge.nn.Linear(2, 3))
    with pytest.raises(TypeError, match="model must be a NativeModule"):
        save_native_checkpoint(path, stable_model)
    with pytest.raises(TypeError, match="optimizer must be None"):
        save_native_checkpoint(
            path, model, optimizer=tensorforge.Adam(stable_model.parameters())
        )
    with pytest.raises(ValueError, match="directory does not exist"):
        save_native_checkpoint(tmp_path / "missing" / "x.npz", model)
    with pytest.raises(ValueError, match="is a directory"):
        save_native_checkpoint(tmp_path, model)
    with pytest.raises(FileNotFoundError, match="no checkpoint file"):
        load_native_checkpoint(tmp_path / "absent.npz", model)
    with pytest.raises(ValueError, match="not a regular file"):
        load_native_checkpoint(tmp_path, model)
    # A closed parameter or closed optimizer fails before anything.
    save_native_checkpoint(path, model)  # a valid file for load checks
    closed_optimizer = NativeAdam(model.parameters())
    closed_optimizer.close()
    with pytest.raises(RuntimeError, match="optimizer has been closed"):
        save_native_checkpoint(path, model, optimizer=closed_optimizer)
    with pytest.raises(RuntimeError, match="optimizer has been closed"):
        load_native_checkpoint(path, model, optimizer=closed_optimizer)
    model.bias.close()
    with pytest.raises(RuntimeError, match="'bias' has been closed"):
        save_native_checkpoint(path, model)
    with pytest.raises(RuntimeError, match="'bias' has been closed"):
        load_native_checkpoint(path, model)


# ======================================================================
# 2. Model-only save/load
# ======================================================================


@needs_native
def test_native_checkpoint_model_only_round_trip(tmp_path):
    source = _small_model(seed=0)
    _set_grads(source)
    saved_values = {
        name: parameter.to_numpy()
        for name, parameter in source.named_parameters()
    }
    path = tmp_path / "model.npz"
    assert save_native_checkpoint(path, source) is None
    # Saving touched nothing.
    assert [p.version for p in source.parameters()] == [0, 0]
    assert all(p.grad is not None for p in source.parameters())

    target = _small_model(seed=7)  # different init, same architecture
    _set_grads(target)
    grads_before = [p.grad for p in target.parameters()]
    identities = [id(p) for p in target.parameters()]
    metadata = load_native_checkpoint(path, target)
    assert metadata == {}  # None saved as the empty metadata dict
    for name, parameter in target.named_parameters():
        assert np.array_equal(parameter.to_numpy(), saved_values[name])
    # Identity, registration, gradient, and version contracts are the
    # module loader's own: same objects, grads preserved by identity,
    # one version increment per loaded parameter.
    assert [id(p) for p in target.parameters()] == identities
    assert [p.grad for p in target.parameters()] == grads_before
    assert [p.version for p in target.parameters()] == [1, 1]
    # The archive is independent of the source model: mutating the
    # source afterwards does not change what loads.
    source.weight.copy_value_(NativeTensor.from_array(np.zeros((2, 3))))
    third = _small_model(seed=3)
    load_native_checkpoint(path, third)
    assert np.array_equal(third.weight.to_numpy(), saved_values["weight"])


@needs_native
def test_native_checkpoint_model_load_makes_retained_graphs_stale(tmp_path):
    model = _small_model()
    path = tmp_path / "model.npz"
    save_native_checkpoint(path, model)
    x = NativeTensor.from_array(P_VALUES, requires_grad=True)
    out = x.matmul(model.weight).sum()
    out.backward(retain_graph=True)
    weight_grad = model.weight.grad
    load_native_checkpoint(path, model)  # same values — still +1 version
    assert model.weight.version == 1
    # The existing v3.7 contract: the retained value-sensitive graph is
    # stale after the model restore, gradients untouched.
    with pytest.raises(RuntimeError, match="stale"):
        out.backward(retain_graph=True)
    assert model.weight.grad is weight_grad


# ======================================================================
# 3 + 4. Optimizer checkpoints
# ======================================================================


@needs_native
def test_native_checkpoint_sgd_round_trip_and_next_step(tmp_path):
    model_a = _small_model(seed=0)
    optimizer_a = NativeSGD(model_a.parameters(), lr=0.5)
    _set_grads(model_a)
    path = tmp_path / "sgd.npz"
    save_native_checkpoint(path, model_a, optimizer=optimizer_a)
    # SGD contributes no tensor arrays — the archive holds exactly the
    # manifest and the two model entries.
    assert sorted(_arrays_of(path)) == [
        "manifest", "model::000000", "model::000001",
    ]
    model_b = _small_model(seed=9)
    optimizer_b = NativeSGD(model_b.parameters(), lr=0.001)
    load_native_checkpoint(path, model_b, optimizer=optimizer_b)
    assert optimizer_b.lr == 0.5
    # Optimizer restoration added no version increments beyond the
    # model load's own one-per-parameter.
    assert [p.version for p in model_b.parameters()] == [1, 1]
    # Identical next step: same values, same gradients, same lr.
    _set_grads(model_b)
    optimizer_a.step()
    optimizer_b.step()
    for parameter_a, parameter_b in zip(
        model_a.parameters(), model_b.parameters()
    ):
        assert np.array_equal(
            parameter_a.to_numpy(), parameter_b.to_numpy()
        )


@needs_native
def test_native_checkpoint_adam_restores_full_state(tmp_path):
    model_a = _mlp()
    optimizer_a = NativeAdam(
        model_a.parameters(), lr=0.05, betas=(0.8, 0.95), eps=1e-6
    )
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    _train(model_a, optimizer_a, x, y, 3)
    _set_grads(model_a)  # live gradients to survive the load
    path = tmp_path / "adam.npz"
    save_native_checkpoint(path, model_a, optimizer=optimizer_a)
    # Saving left the optimizer's internal moments open and usable.
    assert all(not buffer.closed
               for buffer in optimizer_a._m + optimizer_a._v)

    model_b = _mlp()
    optimizer_b = NativeAdam(model_b.parameters())  # default hyperparams
    grads_b_before = None
    load_native_checkpoint(path, model_b, optimizer=optimizer_b)
    assert optimizer_b.lr == 0.05
    assert optimizer_b.betas == (0.8, 0.95)
    assert optimizer_b.eps == 1e-6
    assert optimizer_b.step_counts == (3, 3, 3, 3)
    for index in range(4):
        for label in ("m", "v"):
            restored = getattr(optimizer_b, f"_{label}")[index]
            original = getattr(optimizer_a, f"_{label}")[index]
            assert np.array_equal(restored.to_numpy(), original.to_numpy())
            # Fresh optimizer-owned state, aliasing nothing live.
            assert restored is not original
            assert restored._core.storage is not original._core.storage
    # Optimizer loading changed no parameter versions (model load's +1
    # only) and the optimizer remains fully usable.
    assert [p.version for p in model_b.parameters()] == [1, 1, 1, 1]
    _set_grads(model_b)
    optimizer_b.step()
    assert optimizer_b.step_counts == (4, 4, 4, 4)
    # Gradient contract on the source side: saving preserved them.
    assert all(p.grad is not None for p in model_a.parameters())


# ======================================================================
# 5. Metadata
# ======================================================================


@needs_native
def test_native_checkpoint_metadata_round_trip_and_independence(tmp_path):
    model = _small_model()
    path = tmp_path / "meta.npz"
    metadata = {
        "epoch": 3,
        "loss": 0.25,
        "done": False,
        "note": "midway",
        "nothing": None,
        "nested": {"tags": ("a", "b"), "counts": [1, 2, 3]},
    }
    save_native_checkpoint(path, model, metadata=metadata)
    loaded = load_native_checkpoint(path, _small_model(seed=2))
    assert loaded == {
        "epoch": 3, "loss": 0.25, "done": False, "note": "midway",
        "nothing": None,
        "nested": {"tags": ["a", "b"], "counts": [1, 2, 3]},
    }  # tuples normalize to lists (the stable json.dumps convention)
    # The returned structure is independent: mutating it changes
    # nothing about a later load.
    loaded["nested"]["counts"].append(99)
    loaded["epoch"] = -1
    again = load_native_checkpoint(path, _small_model(seed=4))
    assert again["epoch"] == 3 and again["nested"]["counts"] == [1, 2, 3]


@needs_native
def test_native_checkpoint_metadata_validation(tmp_path):
    model = _small_model()
    path = tmp_path / "invalid-meta.npz"
    cyclic = {"self": None}
    cyclic["self"] = cyclic
    invalid = (
        "not a dict",
        {"k": float("nan")},
        {"k": float("inf")},
        {"k": b"bytes"},
        {"k": {1, 2}},
        {"k": Path("x")},
        {"k": np.float64(1.0)},   # NumPy scalars rejected by exact type
        {"k": np.int64(1)},
        {"k": np.zeros(2)},
        {"k": NativeTensor.from_array(P_VALUES)},
        {"k": NativeParameter(P_VALUES)},
        {"k": model},
        {1: "non-string key"},
        cyclic,
    )
    for bad in invalid:
        with pytest.raises((TypeError, ValueError)):
            save_native_checkpoint(path, model, metadata=bad)
    # Every rejection happened before any file was created.
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []  # no temporary residue either


# ======================================================================
# 6. Archive schema
# ======================================================================


@needs_native
def test_native_checkpoint_archive_schema_is_locked(tmp_path):
    model = _small_model()
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    _set_grads(model)
    optimizer.step()
    path = tmp_path / "schema.npz"
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata={"epoch": 1})
    arrays = _arrays_of(path)
    assert sorted(arrays) == [
        "manifest",
        "model::000000", "model::000001",
        "optimizer::m::000000", "optimizer::m::000001",
        "optimizer::v::000000", "optimizer::v::000001",
    ]
    # Every numerical entry is plain float64 — never an object array —
    # and the manifest is a 1-D uint8 UTF-8 JSON document.
    for name, array in arrays.items():
        if name == "manifest":
            assert array.dtype == np.uint8 and array.ndim == 1
        else:
            assert array.dtype == np.float64
    manifest = json.loads(arrays["manifest"].tobytes().decode("utf-8"))
    assert set(manifest) == {
        "format", "format_version", "model", "optimizer", "generators",
        "metadata",
    }
    assert manifest["format"] == "tensorforge.native_checkpoint"
    assert manifest["format_version"] == 3
    # A model with no registered generators writes an explicit null, so
    # absence is stated rather than inferred from a missing field (G5,
    # design §10.6).
    assert manifest["generators"] is None
    assert manifest["model"]["keys"] == ["weight", "bias"]
    assert set(manifest["model"]["entries"]) == {"weight", "bias"}
    weight_entry = manifest["model"]["entries"]["weight"]
    assert weight_entry == {
        "array": "model::000000", "shape": [2, 3],
        "dtype": "float64", "device": "cpu",
    }
    section = manifest["optimizer"]
    assert set(section) == {
        "type", "state_format_version", "lr", "betas", "eps",
        "parameters", "step_counts", "m", "v",
    }
    assert section["type"] == "NativeAdam"
    assert section["state_format_version"] == 1
    assert section["step_counts"] == [1, 1]
    # Version 3 carries each moment as an entry object of the same shape
    # as a model entry, rather than the bare archive name v1/v2 used, so a
    # moment's shape and dtype are stated rather than inferred positionally
    # from "parameters" (Phase I, milestone I8; design §16.2).
    assert section["m"] == [
        {"array": "optimizer::m::000000", "shape": [2, 3],
         "dtype": "float64", "device": "cpu"},
        {"array": "optimizer::m::000001", "shape": [3],
         "dtype": "float64", "device": "cpu"},
    ]
    assert section["v"] == [
        {"array": "optimizer::v::000000", "shape": [2, 3],
         "dtype": "float64", "device": "cpu"},
        {"array": "optimizer::v::000001", "shape": [3],
         "dtype": "float64", "device": "cpu"},
    ]
    assert manifest["metadata"] == {"epoch": 1}
    # Nothing volatile is serialized: no ids, pointers, reprs,
    # gradients, graphs, versions, or closed flags.
    manifest_text = json.dumps(manifest)
    for banned in ("id", "0x", "grad", "graph", "closed", "repr",
                   "'version'", "\"version\""):
        assert banned not in manifest_text, banned


# ======================================================================
# 7. Save cleanup and overwrite
# ======================================================================


@needs_native
def test_native_checkpoint_atomic_overwrite_and_failure_cleanup(
    tmp_path, monkeypatch
):
    first = _small_model(seed=0)
    second = _small_model(seed=5)
    path = tmp_path / "overwrite.npz"
    save_native_checkpoint(path, first, metadata={"which": "first"})
    save_native_checkpoint(path, second, metadata={"which": "second"})
    # The second save replaced the first atomically.
    assert _manifest_of(path)["metadata"] == {"which": "second"}
    loaded = _small_model(seed=8)
    load_native_checkpoint(path, loaded)
    assert np.array_equal(
        loaded.weight.to_numpy(), second.weight.to_numpy()
    )
    original_bytes = path.read_bytes()

    # A failing archive write leaves the existing destination intact,
    # removes the temporary file, and leaves the caller usable.
    optimizer = NativeAdam(first.parameters(), lr=0.1)

    def failing_savez(*args, **kwargs):
        raise OSError("forced archive-write failure")

    monkeypatch.setattr(np, "savez", failing_savez)
    with pytest.raises(OSError, match="forced archive-write failure"):
        save_native_checkpoint(path, first, optimizer=optimizer)
    monkeypatch.undo()
    assert path.read_bytes() == original_bytes
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "overwrite.npz",
    ]  # no temporary residue
    # The failed save closed its snapshots but touched nothing live.
    assert all(not buffer.closed for buffer in optimizer._m + optimizer._v)
    assert [p.version for p in first.parameters()] == [0, 0]
    save_native_checkpoint(path, first, optimizer=optimizer)  # recovers


@needs_native
def test_native_checkpoint_load_staging_failure_leaves_model_untouched(
    tmp_path, monkeypatch
):
    """A failure while staging a model tensor during load (phase 2, before
    the commit) must close every staged NativeTensor and leave the live
    model byte-for-byte unchanged, then recover on a valid load. Generic
    checkpoint infrastructure: the ``NativeTensor._typed_from_array`` staging seam
    is the same for any model, with or without buffers."""
    model = _mlp()
    _set_grads(model)
    path = tmp_path / "stage.npz"
    save_native_checkpoint(path, model)
    fingerprint = _model_fingerprint(model)

    real_from_array = NativeTensor._typed_from_array
    calls = {"n": 0}

    def failing_from_array(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:            # a later staged model tensor
            raise MemoryError("forced staging failure")
        return real_from_array(*args, **kwargs)

    monkeypatch.setattr(
        NativeTensor, "_typed_from_array", staticmethod(failing_from_array)
    )
    with pytest.raises(MemoryError):
        load_native_checkpoint(path, model)
    monkeypatch.undo()

    _assert_model_untouched(model, fingerprint)
    # Recovers completely on a valid load (versions +1 per parameter).
    load_native_checkpoint(path, model)
    assert [p.version for p in model.parameters()] == [1, 1, 1, 1]


@needs_native
def test_native_checkpoint_snapshot_failure_creates_nothing(tmp_path):
    # A model with a closed parameter fails inside its own state_dict
    # (which cleans up its partial snapshots) before any file exists.
    model = _small_model()
    model.bias.close()
    path = tmp_path / "never.npz"
    with pytest.raises(RuntimeError, match="closed"):
        save_native_checkpoint(path, model)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


# ======================================================================
# 8. Load corruption and atomicity
# ======================================================================


def _corrupt_cases(source, tmp_path):
    """(name, path) pairs, each a differently corrupted copy of the
    checkpoint at ``source`` (a model+Adam archive)."""
    cases = []

    def add(name, mutate_manifest=None, mutate_arrays=None):
        cases.append((name, _tamper(
            source, tmp_path / f"corrupt-{name}.npz",
            mutate_manifest, mutate_arrays,
        )))

    not_zip = tmp_path / "corrupt-not-a-zip.npz"
    not_zip.write_text("this is not an archive")
    cases.append(("not-a-zip", not_zip))

    no_manifest = tmp_path / "corrupt-no-manifest.npz"
    arrays = _arrays_of(source)
    del arrays["manifest"]
    _resave(no_manifest, arrays)
    cases.append(("missing-manifest", no_manifest))

    bad_repr = tmp_path / "corrupt-manifest-repr.npz"
    arrays = _arrays_of(source)
    arrays["manifest"] = np.zeros((2, 2))  # not 1-D uint8
    _resave(bad_repr, arrays)
    cases.append(("malformed-manifest-representation", bad_repr))

    bad_utf8 = tmp_path / "corrupt-utf8.npz"
    arrays = _arrays_of(source)
    arrays["manifest"] = np.frombuffer(b"\xff\xfe{", dtype=np.uint8)
    _resave(bad_utf8, arrays)
    cases.append(("invalid-utf8", bad_utf8))

    bad_json = tmp_path / "corrupt-json.npz"
    arrays = _arrays_of(source)
    arrays["manifest"] = np.frombuffer(b"{not json", dtype=np.uint8)
    _resave(bad_json, arrays)
    cases.append(("malformed-json", bad_json))

    bad_root = tmp_path / "corrupt-root.npz"
    arrays = _arrays_of(source)
    arrays["manifest"] = np.frombuffer(b"[1, 2]", dtype=np.uint8)
    _resave(bad_root, arrays)
    cases.append(("wrong-root-type", bad_root))

    add("wrong-format",
        lambda m: {**m, "format": "tensorforge.checkpoint"})
    add("unsupported-version", lambda m: {**m, "format_version": 4})
    add("version-zero", lambda m: {**m, "format_version": 0})
    add("missing-field",
        lambda m: {k: v for k, v in m.items() if k != "metadata"})
    add("unexpected-field", lambda m: {**m, "extra": 1})
    add("malformed-model-keys",
        lambda m: {**m, "model": {**m["model"], "keys": "weight"}})

    def rename_key(m):
        model = {"keys": ["weight", "renamed"], "entries": {}}
        model["entries"]["weight"] = m["model"]["entries"]["weight"]
        model["entries"]["renamed"] = m["model"]["entries"]["bias"]
        return {**m, "model": model}

    add("model-key-mismatch", rename_key)

    def duplicate_reference(m):
        entries = {k: dict(v) for k, v in m["model"]["entries"].items()}
        entries["bias"]["array"] = entries["weight"]["array"]
        entries["bias"]["shape"] = entries["weight"]["shape"]
        return {**m, "model": {"keys": m["model"]["keys"],
                               "entries": entries}}

    add("duplicate-array-reference", duplicate_reference)

    def missing_array(arrays):
        del arrays["optimizer::v::000001"]

    add("missing-tensor-array", mutate_arrays=missing_array)

    def extra_array(arrays):
        arrays["surprise"] = np.zeros(3)

    add("unexpected-tensor-array", mutate_arrays=extra_array)

    def wrong_dtype(arrays):
        arrays["model::000000"] = arrays["model::000000"].astype(np.float32)

    add("wrong-array-dtype", mutate_arrays=wrong_dtype)

    def wrong_shape(arrays):
        arrays["model::000000"] = arrays["model::000000"].reshape(3, 2)

    add("wrong-array-shape", mutate_arrays=wrong_shape)

    def object_array(arrays):
        arrays["model::000000"] = np.array(
            [{"hostile": True}], dtype=object
        )

    add("object-dtype-array", mutate_arrays=object_array)

    add("wrong-optimizer-type",
        lambda m: {**m, "optimizer": {
            "type": "NativeSGD",
            "state_format_version": 1,
            "lr": m["optimizer"]["lr"],
            "parameters": m["optimizer"]["parameters"],
        }})
    add("invalid-lr",
        lambda m: {**m, "optimizer": {**m["optimizer"], "lr": -1.0}})
    add("invalid-betas",
        lambda m: {**m, "optimizer": {**m["optimizer"],
                                      "betas": [0.9, 1.0]}})
    add("invalid-step-counts",
        lambda m: {**m, "optimizer": {**m["optimizer"],
                                      "step_counts": [-1, 1]}})
    add("malformed-optimizer-metadata",
        lambda m: {**m, "optimizer": {**m["optimizer"],
                                      "parameters": [{"shape": [9, 9],
                                                      "dtype": "float64",
                                                      "device": "cpu"}] * 2}})
    add("malformed-optimizer-section",
        lambda m: {**m, "optimizer": {**m["optimizer"], "m": "nope"}})
    return cases


@needs_native
def test_native_checkpoint_load_rejects_corruption_atomically(tmp_path):
    model = _small_model(seed=0)
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    _set_grads(model)
    optimizer.step()
    source = tmp_path / "good.npz"
    save_native_checkpoint(source, model, optimizer=optimizer)

    fingerprint = _model_fingerprint(model)
    moments_before = [
        (buffer, buffer.to_numpy())
        for buffer in optimizer._m + optimizer._v
    ]
    for name, path in _corrupt_cases(source, tmp_path):
        with pytest.raises((ValueError, TypeError)):
            load_native_checkpoint(path, model, optimizer=optimizer)
        # Every ordinary pre-commit failure left everything untouched.
        _assert_model_untouched(model, fingerprint)
        for (buffer, values), current in zip(
            moments_before, optimizer._m + optimizer._v
        ):
            assert current is buffer, name
            assert np.array_equal(current.to_numpy(), values), name
        assert optimizer.step_counts == (1, 1), name
        assert optimizer.lr == 0.1, name
    # The same model/optimizer recover completely on a valid load
    # (versions were 1 from the pre-save step; the model load adds one).
    metadata = load_native_checkpoint(source, model, optimizer=optimizer)
    assert metadata == {}
    assert [p.version for p in model.parameters()] == [2, 2]


@needs_native
def test_native_checkpoint_optimizer_presence_and_compatibility(tmp_path):
    model = _small_model(seed=0)
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    with_optimizer = tmp_path / "with-optimizer.npz"
    model_only = tmp_path / "model-only.npz"
    save_native_checkpoint(with_optimizer, model, optimizer=optimizer)
    save_native_checkpoint(model_only, model)
    fingerprint = _model_fingerprint(model)
    # Strict presence in both directions — before any model mutation.
    with pytest.raises(ValueError, match="no optimizer was supplied"):
        load_native_checkpoint(with_optimizer, model)
    with pytest.raises(ValueError, match="contains no optimizer state"):
        load_native_checkpoint(model_only, model, optimizer=optimizer)
    # Type mismatch: saved NativeAdam, supplied NativeSGD.
    sgd = NativeSGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="saved with optimizer 'NativeAdam'"):
        load_native_checkpoint(with_optimizer, model, optimizer=sgd)
    # An optimizer over unrelated parameters is rejected at save and
    # load — the live pair must share per-position identity.
    unrelated = NativeAdam(_small_model(seed=3).parameters())
    with pytest.raises(ValueError, match="not the model's parameter"):
        save_native_checkpoint(tmp_path / "x.npz", model,
                               optimizer=unrelated)
    with pytest.raises(ValueError, match="not the model's parameter"):
        load_native_checkpoint(with_optimizer, model, optimizer=unrelated)
    # A parameter-count mismatch is caught too.
    partial = NativeAdam([model.weight])
    with pytest.raises(ValueError, match="stores 1 unique parameters"):
        load_native_checkpoint(with_optimizer, model, optimizer=partial)
    _assert_model_untouched(model, fingerprint)


@needs_native
def test_native_checkpoint_shared_parameters_round_trip(tmp_path):
    shared = NativeParameter(P_VALUES)
    model = NativeModule()
    model.a = shared
    model.b = shared  # alias: one unique parameter, canonical key "a"
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    shared.multiply(NativeTensor.from_array(G_VALUES)).sum().backward()
    optimizer.step()
    path = tmp_path / "shared.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    manifest = _manifest_of(path)
    # The alias appears once everywhere.
    assert manifest["model"]["keys"] == ["a"]
    assert len(manifest["optimizer"]["parameters"]) == 1
    assert manifest["optimizer"]["step_counts"] == [1]

    fresh_shared = NativeParameter(np.zeros((2, 2)))
    fresh = NativeModule()
    fresh.a = fresh_shared
    fresh.b = fresh_shared
    fresh_optimizer = NativeAdam(fresh.parameters())
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert fresh.a is fresh_shared and fresh.b is fresh_shared
    assert np.array_equal(fresh_shared.to_numpy(), shared.to_numpy())
    assert fresh_shared.version == 1  # one shared object, one increment
    assert fresh_optimizer.step_counts == (1,)
    assert np.array_equal(
        fresh_optimizer._m[0].to_numpy(), optimizer._m[0].to_numpy()
    )


# ======================================================================
# 10. Deterministic file resume
# ======================================================================


@needs_native
def test_native_checkpoint_adam_file_resume_matches_uninterrupted(tmp_path):
    n_steps, m_steps = 6, 5
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    model_a = _mlp()
    optimizer_a = NativeAdam(model_a.parameters(), lr=0.05)
    _train(model_a, optimizer_a, x, y, n_steps)
    parameters_a = model_a.parameters()
    assert all(p.grad is None for p in parameters_a)  # boundary cleared

    path = tmp_path / "resume.npz"
    save_native_checkpoint(
        path, model_a, optimizer=optimizer_a,
        metadata={"steps": n_steps, "lr": 0.05},
    )

    model_b = _mlp()
    optimizer_b = NativeAdam(model_b.parameters())
    metadata = load_native_checkpoint(path, model_b, optimizer=optimizer_b)
    assert metadata == {"steps": 6, "lr": 0.05}
    parameters_b = model_b.parameters()
    baseline_b = [p.version for p in parameters_b]
    assert baseline_b == [1, 1, 1, 1]  # model load's own contract
    assert optimizer_b.step_counts == (n_steps,) * 4

    losses_a = _train(model_a, optimizer_a, x, y, m_steps)
    losses_b = _train(model_b, optimizer_b, x, y, m_steps)
    assert losses_a == losses_b  # bit-identical continuation
    for parameter_a, parameter_b in zip(parameters_a, parameters_b):
        assert np.array_equal(
            parameter_a.to_numpy(), parameter_b.to_numpy()
        )
        assert parameter_a.grad is None and parameter_b.grad is None
    for index in range(4):
        for label in ("m", "v"):
            assert np.array_equal(
                getattr(optimizer_a, f"_{label}")[index].to_numpy(),
                getattr(optimizer_b, f"_{label}")[index].to_numpy(),
            )
    assert optimizer_a.step_counts == optimizer_b.step_counts == (
        (n_steps + m_steps,) * 4
    )
    # Future version deltas match (absolute versions differ by design:
    # A never loaded, B's baseline includes the model load).
    assert [p.version for p in parameters_a] == [n_steps + m_steps] * 4
    assert [p.version for p in parameters_b] == [
        baseline + m_steps for baseline in baseline_b
    ]
    # Identities/registrations stayed stable, and the only file residue
    # is the checkpoint itself.
    assert [id(p) for p in model_b.parameters()] == [
        id(p) for p in parameters_b
    ]
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "resume.npz",
    ]
    optimizer_a.close()
    optimizer_b.close()


# ======================================================================
# 11. Guardrails
# ======================================================================


@needs_native
def test_native_checkpoint_uses_no_numpy_compute(monkeypatch, tmp_path):
    model = _small_model()
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    _set_grads(model)
    optimizer.step()
    path = tmp_path / "tripwire.npz"

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    # The serialization boundary itself (savez / load / frombuffer /
    # ascontiguousarray / to_numpy materialization) is allowed and
    # NumPy's own archive reader uses np.multiply.reduce internally —
    # so the tripwire covers the framework's numerical functions that
    # the boundary never needs.
    for name in ("sqrt", "reciprocal", "divide", "subtract",
                 "matmul", "mean", "negative", "power", "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata={"step": 1})
    metadata = load_native_checkpoint(path, model, optimizer=optimizer)
    monkeypatch.undo()
    assert metadata == {"step": 1}
    assert optimizer.step_counts == (1, 1)


@needs_native
def test_native_checkpoint_security_and_scope_guardrails(tmp_path):
    source = (
        Path(__file__).resolve().parent.parent
        / "src" / "tensorforge" / "experimental" / "native_checkpoint.py"
    ).read_text(encoding="utf-8")
    # Pickle stays disabled and absent; no dynamic code execution.
    # ("map_location" appears only in the docstring's not-supported
    # list, so it is checked at the signature level in the exports
    # test rather than banned as a substring here.)
    assert "allow_pickle=False" in source
    for banned in ("import pickle", "pickle.load", "pickle.dump",
                   "eval(", "exec(", "__reduce__"):
        assert banned not in source, banned
    # File APIs live only in the checkpoint module: the rest of the
    # experimental package performs no archive I/O.
    experimental_dir = (
        Path(__file__).resolve().parent.parent
        / "src" / "tensorforge" / "experimental"
    )
    for module_path in experimental_dir.glob("*.py"):
        if module_path.name == "native_checkpoint.py":
            continue
        text = module_path.read_text(encoding="utf-8")
        for banned in ("np.savez", "np.load", "import pickle"):
            assert banned not in text, f"{module_path.name}: {banned}"
    # A saved archive never contains pickled entries: every entry loads
    # with pickle disabled.
    model = _small_model()
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    path = tmp_path / "clean.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            assert archive[name].dtype != object
    # No scheduler/random-state capture and no checkpoint leakage into
    # the optimizers or module.
    for owner in (optimizer, model):
        for absent in ("save", "load", "save_checkpoint",
                       "load_checkpoint", "rng_state"):
            assert not hasattr(owner, absent)


# ======================================================================
# 12. The save side cannot emit an entry its own loader would reject
# ======================================================================
#
# A checkpoint-hardening repair, and its own regression. The compatibility
# proof written for Phase K milestone K5 showed that the *writer* trusted
# whatever dtype live state reported: every public construction path
# already refuses a non-floating parameter or buffer, so the trust held
# for every model a caller could build — but a serializer that can write
# a file its own reader rejects is a defect on its own terms, and the
# archive schema has to be enforced by the thing that emits it.
#
# The forged objects below are **test-only** and are never a supported
# way to use the framework. They exist to drive the writer with state the
# public API cannot produce, which is exactly what a schema guarantee has
# to survive. No production barrier is weakened to build them: the
# parameter constructor still rejects an ``int64`` tensor (asserted
# below), ``register_buffer`` still rejects one, and the buffer registry
# is corrupted directly rather than through it.


import contextlib  # noqa: E402
import gc  # noqa: E402
from collections import namedtuple  # noqa: E402

from tensorforge.experimental.native_module import (  # noqa: E402
    _BufferEntry,
)

_Destination = namedtuple("_Destination", ("exists", "size", "mtime_ns",
                                           "payload"))


def _index_tensor(values=(0, 1)):
    return NativeTensor.from_int64_array(np.asarray(values, dtype=np.int64))


def _step_once(model, optimizer):
    """One real forward/backward/step, so Adam's moments exist — with
    every tensor the test owns closed explicitly rather than dropped.
    The graph's own internal nodes stay framework-owned cycles, which is
    why the caller uses ``settle=True``."""
    features = NativeTensor.from_array(X_VALUES)
    try:
        output = model(features)
        try:
            total = output.sum()
            try:
                total.backward()
            finally:
                total.close()
        finally:
            output.close()
    finally:
        features.close()
    optimizer.step()
    optimizer.zero_grad()


@contextlib.contextmanager
def _forged_index_parameter(tensor):
    """A genuine ``NativeParameter`` instance carrying a real ``int64``
    core, built through ``__new__``.

    It has to be: ``NativeParameter(tensor)`` refuses an index tensor,
    which is the barrier this object exists to get behind. Every field is
    what ``__init__`` would have set except ``_owns_core=False`` — the
    core belongs to the caller's ``tensor``, which closes it — so this is
    never a second owner and closes nothing. Disarmed on exit, ``_closed``
    before the core reference is dropped, so the ``__del__`` fallback is
    already a no-op."""
    fake = NativeParameter.__new__(NativeParameter)
    fake._core = tensor._core
    fake._owns_core = False
    fake._closed = False
    fake._requires_grad = True
    fake._grad = None
    fake._parents = ()
    fake._backward = None
    fake._op = ""
    fake._is_leaf = True
    fake._graph_freed = False
    fake._expected_versions = ()
    fake._graph_resources = ()
    fake._version = 0
    assert type(fake) is NativeParameter
    assert fake.dtype == "int64"
    assert fake.owns_core is False
    try:
        yield fake
    finally:
        fake._closed = True
        fake._core = None
    assert tensor.closed is False


@contextlib.contextmanager
def _live_storage_baseline(settle=False):
    """Native live storage must return exactly to baseline.

    Installed outside ``monkeypatch`` on purpose: a mid-test ``undo()``
    must not be able to disarm the tracker that proves a rejection leaked
    nothing.

    ``settle`` is narrow and is never the proof that anything was
    released: an autograd graph's internal nodes are framework-owned and
    form reference cycles between a node and its backward closure, so a
    block that trained reclaims them rather than closing them. Every
    object a test owns is closed explicitly first, and
    ``test_native_checkpoint_live_storage_tracker_can_fail`` proves a
    retained, unclosed tensor is still reported after a collection."""
    live = {}
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def counting_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        live[id(self)] = self.dtype

    def counting_close(self):
        original_close(self)
        live.pop(id(self), None)

    cpp.NativeStorage.__init__ = counting_init
    cpp.NativeStorage.close = counting_close
    try:
        yield live
        if settle:
            gc.collect()
        assert not live, f"{len(live)} native storages were never closed"
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


def _destination(path):
    """Everything a save must not have changed about a destination."""
    if not os.path.exists(path):
        return _Destination(False, None, None, None)
    stat = os.stat(path)
    with open(path, "rb") as handle:
        payload = handle.read()
    return _Destination(True, stat.st_size, stat.st_mtime_ns, payload)


def _directory(path):
    return sorted(os.listdir(path))


def _assert_baseline_parameters_untouched(model, fingerprint):
    """``_assert_model_untouched`` compares every *current* parameter, and
    while a forged entry is registered the model deliberately holds one
    more than the fingerprint. The baseline entries are therefore compared
    by name; the forged one is not a baseline parameter and is asserted
    separately, as the operand the rejection must have left alone."""
    live = dict(model.named_parameters())
    for name, (values, version, grad) in fingerprint.items():
        parameter = live[name]
        assert np.array_equal(parameter.to_numpy(), values), name
        assert parameter.version == version, name
        assert parameter.grad is grad, name


def _assert_save_rejected(path, entry, model, optimizer, fingerprint,
                          directory, before, tensor):
    """One save-side dtype rejection, and everything it must have left
    alone: no new archive, an existing destination byte-for-byte intact,
    no collision-safe temporary left behind, the live model untouched,
    and the offending caller-owned tensor still open and unchanged."""
    values = tensor.tolist()
    with pytest.raises(ValueError) as error:
        save_native_checkpoint(path, model, optimizer=optimizer)
    message = str(error.value)
    assert "save_native_checkpoint" in message, message
    assert entry in message, message
    assert "int64" in message, message
    assert "floating" in message, message
    assert _destination(path) == before                 # byte-for-byte
    parent = os.path.dirname(path)
    assert _directory(parent) == directory              # no temporary left
    assert not [name for name in _directory(parent) if name.endswith(".tmp")]
    _assert_baseline_parameters_untouched(model, fingerprint)
    assert tensor.closed is False
    assert tensor.tolist() == values
    assert tensor.dtype == "int64"


@needs_native
def test_native_checkpoint_save_rejects_a_forged_index_parameter(tmp_path):
    """Both registration routes, driven for real, then saved.

    Registration itself carries no dtype authority — the parameter
    constructor does, and it is proved separately below — so the writer
    is what must refuse, and this asserts that it does at both routes,
    against a fresh destination and against an existing one."""
    with _live_storage_baseline():
        model = _small_model(seed=0)
        optimizer = NativeAdam(model.parameters(), lr=0.1)
        tensor = _index_tensor()
        try:
            existing = str(tmp_path / "existing.npz")
            save_native_checkpoint(existing, model, optimizer=optimizer)
            existing_before = _destination(existing)
            assert existing_before.exists
            fresh = str(tmp_path / "never-written.npz")
            baseline_names = [n for n, _ in model.named_parameters()]
            fingerprint = _model_fingerprint(model)

            with _forged_index_parameter(tensor) as fake:
                routes = (
                    ("assignment",
                     lambda: model.__setattr__("indices", fake)),
                    ("register_parameter",
                     lambda: model.register_parameter("indices", fake)),
                )
                for label, register in routes:
                    register()
                    # The registration really happened — otherwise the
                    # save below would be rejecting nothing.
                    assert model.indices is fake, label
                    assert "indices" in dict(model.named_parameters()), label
                    assert dict(model._state_named_tensors())["indices"] \
                        is fake, label

                    # A destination that does not exist stays absent...
                    _assert_save_rejected(
                        fresh, "'indices'", model, optimizer, fingerprint,
                        _directory(tmp_path), _destination(fresh), tensor)
                    assert not os.path.exists(fresh), label
                    # ...and one that does exist is untouched.
                    _assert_save_rejected(
                        existing, "'indices'", model, optimizer, fingerprint,
                        _directory(tmp_path), existing_before, tensor)

                    # Unregistering restores the exact baseline; ``del``
                    # is the undo that installs no ordinary attribute.
                    del model.indices
                    assert [n for n, _ in model.named_parameters()] == \
                        baseline_names, label
                    assert "indices" not in model.__dict__, label
                    assert not hasattr(model, "indices"), label

            # The control: with the forgery gone the same save succeeds
            # and the archive declares floating dtypes only.
            control = str(tmp_path / "control.npz")
            save_native_checkpoint(control, model, optimizer=optimizer)
            manifest = _manifest_of(control)
            declared = [entry["dtype"]
                        for entry in manifest["model"]["entries"].values()]
            declared += [entry["dtype"]
                         for entry in manifest["optimizer"]["parameters"]]
            for label in ("m", "v"):
                declared += [entry["dtype"]
                             for entry in manifest["optimizer"][label]]
            assert declared and set(declared) == {"float64"}
            load_native_checkpoint(control, model, optimizer=optimizer)
            assert _destination(existing) == existing_before
        finally:
            tensor.close()
            optimizer.close()
            for _, parameter in model.named_parameters():
                if parameter.grad is not None:
                    parameter.zero_grad()
                parameter.close()


@needs_native
def test_native_checkpoint_parameter_constructor_still_refuses_int64():
    """The separate, differently located barrier: construction.

    Kept apart from the save-side proof deliberately. The constructor is
    the supported authority for the parameter role and the reason no
    public route can reach the writer with an integer parameter at all;
    the writer's own check is what makes the archive schema true anyway."""
    with _live_storage_baseline():
        tensor = _index_tensor()
        try:
            with pytest.raises(ValueError, match="int64"):
                NativeParameter(tensor)
            with pytest.raises(ValueError, match="int64"):
                NativeParameter(tensor, dtype="int64")
            with pytest.raises(ValueError, match="int64"):
                NativeParameter(np.array([0, 1], dtype=np.int64),
                                dtype="int64")
            assert tensor.closed is False and tensor.tolist() == [0, 1]
            # ...and register_buffer, the other public state door.
            model = _small_model()
            try:
                for persistent in (True, False):
                    with pytest.raises(ValueError, match="int64"):
                        model.register_buffer("stat", tensor,
                                              persistent=persistent)
                assert list(model.named_buffers()) == []
            finally:
                for _, parameter in model.named_parameters():
                    parameter.close()
        finally:
            tensor.close()


@needs_native
def test_native_checkpoint_save_rejects_a_forged_persistent_buffer(tmp_path):
    """The buffer role, driven by corrupting the buffer **registry**
    directly rather than by weakening ``register_buffer``.

    ``_state_named_tensors()`` then presents an ``int64`` persistent
    buffer, which is precisely the state the writer must refuse."""
    with _live_storage_baseline():
        model = _small_model(seed=1)
        optimizer = NativeAdam(model.parameters(), lr=0.1)
        tensor = _index_tensor((3, 4, 5))
        floating = NativeTensor.from_array(np.array([1.0, 2.0]))
        try:
            model.register_buffer("stat", floating, persistent=True)
            existing = str(tmp_path / "buffers.npz")
            save_native_checkpoint(existing, model, optimizer=optimizer)
            before = _destination(existing)
            fingerprint = _model_fingerprint(model)

            # The narrow injection: one registry entry, replaced.
            model._buffers["stat"] = _BufferEntry(tensor, True)
            assert dict(model._state_named_tensors())["stat"] is tensor
            _assert_save_rejected(existing, "'stat'", model, optimizer,
                                  fingerprint, _directory(tmp_path), before,
                                  tensor)
            absent = str(tmp_path / "absent.npz")
            _assert_save_rejected(absent, "'stat'", model, optimizer,
                                  fingerprint, _directory(tmp_path),
                                  _destination(absent), tensor)
            assert not os.path.exists(absent)

            # A **non**-persistent forged buffer is not checkpoint state
            # at all, so the save succeeds and the archive omits it —
            # the negative control that keeps the claim about *persisted*
            # state rather than about every registered tensor.
            model._buffers["stat"] = _BufferEntry(tensor, False)
            transient = str(tmp_path / "transient.npz")
            save_native_checkpoint(transient, model, optimizer=optimizer)
            assert "stat" not in _manifest_of(transient)["model"]["entries"]

            # Restored, the same save succeeds and declares floating only.
            model._buffers["stat"] = _BufferEntry(floating, True)
            control = str(tmp_path / "restored.npz")
            save_native_checkpoint(control, model, optimizer=optimizer)
            entries = _manifest_of(control)["model"]["entries"]
            assert "stat" in entries
            assert {entry["dtype"] for entry in entries.values()} == \
                {"float64"}
            load_native_checkpoint(control, model, optimizer=optimizer)
            assert tensor.closed is False and tensor.tolist() == [3, 4, 5]
        finally:
            tensor.close()
            floating.close()
            optimizer.close()
            for _, parameter in model.named_parameters():
                if parameter.grad is not None:
                    parameter.zero_grad()
                parameter.close()


@needs_native
@pytest.mark.parametrize("role", ["parameters", "m", "v"])
def test_native_checkpoint_save_rejects_forged_optimizer_state(tmp_path, role):
    """The optimizer's three persisted roles, one at a time, injected
    around the real ``state_dict()`` authority.

    The injection is restored in ``finally``, proved to have fired, and
    every snapshot it created is proved closed after the rejection — by
    the writer's own ``finally``, not by the test and not by collection."""
    with _live_storage_baseline(settle=True):
        model = _small_model(seed=2)
        optimizer = NativeAdam(model.parameters(), lr=0.1)
        injected = []
        displaced = []
        fired = []
        try:
            _step_once(model, optimizer)
            existing = str(tmp_path / "optimizer.npz")
            save_native_checkpoint(existing, model, optimizer=optimizer)
            before = _destination(existing)
            fingerprint = _model_fingerprint(model)
            original = NativeAdam.state_dict

            def poisoned(self):
                state = original(self)
                fired.append(role)
                if role == "parameters":
                    state["parameters"] = [
                        {**entry, "dtype": "int64"}
                        for entry in state["parameters"]
                    ]
                else:
                    snapshot = _index_tensor((6, 7))
                    injected.append(snapshot)
                    # The displaced real snapshot leaves the list the
                    # writer closes, so the test closes it instead —
                    # explicitly, never by collection.
                    displaced.append(state[role][0])
                    state[role] = [snapshot] + list(state[role][1:])
                return state

            NativeAdam.state_dict = poisoned
            try:
                absent = str(tmp_path / "optimizer-absent.npz")
                for path, expected in (
                    (existing, before),
                    (absent, _Destination(False, None, None, None)),
                ):
                    with pytest.raises(ValueError) as error:
                        save_native_checkpoint(path, model,
                                               optimizer=optimizer)
                    message = str(error.value)
                    assert "save_native_checkpoint" in message, message
                    assert "optimizer state" in message, message
                    assert role in message, message
                    assert "int64" in message, message
                    assert "floating" in message, message
                    assert _destination(path) == expected
                    assert not [name for name in _directory(tmp_path)
                                if name.endswith(".tmp")]
                    _assert_baseline_parameters_untouched(model, fingerprint)
                assert not os.path.exists(absent)
            finally:
                NativeAdam.state_dict = original
            # The injection really ran, and everything it allocated was
            # released by the writer on the way out.
            assert fired, "the optimizer injection never fired"
            assert all(snapshot.closed for snapshot in injected), injected
            if role != "parameters":
                assert injected, "no snapshot was injected"

            # The control: the same optimizer saves and loads once the
            # injection is gone, declaring floating dtypes only.
            control = str(tmp_path / "optimizer-control.npz")
            save_native_checkpoint(control, model, optimizer=optimizer)
            manifest = _manifest_of(control)
            declared = [entry["dtype"]
                        for entry in manifest["optimizer"]["parameters"]]
            for label in ("m", "v"):
                declared += [entry["dtype"]
                             for entry in manifest["optimizer"][label]]
            assert declared and set(declared) == {"float64"}
            load_native_checkpoint(control, model, optimizer=optimizer)
            assert _destination(existing) == before
        finally:
            for snapshot in injected + displaced:
                if not snapshot.closed:
                    snapshot.close()
            optimizer.close()
            for _, parameter in model.named_parameters():
                if parameter.grad is not None:
                    parameter.zero_grad()
                parameter.close()


@needs_native
def test_native_checkpoint_live_storage_tracker_can_fail():
    """"Nothing leaked" means something only when a leak is detectable —
    and a collection must not be able to launder one."""
    retained = []
    with pytest.raises(AssertionError, match="never closed"):
        with _live_storage_baseline(settle=True):
            retained.append(NativeTensor.from_array(np.array([1.0, 2.0])))
    retained[0].close()
    with _live_storage_baseline():
        tensor = NativeTensor.from_array(np.array([1.0, 2.0]))
        tensor.close()


@needs_native
def test_native_checkpoint_save_seam_rejects_state_the_preflight_never_saw(
        tmp_path):
    """The two save-side checks answer different questions, so the second
    is not redundant: a ``state_dict()`` that returns an entry the live
    registries never held is still refused, at the serialization seam,
    before the manifest or any temporary file exists."""
    with _live_storage_baseline():
        model = _small_model(seed=3)
        injected = []
        try:
            path = str(tmp_path / "seam.npz")
            original = NativeModule.state_dict

            def poisoned(self):
                state = original(self)
                snapshot = _index_tensor((8, 9))
                injected.append(snapshot)
                state["sneaky"] = snapshot
                return state

            NativeModule.state_dict = poisoned
            try:
                with pytest.raises(ValueError) as error:
                    save_native_checkpoint(path, model)
                message = str(error.value)
                assert "'sneaky'" in message, message
                assert "int64" in message, message
            finally:
                NativeModule.state_dict = original
            assert injected, "the seam injection never fired"
            assert all(snapshot.closed for snapshot in injected)
            assert not os.path.exists(path)
            assert _directory(tmp_path) == []
            # The preflight passed — the live model was always valid —
            # which is what makes this the seam's own rejection.
            checkpoint_module._validate_model(model, "probe")
            save_native_checkpoint(path, model)
            assert os.path.exists(path)
        finally:
            for snapshot in injected:
                if not snapshot.closed:
                    snapshot.close()
            for _, parameter in model.named_parameters():
                parameter.close()


@needs_native
def test_native_checkpoint_save_and_load_ask_one_dtype_question():
    """The unit level: the writer's authority and the reader's rule 1 go
    through the same private question, so the two cannot drift apart and
    the writer cannot emit what the reader refuses."""
    where = "save_native_checkpoint()"
    for dtype in cpp.SUPPORTED_DTYPES:
        assert checkpoint_module._validated_persisted_dtype(
            dtype, "model state entry 'w'", where) == dtype
        assert checkpoint_module._canonical_persisted_dtype(dtype) == \
            (dtype, None)
    for rejected in ("int64", "int32", "uint8", "bool", "float16",
                     "complex64", "", "FLOAT64"):
        canonical, reason = checkpoint_module._canonical_persisted_dtype(
            rejected)
        assert canonical is None and reason is not None, rejected
        with pytest.raises(ValueError, match="floating"):
            checkpoint_module._validated_persisted_dtype(
                rejected, "model state entry 'w'", where)
    # ``normalize_dtype(None)`` means "the default", which is the right
    # answer to a different question — so a non-string is refused before
    # the authority is asked, on both sides.
    for bad in (None, 1, np.dtype("float64"), b"float64"):
        canonical, reason = checkpoint_module._canonical_persisted_dtype(bad)
        assert canonical is None and isinstance(reason, TypeError), bad
        with pytest.raises(ValueError):
            checkpoint_module._validated_persisted_dtype(bad, "e", where)
    # The reader's rule 1 asks the same question, and its own version rule
    # still stands on top of it, unchanged.
    for version in checkpoint_module._SUPPORTED_FORMAT_VERSIONS:
        with pytest.raises(ValueError, match="may declare"):
            checkpoint_module._validated_entry_dtype("int64", version, "e",
                                                     "load")
    assert checkpoint_module._validated_entry_dtype("float32", 3, "e",
                                                    "load") == "float32"
    for version in checkpoint_module._FLOAT64_ONLY_VERSIONS:
        with pytest.raises(ValueError, match="float64 only"):
            checkpoint_module._validated_entry_dtype("float32", version, "e",
                                                     "load")
