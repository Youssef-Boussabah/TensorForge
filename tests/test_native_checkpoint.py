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
        "format", "format_version", "model", "optimizer", "metadata",
    }
    assert manifest["format"] == "tensorforge.native_checkpoint"
    assert manifest["format_version"] == 1
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
    assert section["m"] == ["optimizer::m::000000", "optimizer::m::000001"]
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
    checkpoint infrastructure: the ``NativeTensor.from_array`` staging seam
    is the same for any model, with or without buffers."""
    model = _mlp()
    _set_grads(model)
    path = tmp_path / "stage.npz"
    save_native_checkpoint(path, model)
    fingerprint = _model_fingerprint(model)

    real_from_array = NativeTensor.from_array
    calls = {"n": 0}

    def failing_from_array(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:            # a later staged model tensor
            raise MemoryError("forced staging failure")
        return real_from_array(*args, **kwargs)

    monkeypatch.setattr(
        NativeTensor, "from_array", staticmethod(failing_from_array)
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
    add("wrong-version", lambda m: {**m, "format_version": 2})
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
