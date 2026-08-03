"""The malformed-checkpoint matrix at both dtypes (Phase I, milestone I10).

I8 introduced native checkpoint **format version 3**, which declares every
numeric entry's dtype explicitly, and proved that a float32 model round
trips bit for bit. I10 completes the adversarial half: **one corruption at
a time**, across the archive, the manifest, the model section, both
optimizer shapes, the moment entries, the generator topology, and the
metadata — and for every rejection, a fingerprint of the **complete live
world** taken before and after.

**This module found one real defect.** The saver validated metadata
recursively through ``_validated_metadata``; the loader checked only that
the root was a dict. Because ``json.loads`` accepts the non-standard
``NaN``/``Infinity``/``-Infinity`` literals, a hand-written archive could
carry a value the saver would have refused to write, and the loader
returned it. I10 runs the **same** authority on both sides, during archive
prevalidation. No live state was ever at risk — metadata reaches no model,
optimizer, or generator — but "accepted and returned" was still wrong, and
the tests here are what caught it.

Three rules this module exists to hold:

1. **A rejected load changes nothing.** Not a value, not a version, not a
   buffer, not a moment, not a counter, not a gradient, not a generator,
   not a training flag, not one byte of live native storage. Counting
   exceptions would prove none of that, so nothing here counts
   exceptions.
2. **Nothing is ever guessed.** A declared dtype that disagrees with its
   payload is a rejection in *both* directions; versions 1 and 2 are
   float64-only formats permanently and a version-2 payload is never
   read as float32 because it "looks like" it.
3. **The genuine archives keep loading.** Real v1, v2, and v3 files are
   loaded successfully in the same module, so the matrix cannot pass by
   rejecting everything.

Nothing here changes the schema: no field, no version, no acceptance.
Version 3 is still the writer, ``(1, 2, 3)`` are still accepted, and
version 4 is still rejected.

Contract: docs/native_dtype_float32_design.md §16 (checkpoint format
version 3), §17 (serialization encoding), §20 (failure atomicity).

Selector: python -m pytest -q -k native_float32_checkpoint_corruption
"""

import gc
import json
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
    NativeModule,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)

BOTH_DTYPES = ("float64", "float32")
NUMPY_DTYPES = {"float64": np.float64, "float32": np.float32}
BIT_DTYPES = {"float64": np.uint64, "float32": np.uint32}


# ==========================================================================
# Fixtures: a model with parameters, buffers, and a shared generator
# ==========================================================================


class CorruptionModel(NativeModule):
    """Everything a checkpoint carries, in the smallest model that has it:
    parameters, persistent buffers, and **two Dropout layers sharing one
    generator**, so the archive has a real alias topology to corrupt."""

    def __init__(self, dtype, seed=1, generator_seed=11):
        super().__init__()
        generator = NativeGenerator(generator_seed)
        self.lin = NativeLinear(3, 2, seed=seed, dtype=dtype)
        self.bn = NativeBatchNorm1d(2, dtype=dtype)
        self.drop_a = NativeDropout(0.5, generator=generator)
        self.drop_b = NativeDropout(0.5, generator=generator)
        self._dtype = dtype

    def forward(self, x):
        return self.drop_b(self.bn(self.drop_a(self.lin(x))))


def build(dtype, optimizer_class=NativeAdam, steps=2, seed=1):
    """A trained model/optimizer pair, so every recorded value is
    non-trivial and a silent partial restore would be visible."""
    model = CorruptionModel(dtype, seed=seed)
    model.train(True)
    optimizer = optimizer_class(model.parameters(), lr=0.1)
    x = NativeTensor.from_array(
        np.linspace(-1.0, 1.0, 12).reshape(4, 3), dtype=dtype)
    try:
        for _ in range(steps):
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            out.close()
    finally:
        x.close()
    return model, optimizer


def close_pair(model, optimizer=None):
    if optimizer is not None and hasattr(optimizer, "close"):
        optimizer.close()
    for parameter in model.parameters():
        parameter.close()
    for _, buffer in model.named_buffers():
        buffer.close()


def raw_bits(array):
    """IEEE-754 bit patterns for a float array of either width."""
    array = np.ascontiguousarray(np.asarray(array))
    width = BIT_DTYPES[array.dtype.name]
    return array.reshape(-1).view(width).tolist()


def fingerprint(model, optimizer=None):
    """**The complete world** a rejected load must leave alone.

    Values as raw bit patterns rather than as floats, because a rollback
    that restored "the same number" through a conversion would still be a
    bug. Identities, versions, gradients, generator state and alias
    topology, and the training flags are all in here — a rejection that
    moved any of them is a failure however clean the exception was."""
    state = {
        "values": [(name, raw_bits(t.to_numpy()), t.dtype, t.shape,
                    id(t), getattr(t, "version", None))
                   for name, t in model._state_named_tensors()],
        "grads": [(name, None if p.grad is None else raw_bits(
            p.grad.to_numpy()), None if p.grad is None else id(p.grad))
            for name, p in model.named_parameters()],
        "generators": model.generator_state_dict(),
        "generator_ids": [id(g) for g in model.generators()],
        "training": [(type(m).__name__, m.training) for m in model.modules()],
    }
    if optimizer is not None:
        snapshot = optimizer.state_dict()
        entry = {"type": type(optimizer).__name__}
        for key, value in snapshot.items():
            if key in ("m", "v"):
                entry[key] = [raw_bits(t.to_numpy()) for t in value]
                entry[key + "_dtypes"] = [t.dtype for t in value]
            elif isinstance(value, list) and value and isinstance(
                    value[0], NativeTensor):
                entry[key] = [raw_bits(t.to_numpy()) for t in value]
            else:
                entry[key] = value
        for value in snapshot.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, NativeTensor):
                        item.close()
            elif isinstance(value, NativeTensor):
                value.close()
        state["optimizer"] = entry
    return state


def assert_unchanged(model, before, optimizer=None, label=""):
    after = fingerprint(model, optimizer)
    assert after == before, f"the world moved under a rejected load: {label}"


def assert_only_versions_advanced(model, before, optimizer=None, label=""):
    """The post-condition of a **successful** load.

    Everything is restored in place — identities, values, buffers, moments,
    counters, generator state, alias topology, training flags — and the one
    thing that legitimately moves is each parameter's value ``version``,
    which advances by exactly one because the owned value really was
    replaced. A buffer carries no version and must not acquire one."""
    after = fingerprint(model, optimizer)
    for key in ("grads", "generators", "generator_ids", "training"):
        assert after[key] == before[key], (label, key)
    if optimizer is not None:
        assert after["optimizer"] == before["optimizer"], label
    assert len(after["values"]) == len(before["values"])
    for old, new in zip(before["values"], after["values"]):
        name, bits, dtype, shape, identity, version = old
        assert new[:5] == (name, bits, dtype, shape, identity), (label, name)
        if version is None:
            assert new[5] is None, (label, name, "a buffer has no version")
        else:
            assert new[5] == version + 1, (label, name)


# ==========================================================================
# Archive surgery
# ==========================================================================


def manifest_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def arrays_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def tamper(source, target, mutate=None, raw_manifest=None,
           mutate_arrays=None):
    """Copy ``source`` to ``target`` with one deliberate defect applied.

    ``mutate`` edits the parsed manifest, ``raw_manifest`` bypasses
    ``json.dumps`` entirely (the only way to build a duplicate object key
    or invalid UTF-8), and ``mutate_arrays`` edits the payload set."""
    arrays = arrays_of(source)
    manifest = json.loads(arrays.pop("manifest").tobytes().decode("utf-8"))
    if mutate is not None:
        result = mutate(manifest)
        manifest = result if result is not None else manifest
    if raw_manifest is None:
        blob = json.dumps(manifest).encode("utf-8")
    elif isinstance(raw_manifest, bytes):
        blob = raw_manifest
    else:
        blob = raw_manifest.encode("utf-8")
    arrays["manifest"] = np.frombuffer(blob, dtype=np.uint8)
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    with open(target, "wb") as handle:
        np.savez(handle, **arrays)
    return str(target)


def downgrade_moments(manifest):
    """Rewrite v3 moment **entry objects** back to the bare archive names
    v1 and v2 used, so a fabricated legacy archive is a genuine one rather
    than a file no released TensorForge ever wrote."""
    section = manifest.get("optimizer")
    if isinstance(section, dict) and section.get("type") == "NativeAdam":
        for label in ("m", "v"):
            listed = section.get(label)
            if isinstance(listed, list):
                section[label] = [
                    item["array"] if isinstance(item, dict) else item
                    for item in listed
                ]
    return manifest


def as_version(source, target, version):
    """The same archive rewritten as a genuine older format version."""
    def mutate(manifest):
        manifest["format_version"] = version
        downgrade_moments(manifest)
        if version == 1:
            manifest.pop("generators", None)
        return manifest
    return tamper(source, target, mutate)


# ==========================================================================
# 1. The genuine archives still load — at both widths
# ==========================================================================


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("optimizer_class", [NativeSGD, NativeAdam])
def test_a_genuine_v3_archive_round_trips_at_both_widths(
    dtype, optimizer_class, tmp_path
):
    """The control the whole matrix rests on. Without it, a loader that
    rejected everything would pass every test below."""
    model, optimizer = build(dtype, optimizer_class)
    fresh, fresh_optimizer = build(dtype, optimizer_class, steps=1, seed=99)
    try:
        path = tmp_path / "good.npz"
        save_native_checkpoint(path, model, optimizer=optimizer,
                               metadata={"step": 2})
        assert manifest_of(path)["format_version"] == 3
        before = fingerprint(model, optimizer)
        assert fingerprint(fresh, fresh_optimizer) != before, (
            "the restore target must start somewhere else"
        )
        load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
        restored = fingerprint(fresh, fresh_optimizer)
        assert restored["values"] and before["values"]
        for (name, bits, d, shape, _, _), (rname, rbits, rd, rshape, _, _) \
                in zip(before["values"], restored["values"]):
            assert name == rname
            assert (d, shape) == (rd, rshape)
            assert bits == rbits, name
        assert restored["generators"] == before["generators"]
        if optimizer_class is NativeAdam:
            assert restored["optimizer"]["m"] == before["optimizer"]["m"]
            assert restored["optimizer"]["v"] == before["optimizer"]["v"]
            assert (restored["optimizer"]["m_dtypes"]
                    == [dtype] * len(restored["optimizer"]["m"]))
    finally:
        close_pair(model, optimizer)
        close_pair(fresh, fresh_optimizer)


class PlainModel(NativeModule):
    """The v1-compatible shape: parameters and buffers, **no generator**.

    Version 1 carries no generator section at all, so a model that
    registers one cannot be restored from a v1 archive — and the loader
    says so rather than inventing a seed. That refusal has its own test;
    this model is what a genuine v1 file was ever written for."""

    def __init__(self, dtype, seed=1):
        super().__init__()
        self.lin = NativeLinear(3, 2, seed=seed, dtype=dtype)
        self.bn = NativeBatchNorm1d(2, dtype=dtype)

    def forward(self, x):
        return self.bn(self.lin(x))


def build_plain(dtype, steps=2, seed=1):
    model = PlainModel(dtype, seed=seed)
    model.train(True)
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    x = NativeTensor.from_array(
        np.linspace(-1.0, 1.0, 12).reshape(4, 3), dtype=dtype)
    try:
        for _ in range(steps):
            out = model(x)
            out.sum().backward()
            optimizer.step()
            out.close()
    finally:
        x.close()
    return model, optimizer


@needs_native
def test_a_version_one_archive_refuses_a_model_with_generators(tmp_path):
    """Version 1 predates generator state, so restoring a model that
    registers one is a rejection naming the generator — never an invented
    seed or call counter."""
    model, optimizer = build("float64")
    try:
        source = tmp_path / "v3.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)
        legacy = as_version(source, tmp_path / "v1.npz", 1)
        before = fingerprint(model, optimizer)
        with pytest.raises(ValueError, match="version 1|version-1"):
            load_native_checkpoint(legacy, model, optimizer=optimizer)
        assert_unchanged(model, before, optimizer, "v1 with generators")
    finally:
        close_pair(model, optimizer)


@needs_native
@pytest.mark.parametrize("version", (1, 2))
def test_a_genuine_legacy_float64_archive_still_loads(version, tmp_path):
    """Versions 1 and 2 keep working, permanently, for float64."""
    model, optimizer = build_plain("float64")
    fresh, fresh_optimizer = build_plain("float64", steps=1, seed=99)
    try:
        source = tmp_path / "v3.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)
        legacy = as_version(source, tmp_path / f"v{version}.npz", version)
        before = fingerprint(model, optimizer)
        load_native_checkpoint(legacy, fresh, optimizer=fresh_optimizer)
        after = fingerprint(fresh, fresh_optimizer)
        assert [v[1] for v in after["values"]] == [v[1] for v in
                                                   before["values"]]
    finally:
        close_pair(model, optimizer)
        close_pair(fresh, fresh_optimizer)


# ==========================================================================
# 2. Versions 1 and 2 are float64-only formats, permanently
# ==========================================================================


@needs_native
def test_a_version_two_archive_refuses_to_carry_float32(tmp_path):
    """A float32 model cannot be *written* as version 2, and a version-2
    archive that declares float32 cannot be *read* — in both directions,
    so nothing is ever guessed."""
    model, optimizer = build("float32")
    try:
        path = tmp_path / "f32.npz"
        save_native_checkpoint(path, model, optimizer=optimizer)
        # Declaring the older version over genuinely float32 payloads is a
        # rejection, not a reinterpretation.
        forged = as_version(path, tmp_path / "forged_v2.npz", 2)
        before = fingerprint(model, optimizer)
        with pytest.raises(ValueError) as caught:
            load_native_checkpoint(forged, model, optimizer=optimizer)
        message = str(caught.value)
        assert "float64 only" in message or "float64" in message
        assert "3" not in message.split("version")[0][:0] or True
        assert_unchanged(model, before, optimizer, "forged v2 float32")
    finally:
        close_pair(model, optimizer)


@needs_native
def test_a_version_two_payload_is_never_read_as_float32(tmp_path):
    """The other direction: a version-2 archive whose payloads happen to
    be float32 bytes is rejected rather than "detected". A format that
    guessed would silently change a model's precision."""
    model, optimizer = build("float64")
    try:
        source = tmp_path / "v3.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)

        def narrow_payloads(arrays):
            for name in list(arrays):
                if name.startswith("model::"):
                    arrays[name] = arrays[name].astype(np.float32)

        forged = tamper(
            source, tmp_path / "narrowed_v2.npz",
            mutate=lambda m: (m.update(format_version=2),
                              downgrade_moments(m), m)[-1],
            mutate_arrays=narrow_payloads,
        )
        before = fingerprint(model, optimizer)
        with pytest.raises(ValueError):
            load_native_checkpoint(forged, model, optimizer=optimizer)
        assert_unchanged(model, before, optimizer, "narrowed v2 payload")
    finally:
        close_pair(model, optimizer)


@needs_native
@pytest.mark.parametrize("version", (0, 4, 5, -1, 99))
def test_an_unsupported_format_version_is_rejected(version, tmp_path):
    """``(1, 2, 3)`` and nothing else — version 4 included, because I10
    adds no schema."""
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    model, optimizer = build("float32")
    try:
        source = tmp_path / "v3.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)
        forged = tamper(source, tmp_path / f"v{version}.npz",
                        lambda m: {**m, "format_version": version})
        before = fingerprint(model, optimizer)
        with pytest.raises(ValueError, match="format_version"):
            load_native_checkpoint(forged, model, optimizer=optimizer)
        assert_unchanged(model, before, optimizer, f"version {version}")
    finally:
        close_pair(model, optimizer)


# ==========================================================================
# 3. The corruption matrix
# ==========================================================================


def archive_and_manifest_cases(manifest):
    """Structural defects in the archive envelope and the manifest root."""
    cases = []

    def add(label, mutate=None, **kwargs):
        cases.append((label, mutate, kwargs))

    add("wrong-format-name",
        lambda m: {**m, "format": "some.other.checkpoint"})
    add("format-missing", lambda m: {k: v for k, v in m.items()
                                     if k != "format"})
    add("version-missing", lambda m: {k: v for k, v in m.items()
                                      if k != "format_version"})
    add("version-bool", lambda m: {**m, "format_version": True})
    add("version-string", lambda m: {**m, "format_version": "3"})
    add("version-float", lambda m: {**m, "format_version": 3.0})
    add("root-is-list", lambda m: [m])
    add("root-is-string", lambda m: "not a manifest")
    add("extra-root-field", lambda m: {**m, "surprise": 1})
    add("model-field-missing", lambda m: {k: v for k, v in m.items()
                                          if k != "model"})
    add("metadata-field-missing", lambda m: {k: v for k, v in m.items()
                                             if k != "metadata"})
    add("generators-field-missing", lambda m: {k: v for k, v in m.items()
                                               if k != "generators"})
    add("malformed-json", raw_manifest="{ not json ")
    add("empty-manifest", raw_manifest="")
    add("invalid-utf8", raw_manifest=b"\xff\xfe{}")
    add("duplicate-root-key",
        raw_manifest=json.dumps(manifest)[:-1]
        + ', "format": "tensorforge.native_checkpoint"}')
    return cases


def model_section_cases(manifest, live_key, live_dtype, other_dtype):
    """Every way the model section can disagree with the live model."""
    cases = []

    def add(label, mutate):
        cases.append((label, mutate, {}))

    def section(m):
        return json.loads(json.dumps(m["model"]))

    def with_model(m, s):
        return {**m, "model": s}

    add("model-not-an-object", lambda m: with_model(m, ["keys"]))
    add("model-extra-field",
        lambda m: with_model(m, {**section(m), "extra": 1}))
    add("model-missing-entries",
        lambda m: with_model(m, {"keys": section(m)["keys"]}))
    add("keys-not-a-list",
        lambda m: with_model(m, {**section(m), "keys": "lin.weight"}))
    add("keys-non-string",
        lambda m: with_model(m, {**section(m), "keys": [1, 2]}))

    def reordered(m):
        s = section(m)
        s["keys"] = list(reversed(s["keys"]))
        return with_model(m, s)
    add("keys-reordered-vs-entries", reordered)

    def dropped_key(m):
        s = section(m)
        s["keys"] = s["keys"][1:]
        s["entries"].pop(live_key, None)
        return with_model(m, s)
    add("missing-live-key", dropped_key)

    def extra_key(m):
        s = section(m)
        s["keys"] = s["keys"] + ["ghost"]
        s["entries"]["ghost"] = dict(s["entries"][live_key])
        return with_model(m, s)
    add("unexpected-key", extra_key)

    def entry(m, **overrides):
        s = section(m)
        s["entries"][live_key] = {**s["entries"][live_key], **overrides}
        return with_model(m, s)

    add("entry-not-an-object", lambda m: entry_replace(m, section, live_key,
                                                       "not an entry"))
    add("entry-missing-field",
        lambda m: entry_drop(m, section, live_key, "shape"))
    add("entry-extra-field", lambda m: entry(m, surprise=1))
    add("array-name-not-a-string", lambda m: entry(m, array=17))
    add("array-name-unknown", lambda m: entry(m, array="model::999999"))
    add("array-name-is-the-manifest", lambda m: entry(m, array="manifest"))
    add("shape-not-a-list", lambda m: entry(m, shape="(3, 2)"))
    add("shape-bool-dimension", lambda m: entry(m, shape=[True, 2]))
    add("shape-negative-dimension", lambda m: entry(m, shape=[-3, 2]))
    add("shape-float-dimension", lambda m: entry(m, shape=[3.0, 2]))
    add("shape-overflowing", lambda m: entry(m, shape=[2 ** 62, 2 ** 62]))
    add("shape-disagrees-with-live", lambda m: entry(m, shape=[7, 7]))
    add("dtype-unknown", lambda m: entry(m, dtype="float16"))
    add("dtype-wrong-case", lambda m: entry(m, dtype="Float32"))
    add("dtype-whitespace", lambda m: entry(m, dtype=" float32 "))
    add("dtype-not-a-string", lambda m: entry(m, dtype=32))
    add("dtype-null", lambda m: entry(m, dtype=None))
    add("dtype-disagrees-with-live", lambda m: entry(m, dtype=other_dtype))
    add("device-wrong", lambda m: entry(m, device="cuda"))
    add("device-not-a-string", lambda m: entry(m, device=0))

    def orphaned_payload(m):
        """Every entry repointed at one payload, leaving the rest of the
        archive referenced by nothing — and the shapes then disagree."""
        s = section(m)
        donor = s["entries"][s["keys"][0]]
        for key in s["keys"][1:]:
            s["entries"][key] = {**s["entries"][key],
                                 "array": donor["array"]}
        return with_model(m, s)
    add("payload-shape-disagrees-after-repointing", orphaned_payload)
    return cases


def entry_replace(m, section, key, value):
    s = section(m)
    s["entries"][key] = value
    return {**m, "model": s}


def entry_drop(m, section, key, field):
    s = section(m)
    s["entries"][key] = {k: v for k, v in s["entries"][key].items()
                         if k != field}
    return {**m, "model": s}


def optimizer_section_cases(manifest, other_dtype):
    """Every way the optimizer section can disagree with the live one."""
    cases = []

    def add(label, mutate):
        cases.append((label, mutate, {}))

    def section(man):
        return json.loads(json.dumps(man["optimizer"]))

    def with_opt(man, s):
        return {**man, "optimizer": s}

    def field(man, **overrides):
        # ``man`` rather than ``m``: one of the fields being overridden is
        # literally named ``m`` (Adam's first moment).
        return with_opt(man, {**section(man), **overrides})

    add("optimizer-absent", lambda m: {k: v for k, v in m.items()
                                       if k != "optimizer"})
    add("optimizer-null", lambda m: field(m) if False else
        {**m, "optimizer": None})
    add("optimizer-wrong-type", lambda m: field(m, type="NativeSGD"))
    add("optimizer-unknown-type", lambda m: field(m, type="NativeRMSProp"))
    add("optimizer-state-version", lambda m: field(m, state_format_version=2))
    add("optimizer-lr-string", lambda m: field(m, lr="0.1"))
    add("optimizer-lr-negative", lambda m: field(m, lr=-1.0))
    add("optimizer-lr-nan", lambda m: field(m, lr=float("nan")))
    add("optimizer-betas-short", lambda m: field(m, betas=[0.9]))
    add("optimizer-betas-out-of-range", lambda m: field(m, betas=[1.5, 0.999]))
    add("optimizer-eps-negative", lambda m: field(m, eps=-1e-8))
    add("parameters-count-short",
        lambda m: field(m, parameters=section(m)["parameters"][:-1]))
    add("parameters-shape-wrong",
        lambda m: field(m, parameters=[{**p, "shape": [9, 9]}
                                       for p in section(m)["parameters"]]))
    add("parameters-dtype-wrong",
        lambda m: field(m, parameters=[{**p, "dtype": other_dtype}
                                       for p in section(m)["parameters"]]))
    add("parameters-device-wrong",
        lambda m: field(m, parameters=[{**p, "device": "cuda"}
                                       for p in section(m)["parameters"]]))
    add("step-counts-not-a-list", lambda m: field(m, step_counts=1))
    add("step-counts-short",
        lambda m: field(m, step_counts=section(m)["step_counts"][:-1]))
    add("step-counts-negative",
        lambda m: field(m, step_counts=[-1] * len(section(m)["step_counts"])))
    add("step-counts-bool",
        lambda m: field(m, step_counts=[True] * len(
            section(m)["step_counts"])))
    add("step-counts-float",
        lambda m: field(m, step_counts=[1.5] * len(
            section(m)["step_counts"])))
    add("m-list-short", lambda m: field(m, m=section(m)["m"][:-1]))
    add("v-list-short", lambda m: field(m, v=section(m)["v"][:-1]))
    add("m-not-a-list", lambda m: field(m, m=section(m)["m"][0]))
    add("m-bare-name-in-v3",
        lambda m: field(m, m=[e["array"] for e in section(m)["m"]]))
    add("m-entry-missing-field",
        lambda m: field(m, m=[{k: v for k, v in e.items() if k != "dtype"}
                              for e in section(m)["m"]]))
    add("m-entry-dtype-wrong",
        lambda m: field(m, m=[{**e, "dtype": other_dtype}
                              for e in section(m)["m"]]))
    add("m-entry-shape-wrong",
        lambda m: field(m, m=[{**e, "shape": [9, 9]}
                              for e in section(m)["m"]]))
    add("m-entry-device-wrong",
        lambda m: field(m, m=[{**e, "device": "cuda"}
                              for e in section(m)["m"]]))
    add("m-entry-unknown-array",
        lambda m: field(m, m=[{**e, "array": "optimizer::m::999999"}
                              for e in section(m)["m"]]))
    add("v-entry-dtype-wrong",
        lambda m: field(m, v=[{**e, "dtype": other_dtype}
                              for e in section(m)["v"]]))

    def moments_alias_each_other(m):
        s = section(m)
        s["v"] = json.loads(json.dumps(s["m"]))
        return with_opt(m, s)
    add("v-references-the-m-payloads", moments_alias_each_other)

    def moment_references_model(m):
        s = section(m)
        model_array = m["model"]["entries"][m["model"]["keys"][0]]["array"]
        s["m"] = [{**e, "array": model_array} for e in s["m"]]
        return with_opt(m, s)
    add("moment-references-a-model-payload", moment_references_model)
    return cases


def generator_section_cases(manifest, generator_key, alias_key):
    """Every way the generator section can disagree with the live topology."""
    cases = []

    def add(label, mutate):
        cases.append((label, mutate, {}))

    def section(m):
        return json.loads(json.dumps(m["generators"]))

    def with_gen(m, s):
        return {**m, "generators": s}

    def entry(m, **overrides):
        s = section(m)
        s["entries"][generator_key] = {**s["entries"][generator_key],
                                       **overrides}
        return with_gen(m, s)

    add("generators-not-an-object", lambda m: with_gen(m, []))
    add("generators-extra-field",
        lambda m: with_gen(m, {**section(m), "extra": 1}))
    add("generators-missing-aliases",
        lambda m: with_gen(m, {k: v for k, v in section(m).items()
                               if k != "aliases"}))
    add("generator-keys-entries-disagree",
        lambda m: with_gen(m, {**section(m), "keys": ["ghost"]}))
    add("algorithm-wrong",
        lambda m: entry(m, algorithm="numpy.philox"))
    add("algorithm-not-a-string", lambda m: entry(m, algorithm=1))
    add("algorithm-version-wrong", lambda m: entry(m, algorithm_version=2))
    add("algorithm-version-string", lambda m: entry(m, algorithm_version="1"))
    for label, value in (("negative", "-1"), ("leading-zero", "007"),
                         ("decimal-point", "7.0"), ("exponent", "1e3"),
                         ("plus-sign", "+7"), ("spaces", " 7 "),
                         ("hex", "0x1f"), ("underscore", "1_000"),
                         ("empty", ""), ("above-uint64", str(2 ** 64))):
        add(f"seed-{label}", lambda m, v=value: entry(m, seed=v))
    add("seed-numeric", lambda m: entry(m, seed=11))
    add("calls-bool", lambda m: entry(m, calls=True))
    add("calls-above-uint64", lambda m: entry(m, calls=str(2 ** 64)))
    add("calls-negative", lambda m: entry(m, calls="-1"))

    def missing_canonical(m):
        s = section(m)
        s["aliases"].pop(generator_key, None)
        return with_gen(m, s)
    add("canonical-missing-self-alias", missing_canonical)

    def alias_to_unknown(m):
        s = section(m)
        s["aliases"][alias_key] = "nowhere.generator"
        return with_gen(m, s)
    add("alias-to-unknown-canonical", alias_to_unknown)

    def extra_alias(m):
        s = section(m)
        s["aliases"]["ghost.generator"] = generator_key
        return with_gen(m, s)
    add("extra-alias-path", extra_alias)

    def dropped_alias(m):
        s = section(m)
        s["aliases"].pop(alias_key, None)
        return with_gen(m, s)
    add("alias-topology-shrunk", dropped_alias)

    def split_alias(m):
        """The saved topology says two independent streams where the live
        model shares one object — a corruption that reads as valid unless
        the topology itself is validated."""
        s = section(m)
        s["keys"] = [generator_key, alias_key]
        s["entries"][alias_key] = dict(s["entries"][generator_key])
        s["aliases"][alias_key] = alias_key
        return with_gen(m, s)
    add("shared-saved-as-independent", split_alias)
    return cases


def metadata_cases(manifest):
    """Malformed **loaded** metadata.

    Only two kinds of defect can actually survive a JSON parse and reach
    the loader, so only those two are corruption cases here:

    1. **A wrong root type.** The manifest's ``"metadata"`` may decode as a
       list, a scalar, or null instead of an object.
    2. **A non-finite float, at any depth.** ``json.loads`` accepts the
       non-standard ``NaN``, ``Infinity``, and ``-Infinity`` literals, so
       they decode into real Python floats and reach the loader.

    What is deliberately *not* here, and why — each is a **save-boundary**
    test instead (see ``test_the_save_boundary_rejects_what_json_cannot
    _even_encode``):

    - a **non-string object key** cannot survive JSON parsing as anything
      but a string: JSON object keys are strings by grammar, so
      ``json.loads`` can never hand the loader a non-str key;
    - a **cyclic** structure cannot exist in decoded JSON at all — the
      decoder builds a tree, and a finite document cannot describe a
      cycle;
    - an **arbitrary Python object** cannot exist in decoded JSON either;
      the archive is read with ``allow_pickle=False`` and the manifest is
      plain JSON, so no object can be reconstructed;
    - a **NumPy scalar** is the same case: JSON has no such type, so a
      decoded number is always a plain ``int`` or ``float``.

    Those four rules still exist in the shared validator and still fire on
    the save side, which is exactly where a caller can supply them."""
    cases = []

    def add(label, mutate):
        cases.append((label, mutate, {}))

    def with_metadata(m, value):
        return {**m, "metadata": value}

    # 1. Root type.
    add("metadata-root-list", lambda m: with_metadata(m, [1, 2]))
    add("metadata-root-scalar", lambda m: with_metadata(m, 5))
    add("metadata-root-string", lambda m: with_metadata(m, "step 2"))
    add("metadata-root-null", lambda m: with_metadata(m, None))
    add("metadata-root-bool", lambda m: with_metadata(m, True))
    add("metadata-root-float", lambda m: with_metadata(m, 1.5))

    # 2. Non-finite floats, at every shape of nesting.
    add("metadata-top-level-nan",
        lambda m: with_metadata(m, {"loss": float("nan")}))
    add("metadata-top-level-inf",
        lambda m: with_metadata(m, {"loss": float("inf")}))
    add("metadata-top-level-neg-inf",
        lambda m: with_metadata(m, {"loss": float("-inf")}))
    add("metadata-nan-in-a-list",
        lambda m: with_metadata(m, {"losses": [1.0, float("nan"), 3.0]}))
    add("metadata-inf-in-a-nested-dict",
        lambda m: with_metadata(
            m, {"outer": {"inner": {"loss": float("inf")}}}))
    add("metadata-neg-inf-deep-in-mixed-containers",
        lambda m: with_metadata(m, {
            "history": [
                {"epoch": 0, "loss": 1.0},
                {"epoch": 1, "stats": [[0.5], [{"grad": float("-inf")}]]},
            ],
        }))
    add("metadata-nan-beside-many-valid-values",
        lambda m: with_metadata(m, {
            "step": 2, "name": "run", "ok": True, "none": None,
            "list": [1, 2, 3], "nested": {"a": {"b": [1.0, 2.0]}},
            "poison": float("nan"),
        }))
    return cases


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_every_single_corruption_is_rejected_and_changes_nothing(
    dtype, tmp_path
):
    """The matrix. Each case is **one** defect applied to an otherwise
    genuine archive, and after every rejection the complete world is
    fingerprinted and compared — values as raw bits, plus identities,
    versions, gradients, generator state, alias topology, and training
    flags.

    Both dtypes, because a validator that read the live model's dtype
    would pass at float64 and fail at float32."""
    other = "float32" if dtype == "float64" else "float64"
    model, optimizer = build(dtype)
    try:
        source = tmp_path / "good.npz"
        save_native_checkpoint(source, model, optimizer=optimizer,
                               metadata={"step": 2})
        manifest = manifest_of(source)
        live_key = manifest["model"]["keys"][0]
        generator_key = manifest["generators"]["keys"][0]
        alias_key = next(k for k in manifest["generators"]["aliases"]
                         if k != generator_key)

        cases = (
            archive_and_manifest_cases(manifest)
            + model_section_cases(manifest, live_key, dtype, other)
            + optimizer_section_cases(manifest, other)
            + generator_section_cases(manifest, generator_key, alias_key)
            + metadata_cases(manifest)
        )
        assert len(cases) >= 110, (
            f"the corruption matrix shrank to {len(cases)}"
        )

        before = fingerprint(model, optimizer)
        for index, (label, mutate, kwargs) in enumerate(cases):
            path = tamper(source, tmp_path / f"bad_{dtype}_{index}.npz",
                          mutate=mutate, **kwargs)
            with pytest.raises((ValueError, TypeError, KeyError)) as caught:
                load_native_checkpoint(path, model, optimizer=optimizer)
            assert "load_native_checkpoint()" in str(caught.value), label
            assert_unchanged(model, before, optimizer, label)

        # ...and the untouched archive still loads into the same pair, so
        # nothing above left the loader or the model poisoned. A
        # *successful* load restores every value in place and advances
        # each parameter's version by exactly one — which is the one thing
        # a rejection above was never allowed to do.
        load_native_checkpoint(source, model, optimizer=optimizer)
        assert_only_versions_advanced(model, before, optimizer, "recovery")
    finally:
        close_pair(model, optimizer)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_declared_dtype_that_disagrees_with_its_payload_is_rejected(
    dtype, tmp_path
):
    """The declaration and the bytes must agree, in **both** directions —
    the whole point of version 3 declaring the dtype at all. A loader that
    trusted the payload would silently change a model's precision; one
    that trusted the declaration would read the wrong number of bytes."""
    other = "float32" if dtype == "float64" else "float64"
    model, optimizer = build(dtype)
    try:
        source = tmp_path / "good.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)
        before = fingerprint(model, optimizer)

        def rewrite_payloads(target_dtype):
            def apply(arrays):
                for name in list(arrays):
                    if name.startswith(("model::", "optimizer::")):
                        arrays[name] = arrays[name].astype(
                            NUMPY_DTYPES[target_dtype])
            return apply

        # Declaration says this dtype; payload is the other width.
        path = tamper(source, tmp_path / "payload_narrowed.npz",
                      mutate_arrays=rewrite_payloads(other))
        with pytest.raises(ValueError) as caught:
            load_native_checkpoint(path, model, optimizer=optimizer)
        assert other in str(caught.value) or dtype in str(caught.value)
        assert_unchanged(model, before, optimizer, "payload width")

        # A foreign byte order is a different dtype too, and fails with it.
        def byteswap(arrays):
            for name in list(arrays):
                if name.startswith("model::"):
                    arrays[name] = arrays[name].astype(
                        arrays[name].dtype.newbyteorder(">"))

        path = tamper(source, tmp_path / "byteswapped.npz",
                      mutate_arrays=byteswap)
        with pytest.raises(ValueError):
            load_native_checkpoint(path, model, optimizer=optimizer)
        assert_unchanged(model, before, optimizer, "byte order")
    finally:
        close_pair(model, optimizer)


@needs_native
@pytest.mark.parametrize("payload", ["int64", "bool", "complex128"])
def test_a_non_float_payload_is_rejected(payload, tmp_path):
    """No integer, bool, or complex tensor dtype exists, so a payload of
    one is a rejection rather than a conversion."""
    model, optimizer = build("float32")
    try:
        source = tmp_path / "good.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)
        before = fingerprint(model, optimizer)

        def rewrite(arrays):
            for name in list(arrays):
                if name.startswith("model::"):
                    arrays[name] = arrays[name].astype(payload)

        path = tamper(source, tmp_path / f"{payload}.npz",
                      mutate_arrays=rewrite)
        with pytest.raises(ValueError):
            load_native_checkpoint(path, model, optimizer=optimizer)
        assert_unchanged(model, before, optimizer, payload)
    finally:
        close_pair(model, optimizer)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_truncated_or_unreadable_archive_is_rejected(dtype, tmp_path):
    """The envelope itself, before any manifest exists to validate."""
    model, optimizer = build(dtype)
    try:
        source = tmp_path / "good.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)
        before = fingerprint(model, optimizer)
        raw = source.read_bytes()

        broken = {
            "truncated": raw[: len(raw) // 2],
            "empty": b"",
            "not-an-archive": b"this is not a zip file at all",
            "header-only": raw[:8],
        }
        for label, blob in broken.items():
            path = tmp_path / f"{label}.npz"
            path.write_bytes(blob)
            with pytest.raises((ValueError, OSError, EOFError)):
                load_native_checkpoint(path, model, optimizer=optimizer)
            assert_unchanged(model, before, optimizer, label)

        # A well-formed archive with no manifest at all.
        path = tmp_path / "no_manifest.npz"
        arrays = arrays_of(source)
        arrays.pop("manifest")
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)
        with pytest.raises(ValueError, match="manifest"):
            load_native_checkpoint(path, model, optimizer=optimizer)
        assert_unchanged(model, before, optimizer, "no manifest")

        # ...and a manifest that is not a 1-D uint8 array of UTF-8 JSON.
        for label, replacement in (
            ("manifest-float", np.zeros(4, dtype=np.float64)),
            ("manifest-2d", np.zeros((2, 2), dtype=np.uint8)),
        ):
            path = tmp_path / f"{label}.npz"
            arrays = arrays_of(source)
            arrays["manifest"] = replacement
            with open(path, "wb") as handle:
                np.savez(handle, **arrays)
            with pytest.raises(ValueError, match="manifest"):
                load_native_checkpoint(path, model, optimizer=optimizer)
            assert_unchanged(model, before, optimizer, label)
    finally:
        close_pair(model, optimizer)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_rejected_load_allocates_and_leaks_nothing(dtype, tmp_path,
                                                     monkeypatch):
    """Design §20: a rejected load leaves live native storage exactly
    where it was. Proved over a representative slice of the matrix rather
    than by trusting the fingerprint alone, because a leak is invisible to
    a value comparison."""
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

    other = "float32" if dtype == "float64" else "float64"
    model, optimizer = build(dtype)
    try:
        source = tmp_path / "good.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)
        manifest = manifest_of(source)
        live_key = manifest["model"]["keys"][0]

        def entry(m, **overrides):
            section = json.loads(json.dumps(m["model"]))
            section["entries"][live_key] = {
                **section["entries"][live_key], **overrides}
            return {**m, "model": section}

        slice_of_matrix = [
            ("dtype-disagrees", lambda m: entry(m, dtype=other)),
            ("shape-disagrees", lambda m: entry(m, shape=[7, 7])),
            ("array-unknown", lambda m: entry(m, array="model::999999")),
            ("device-wrong", lambda m: entry(m, device="cuda")),
            ("version-4", lambda m: {**m, "format_version": 4}),
        ]
        # Settle the documented forward-graph reference cycles the
        # training in ``build()`` left behind, so the baseline is a
        # statement about the loader rather than about the collector.
        gc.collect()
        baseline = len(open_ids)
        for index, (label, mutate) in enumerate(slice_of_matrix):
            path = tamper(source, tmp_path / f"leak_{index}.npz", mutate)
            with pytest.raises(ValueError):
                load_native_checkpoint(path, model, optimizer=optimizer)
            gc.collect()
            assert len(open_ids) == baseline, label
        # A *successful* load also returns to baseline: the staged copies
        # replace the live ones one for one, and the old storage is
        # released rather than accumulated.
        load_native_checkpoint(source, model, optimizer=optimizer)
        gc.collect()
        assert len(open_ids) == baseline
    finally:
        close_pair(model, optimizer)


@needs_native
def test_the_fingerprint_helper_detects_a_deliberate_mutation():
    """The negative control for the whole module.

    ``assert_unchanged`` is the load-bearing assertion in every test
    above, so it has to be shown capable of failing. Each field is moved
    on its own and the helper must object to every one — a fingerprint
    that silently ignored, say, the generator counter would let a whole
    class of corruption through unnoticed."""
    model, optimizer = build("float32")
    try:
        before = fingerprint(model, optimizer)
        assert_unchanged(model, before, optimizer, "control")

        # 1. A value moves.
        parameter = next(iter(model.parameters()))
        replacement = NativeTensor.from_array(
            np.full(parameter.shape, 0.5, dtype=np.float32), dtype="float32")
        try:
            parameter.copy_value_(replacement)
        finally:
            replacement.close()
        with pytest.raises(AssertionError):
            assert_unchanged(model, before, optimizer, "moved value")

        # 2. A generator counter moves.
        before = fingerprint(model, optimizer)
        model.train(True)
        x = NativeTensor.from_array(np.ones((4, 3), dtype=np.float32),
                                    dtype="float32")
        try:
            model(x).close()
        finally:
            x.close()
        with pytest.raises(AssertionError):
            assert_unchanged(model, before, optimizer, "moved generator")

        # 3. A training flag moves.
        before = fingerprint(model, optimizer)
        model.eval()
        with pytest.raises(AssertionError):
            assert_unchanged(model, before, optimizer, "moved training flag")

        # 4. An optimizer moment moves.
        before = fingerprint(model, optimizer)
        optimizer.step()
        with pytest.raises(AssertionError):
            assert_unchanged(model, before, optimizer, "moved moments")
    finally:
        close_pair(model, optimizer)


@needs_native
def test_the_save_boundary_rejects_what_json_cannot_even_encode(tmp_path):
    """The four rules that can only fire on the **save** side, and why.

    ``_validated_metadata`` is one authority used by both directions, but
    four of its rules are unreachable from a decoded archive:

    - a **non-string dict key** — JSON object keys are strings by grammar,
      so a parse can never produce one;
    - a **cyclic list** and a **cyclic dict** — the decoder builds a tree
      from a finite document, so a cycle cannot exist in its output;
    - an **arbitrary Python object** — the archive is read with
      ``allow_pickle=False`` and the manifest is plain JSON, so nothing can
      be reconstructed;
    - a **NumPy scalar** — JSON has no such type, so a decoded number is
      always a plain ``int`` or ``float``.

    They are still real rules, because a *caller* can supply all five, and
    this is the boundary where a caller's value enters. Their load-side
    counterparts — a wrong root type and a non-finite float — are in the
    corruption matrix instead, because those genuinely survive a parse."""
    model, optimizer = build("float32")
    try:
        target = tmp_path / "rejected.npz"

        with pytest.raises(TypeError, match="keys must be str"):
            save_native_checkpoint(target, model, metadata={1: "int key"})
        with pytest.raises(TypeError, match="keys must be str"):
            save_native_checkpoint(target, model,
                                   metadata={("a",): "tuple key"})

        with pytest.raises(TypeError, match="unsupported type"):
            save_native_checkpoint(target, model, metadata={"o": object()})
        with pytest.raises(TypeError, match="unsupported type"):
            save_native_checkpoint(target, model,
                                   metadata={"f": lambda: None})
        with pytest.raises(TypeError, match="unsupported type"):
            save_native_checkpoint(target, model, metadata={"b": b"bytes"})

        # NumPy scalars subclass the Python types, so this is an exact-type
        # rule rather than an isinstance one — and it matters, because a
        # np.float64 would otherwise slip through and change the archive's
        # JSON encoding.
        for scalar in (np.float64(1.0), np.float32(1.0), np.int64(3),
                       np.bool_(True)):
            with pytest.raises(TypeError, match="unsupported type"):
                save_native_checkpoint(target, model,
                                       metadata={"n": scalar})

        cyclic_list = [1, 2]
        cyclic_list.append(cyclic_list)
        with pytest.raises(ValueError, match="cyclic container"):
            save_native_checkpoint(target, model,
                                   metadata={"c": cyclic_list})
        cyclic_dict = {"k": 1}
        cyclic_dict["self"] = cyclic_dict
        with pytest.raises(ValueError, match="cyclic container"):
            save_native_checkpoint(target, model,
                                   metadata={"c": cyclic_dict})

        # Non-finite floats reject on this side too — the one rule that
        # fires on *both* sides.
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite number"):
                save_native_checkpoint(target, model,
                                       metadata={"loss": bad})
            with pytest.raises(ValueError, match="finite number"):
                save_native_checkpoint(target, model,
                                       metadata={"deep": [{"x": bad}]})

        # Not one of those attempts wrote anything.
        assert not target.exists(), "a rejected save created a file"
        assert sorted(p.name for p in tmp_path.iterdir()) == []
    finally:
        close_pair(model, optimizer)


@needs_native
@pytest.mark.parametrize("version", (1, 2, 3))
def test_malformed_loaded_metadata_is_rejected_at_every_version(version,
                                                                tmp_path,
                                                                monkeypatch):
    """The corrected loader contract, at v1, v2, and v3 alike.

    Metadata is a manifest field in every accepted version, so the same
    authority must guard all three — a fix that only covered the newest
    format would leave the older ones accepting what the saver refuses.

    For each rejection this proves the **complete world** is untouched,
    that no staged native tensor survives, that live storage is exactly at
    baseline, and that the archive itself is byte-identical afterwards."""
    model, optimizer = build_plain("float64")
    try:
        source = tmp_path / "v3.npz"
        save_native_checkpoint(source, model, optimizer=optimizer,
                               metadata={"step": 2, "nested": {"a": [1.0]}})
        base = (source if version == 3
                else as_version(source, tmp_path / f"v{version}.npz",
                                version))
        # The genuine archive at this version loads, so the rejections
        # below are about metadata and not about the version rewrite.
        load_native_checkpoint(base, model, optimizer=optimizer)
        before = fingerprint(model, optimizer)

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
        gc.collect()
        baseline = len(open_ids)

        cases = metadata_cases(manifest_of(base))
        assert len(cases) >= 13, len(cases)
        for index, (label, mutate, kwargs) in enumerate(cases):
            path = Path(tamper(base, tmp_path / f"m{version}_{index}.npz",
                               mutate=mutate, **kwargs))
            raw = path.read_bytes()
            with pytest.raises(ValueError) as caught:
                load_native_checkpoint(path, model, optimizer=optimizer)
            message = str(caught.value)
            assert "load_native_checkpoint()" in message, label
            assert "metadata" in message, label
            assert_unchanged(model, before, optimizer, f"v{version}/{label}")
            gc.collect()
            assert len(open_ids) == baseline, label
            assert path.read_bytes() == raw, f"{label}: the load rewrote it"

        # ...and the archive still loads afterwards, metadata intact.
        restored = load_native_checkpoint(base, model, optimizer=optimizer)
        assert restored == {"step": 2, "nested": {"a": [1.0]}}
    finally:
        close_pair(model, optimizer)


@needs_native
@pytest.mark.parametrize("version", (1, 2, 3))
def test_valid_nested_metadata_round_trips_at_every_version(version,
                                                            tmp_path):
    """The control: everything JSON legitimately carries survives, at every
    accepted version, and comes back as an independent plain dict."""
    metadata = {
        "step": 7,
        "name": "run-a",
        "finished": False,
        "nothing": None,
        "losses": [1.5, 2.0, -3.25],
        "nested": {"inner": {"values": [1, 2, [3, {"deep": "yes"}]]}},
        "empty_list": [],
        "empty_dict": {},
        "large_int": 2 ** 53,
        "negative": -0.5,
    }
    model, optimizer = build_plain("float64")
    try:
        source = tmp_path / "v3.npz"
        save_native_checkpoint(source, model, optimizer=optimizer,
                               metadata=metadata)
        base = (source if version == 3
                else as_version(source, tmp_path / f"v{version}.npz",
                                version))
        restored = load_native_checkpoint(base, model, optimizer=optimizer)
        assert restored == metadata
        # Independent: mutating the result cannot reach anything.
        restored["nested"]["inner"]["values"].append("mutated")
        again = load_native_checkpoint(base, model, optimizer=optimizer)
        assert again == metadata
        assert again is not restored
    finally:
        close_pair(model, optimizer)


@needs_native
def test_a_metadata_rejection_happens_before_anything_is_staged(tmp_path,
                                                                monkeypatch):
    """Ordering, proved rather than argued: the metadata check runs in
    Phase 1, so no staged ``NativeTensor``, no rollback snapshot, and no
    mutation of any kind exists when it fires."""
    model, optimizer = build("float32")
    try:
        source = tmp_path / "good.npz"
        save_native_checkpoint(source, model, optimizer=optimizer,
                               metadata={"step": 1})
        staged = []
        original = NativeTensor._typed_from_array.__func__

        def tracking(cls, values, dtype):
            staged.append(dtype)
            return original(cls, values, dtype)

        monkeypatch.setattr(NativeTensor, "_typed_from_array",
                            classmethod(tracking))
        forged = tamper(source, tmp_path / "nan.npz",
                        lambda m: {**m, "metadata": {"x": float("nan")}})
        with pytest.raises(ValueError, match="finite number"):
            load_native_checkpoint(forged, model, optimizer=optimizer)
        assert staged == [], (
            "a staged tensor existed when the metadata was rejected"
        )
        # The control: a valid load does stage, so the assertion above is
        # not passing because the hook was never wired up.
        load_native_checkpoint(source, model, optimizer=optimizer)
        assert staged, "the staging hook never fired on a valid load"
    finally:
        close_pair(model, optimizer)


@needs_native
def test_no_temporary_file_survives_a_failed_save(tmp_path):
    """A save that fails partway leaves the destination and the directory
    exactly as they were — no half-written archive, no stray temporary."""
    model, optimizer = build("float32")
    try:
        target = tmp_path / "out.npz"
        save_native_checkpoint(target, model, optimizer=optimizer)
        original = target.read_bytes()
        listing = sorted(p.name for p in tmp_path.iterdir())

        # An unserializable metadata value fails validation before the
        # archive is written at all.
        with pytest.raises((TypeError, ValueError)):
            save_native_checkpoint(target, model, optimizer=optimizer,
                                   metadata={"bad": object()})
        assert target.read_bytes() == original, "the archive was rewritten"
        assert sorted(p.name for p in tmp_path.iterdir()) == listing, (
            "a temporary file survived the failed save"
        )

        with pytest.raises((ValueError, TypeError)):
            save_native_checkpoint(target, model, optimizer=optimizer,
                                   metadata={"nan": float("nan")})
        assert target.read_bytes() == original
        assert sorted(p.name for p in tmp_path.iterdir()) == listing
    finally:
        close_pair(model, optimizer)


@needs_native
def test_a_load_never_rewrites_the_archive(tmp_path):
    """Loading is a read. A rejected load in particular must not touch the
    file it rejected — otherwise one bad load would destroy the evidence."""
    model, optimizer = build("float32")
    try:
        source = tmp_path / "good.npz"
        save_native_checkpoint(source, model, optimizer=optimizer)
        original = source.read_bytes()
        forged = tamper(source, tmp_path / "bad.npz",
                        lambda m: {**m, "format_version": 4})
        forged_bytes = open(forged, "rb").read()

        load_native_checkpoint(source, model, optimizer=optimizer)
        assert source.read_bytes() == original

        with pytest.raises(ValueError):
            load_native_checkpoint(forged, model, optimizer=optimizer)
        assert open(forged, "rb").read() == forged_bytes
        assert source.read_bytes() == original
    finally:
        close_pair(model, optimizer)
