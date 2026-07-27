"""Native checkpoint format version 2 — persisted generator state and the
whole-checkpoint load transaction (Phase G, milestone G5; contract locked
in docs/native_rng_dropout_design.md §10).

G5 moves ``_FORMAT_VERSION`` from 1 to 2 and adds exactly one manifest
field, ``"generators"``: ``null`` for a model that registers none, or the
``keys``/``entries``/``aliases`` object that records **every** canonical
generator's state *and* the model's shared-versus-independent sharing
topology. Seeds and call counters ride as canonical decimal strings
because a ``uint64`` above ``2**53`` cannot survive an IEEE double, and no
array joins the NPZ payload. A load restores state **in place**, so every
registered ``NativeGenerator`` keeps its identity and every sharing
relationship survives.

These tests cover the schema and its determinism, exact restoration
(including the next Dropout mask against the G2 Core), the strict
both-directions topology validation, version-1 compatibility in both
directions, the four §10.7 transaction phases — with an injected
synchronous commit failure in **each** of the model, optimizer, and
generator components, and a deliverable ``KeyboardInterrupt`` — the
reservation rules that refuse an ambiguous mid-draw save or load,
concurrency against reservations and other loads, and the independence of
graph-owned Dropout masks from every load, successful or failed.

The existing version-1 suite (tests/test_native_checkpoint.py) is
unchanged in scope and still owns the model/optimizer/metadata contract;
this file owns what G5 added.

Selector: python -m pytest -q -k native_checkpoint_v2
"""

import gc
import json
import os
import threading

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeDropout,
    NativeGenerator,
    NativeLinear,
    NativeModule,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
    native_checkpoint,
)
from tensorforge.experimental import (
    _native_checkpoint_transaction as transaction,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

UINT64_MAX = 2 ** 64 - 1
X = np.arange(1.0, 13.0).reshape(3, 4)


# ======================================================================
# Helpers
# ======================================================================


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every open NativeStorage — a real live-allocation
    count, so a rollback test can prove the count returns exactly to its
    baseline instead of trusting collection."""
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


def _manifest_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def _arrays_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _resave(path, arrays):
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def _tamper(source, target, mutate_manifest=None, raw_manifest=None):
    """Copy the checkpoint at ``source`` to ``target`` with a manifest
    mutation applied. ``raw_manifest`` bypasses ``json.dumps`` so a
    duplicate object key can be constructed at all."""
    arrays = _arrays_of(source)
    manifest = json.loads(arrays.pop("manifest").tobytes().decode("utf-8"))
    if mutate_manifest is not None:
        result = mutate_manifest(manifest)
        manifest = result if result is not None else manifest
    text = raw_manifest if raw_manifest is not None else json.dumps(manifest)
    arrays["manifest"] = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    _resave(target, arrays)
    return str(target)


def _as_version_1(source, target):
    """The same archive rewritten as a genuine format-version 1 file:
    version 1 and **no** ``"generators"`` field at all, which is exactly
    what a pre-G5 TensorForge wrote."""
    def strip(manifest):
        manifest = {k: v for k, v in manifest.items() if k != "generators"}
        manifest["format_version"] = 1
        return manifest

    return _tamper(source, target, strip)


class SharedStreamModel(NativeModule):
    """Two Dropout layers on **one** generator, one on its own — the
    smallest model whose topology is not recoverable from states alone.
    Plus a real parameterized layer, so model/optimizer/generator state
    all move together."""

    def __init__(self, shared_seed=11, own_seed=22, linear_seed=1):
        super().__init__()
        self.linear = NativeLinear(4, 4, seed=linear_seed)
        shared = NativeGenerator(shared_seed)
        self.drop_a = NativeDropout(0.5, generator=shared)
        self.drop_b = NativeDropout(0.5, generator=shared)
        self.drop_c = NativeDropout(0.25, seed=own_seed)

    def forward(self, x):
        return self.drop_c(self.drop_b(self.drop_a(self.linear(x))))


def _plain_model(seed=0):
    """A generator-free model: the v1-compatible shape."""
    return NativeSequential(
        NativeLinear(4, 4, seed=seed), NativeReLU(), NativeLinear(4, 2, seed=seed + 1)
    )


def _advance(model, steps=1, optimizer=None):
    """Run ``steps`` forwards (and optimizer steps) so parameters,
    optimizer moments, and generator counters are all non-trivial."""
    x = NativeTensor.from_array(X)
    for _ in range(steps):
        y = model(x)
        y.sum().backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
        y.close()
    x.close()


def _core_mask_output(values, p, seed, call_index):
    """What the G2 Core produces at an exact ``(seed, call_index)`` — the
    oracle a restored stream must match on its very next draw."""
    source = cpp.NativeTensorCore.from_array(np.asarray(values, dtype=float))
    try:
        out = source.dropout_forward(p, seed=seed, call_index=call_index)
        try:
            return out.to_numpy().copy()
        finally:
            out.close()
    finally:
        source.close()


def _fingerprint(model, optimizer=None):
    """Everything a rollback must restore exactly."""
    state = {
        "parameters": [
            (name, tensor.to_numpy().copy(),
             getattr(tensor, "version", None), id(tensor))
            for name, tensor in model._state_named_tensors()
        ],
        "generators": model.generator_state_dict(),
        "generator_ids": [id(g) for g in model.generators()],
        "generator_names": [n for n, _ in model.named_generators()],
    }
    if optimizer is not None:
        state["lr"] = optimizer.lr
        state["optimizer_id"] = id(optimizer)
        if isinstance(optimizer, NativeAdam):
            state["steps"] = optimizer.step_counts
            state["betas"] = optimizer.betas
            state["eps"] = optimizer.eps
            state["moments"] = [
                buffer.to_numpy().copy()
                for buffer in optimizer._m + optimizer._v
            ]
    return state


def _assert_unchanged(model, before, optimizer=None, label=""):
    current = _fingerprint(model, optimizer)
    for (name, values, version, identity), (n2, v2, ver2, id2) in zip(
        before["parameters"], current["parameters"]
    ):
        assert name == n2, label
        assert np.array_equal(values, v2), (label, name)
        assert version == ver2, (label, name, "version moved")
        assert identity == id2, (label, name, "identity changed")
    assert before["generators"] == current["generators"], label
    assert before["generator_ids"] == current["generator_ids"], label
    assert before["generator_names"] == current["generator_names"], label
    if optimizer is not None:
        assert before["lr"] == current["lr"], label
        assert before["optimizer_id"] == current["optimizer_id"], label
        if isinstance(optimizer, NativeAdam):
            assert before["steps"] == current["steps"], label
            assert before["betas"] == current["betas"], label
            assert before["eps"] == current["eps"], label
            for saved, live in zip(before["moments"], current["moments"]):
                assert np.array_equal(saved, live), label


def _close(model, optimizer=None):
    for _, parameter in model.named_parameters():
        parameter.close()
    if optimizer is not None and isinstance(optimizer, NativeAdam):
        optimizer.close()


# ======================================================================
# 1. Version and schema
# ======================================================================


@needs_native
def test_the_format_name_is_unchanged_and_the_version_is_two():
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 2
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2)
    assert native_checkpoint._MANIFEST_KEYS == {
        "format", "format_version", "model", "optimizer", "generators",
        "metadata",
    }
    assert native_checkpoint._MANIFEST_KEYS_V1 == {
        "format", "format_version", "model", "optimizer", "metadata",
    }


@needs_native
def test_a_generator_free_model_writes_an_explicit_null(tmp_path):
    """Absence is stated, never inferred from a missing field — so a
    reader can tell "this model had no generators" from "this archive
    predates generators"."""
    model = _plain_model()
    path = tmp_path / "plain.npz"
    save_native_checkpoint(path, model)
    manifest = _manifest_of(path)
    assert manifest["format_version"] == 2
    assert "generators" in manifest
    assert manifest["generators"] is None
    load_native_checkpoint(path, model)          # round-trips unchanged
    _close(model)


@needs_native
def test_the_shared_generator_manifest_is_exactly_the_locked_shape(tmp_path):
    model = SharedStreamModel()
    _advance(model, steps=2)
    path = tmp_path / "shared.npz"
    save_native_checkpoint(path, model)
    section = _manifest_of(path)["generators"]

    # One entry per unique generator; every registered path in aliases.
    assert list(section) == ["keys", "entries", "aliases"]
    assert section["keys"] == ["drop_a.generator", "drop_c.generator"]
    assert list(section["entries"]) == section["keys"]
    assert section["aliases"] == {
        "drop_a.generator": "drop_a.generator",
        "drop_b.generator": "drop_a.generator",     # the shared path
        "drop_c.generator": "drop_c.generator",
    }
    assert section["entries"]["drop_a.generator"] == {
        "algorithm": "tensorforge.splitmix64",
        "algorithm_version": 1,
        "seed": "11",
        "calls": "4",        # two forwards through two sharing layers
    }
    assert section["entries"]["drop_c.generator"]["calls"] == "2"
    _close(model)


@needs_native
def test_generator_state_adds_no_npz_array(tmp_path):
    """Four scalars per generator belong in the manifest; the array-name
    space is untouched, so the existing duplicate/missing/extra array
    checks need no new cases."""
    model = SharedStreamModel()
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    _advance(model, steps=1, optimizer=optimizer)
    path = tmp_path / "arrays.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    with np.load(path, allow_pickle=False) as archive:
        names = sorted(archive.files)
    assert names == [
        "manifest",
        "model::000000", "model::000001",
        "optimizer::m::000000", "optimizer::m::000001",
        "optimizer::v::000000", "optimizer::v::000001",
    ]
    assert not any("generator" in name for name in names)
    _close(model, optimizer)


@needs_native
def test_seed_and_calls_are_canonical_decimal_strings(tmp_path):
    """Never JSON numbers: ``2**64 - 1`` does not survive an IEEE
    double, and one spelling per value keeps two saves byte-identical."""
    model = NativeDropout(0.5, seed=UINT64_MAX)
    model.generator.load_state({
        "algorithm": "tensorforge.splitmix64", "algorithm_version": 1,
        "seed": UINT64_MAX, "calls": UINT64_MAX - 1,
    })
    path = tmp_path / "extreme.npz"
    save_native_checkpoint(path, model)
    entry = _manifest_of(path)["generators"]["entries"]["generator"]
    assert entry["seed"] == "18446744073709551615"
    assert entry["calls"] == "18446744073709551614"
    assert type(entry["seed"]) is str and type(entry["calls"]) is str
    # The raw JSON really carries them quoted — no reader can round them.
    with np.load(path, allow_pickle=False) as archive:
        text = archive["manifest"].tobytes().decode("utf-8")
    assert '"seed": "18446744073709551615"' in text
    assert "1.8446744073709552e+19" not in text
    # ...and they come back exactly.
    model.generator.reseed(1)
    load_native_checkpoint(path, model)
    assert model.generator.seed == UINT64_MAX
    assert model.generator.calls == UINT64_MAX - 1


@needs_native
def test_zero_is_canonical_and_round_trips(tmp_path):
    model = NativeDropout(0.5, seed=0)
    path = tmp_path / "zero.npz"
    save_native_checkpoint(path, model)
    entry = _manifest_of(path)["generators"]["entries"]["generator"]
    assert entry == {
        "algorithm": "tensorforge.splitmix64", "algorithm_version": 1,
        "seed": "0", "calls": "0",
    }
    model.generator.reseed(7)
    load_native_checkpoint(path, model)
    assert model.generator.seed == 0 and model.generator.calls == 0


@needs_native
def test_serialization_is_deterministic(tmp_path):
    """Canonical names and both orders are functions of the model alone,
    so saving the same model twice is byte-identical — the property a
    reviewer needs to diff two checkpoints at all."""
    model = SharedStreamModel()
    _advance(model, steps=1)
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    save_native_checkpoint(first, model)
    save_native_checkpoint(second, model)
    with np.load(first, allow_pickle=False) as a:
        text_a = a["manifest"].tobytes()
    with np.load(second, allow_pickle=False) as b:
        text_b = b["manifest"].tobytes()
    assert text_a == text_b
    _close(model)


@needs_native
def test_user_mapping_order_cannot_alter_the_archive(tmp_path):
    """The archive comes from the model traversal, not from any
    caller-supplied ordering: loading generator state through a
    differently-ordered mapping first changes nothing about the file."""
    model = SharedStreamModel()
    _advance(model, steps=1)
    baseline = tmp_path / "base.npz"
    save_native_checkpoint(baseline, model)
    state = model.generator_state_dict()
    reordered = {k: state[k] for k in reversed(list(state))}
    model.load_generator_state_dict(reordered)
    reissued = tmp_path / "reissued.npz"
    save_native_checkpoint(reissued, model)
    assert _manifest_of(baseline) == _manifest_of(reissued)
    _close(model)


@needs_native
def test_independent_generators_with_equal_values_stay_separate(tmp_path):
    """Sharing is identity, never state equality: two generators with the
    same seed and counter are two entries and two aliases."""
    class Twins(NativeModule):
        def __init__(self):
            super().__init__()
            self.a = NativeDropout(0.5, seed=5)
            self.b = NativeDropout(0.5, seed=5)

        def forward(self, x):
            return self.b(self.a(x))

    model = Twins()
    path = tmp_path / "twins.npz"
    save_native_checkpoint(path, model)
    section = _manifest_of(path)["generators"]
    assert section["keys"] == ["a.generator", "b.generator"]
    assert section["aliases"] == {
        "a.generator": "a.generator", "b.generator": "b.generator",
    }
    assert (section["entries"]["a.generator"]
            == section["entries"]["b.generator"])
    load_native_checkpoint(path, model)
    assert model.a.generator is not model.b.generator


# ======================================================================
# 2. Exact restoration
# ======================================================================


@needs_native
def test_exact_restoration_of_state_identity_and_topology(tmp_path):
    model = SharedStreamModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _advance(model, steps=3, optimizer=optimizer)
    path = tmp_path / "exact.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    saved = _fingerprint(model, optimizer)
    identities = [id(g) for g in model.generators()]

    # Drift everything away from the checkpoint.
    _advance(model, steps=2, optimizer=optimizer)
    model.drop_a.generator.reseed(123456789)
    model.drop_c.generator.reset()
    assert _fingerprint(model, optimizer)["generators"] != saved["generators"]

    load_native_checkpoint(path, model, optimizer=optimizer)
    restored = _fingerprint(model, optimizer)
    assert restored["generators"] == saved["generators"]
    assert restored["generator_names"] == saved["generator_names"]
    # Loaded in place: the same objects, still shared the same way.
    assert [id(g) for g in model.generators()] == identities
    assert model.drop_a.generator is model.drop_b.generator
    assert model.drop_a.generator is not model.drop_c.generator
    for saved_values, live_values in zip(saved["moments"],
                                         restored["moments"]):
        assert np.array_equal(saved_values, live_values)
    assert restored["steps"] == saved["steps"]
    _close(model, optimizer)


@needs_native
def test_the_next_mask_matches_the_core_at_the_restored_index(tmp_path):
    """The property the whole milestone exists for: after a load, the
    very next draw is the one the saved run would have made — checked
    against the G2 Core at the restored ``(seed, call_index)``, not
    against "these two look different"."""
    module = NativeDropout(0.5, seed=97)
    x = NativeTensor.from_array(X)
    for _ in range(3):
        module(x).close()
    assert module.generator.calls == 3
    path = tmp_path / "stream.npz"
    save_native_checkpoint(path, module)

    # Consume more, then reseed entirely — the stream is gone.
    module(x).close()
    module.generator.reseed(1)

    load_native_checkpoint(path, module)
    assert module.generator.state() == {
        "algorithm": "tensorforge.splitmix64", "algorithm_version": 1,
        "seed": 97, "calls": 3,
    }
    y = module(x)
    assert np.array_equal(y.to_numpy(), _core_mask_output(X, 0.5, 97, 3))
    assert module.generator.calls == 4
    y.close()
    x.close()


@needs_native
def test_a_shared_stream_resumes_interleaved(tmp_path):
    """Restoring the states without the topology would diverge on the
    very next step; this pins that it does not."""
    model = SharedStreamModel()
    _advance(model, steps=1)
    path = tmp_path / "interleaved.npz"
    save_native_checkpoint(path, model)
    shared_calls = model.drop_a.generator.calls

    _advance(model, steps=3)
    load_native_checkpoint(path, model)
    assert model.drop_a.generator.calls == shared_calls
    # Two more layer forwards advance the ONE shared stream twice.
    x = NativeTensor.from_array(X)
    model.drop_a(x).close()
    model.drop_b(x).close()
    assert model.drop_a.generator.calls == shared_calls + 2
    assert model.drop_b.generator.calls == shared_calls + 2
    x.close()
    _close(model)


@needs_native
def test_calls_at_zero_and_at_the_maximum_round_trip(tmp_path):
    module = NativeDropout(0.5, seed=3)
    path = tmp_path / "counts.npz"
    save_native_checkpoint(path, module)
    module(NativeTensor.from_array(X)).close()
    load_native_checkpoint(path, module)
    assert module.generator.calls == 0

    # An exhausted generator stays exhausted: the counter is a count, so
    # UINT64_MAX is a reachable value, not a sentinel.
    module.generator.load_state({
        "algorithm": "tensorforge.splitmix64", "algorithm_version": 1,
        "seed": 3, "calls": UINT64_MAX,
    })
    exhausted = tmp_path / "exhausted.npz"
    save_native_checkpoint(exhausted, module)
    assert (_manifest_of(exhausted)["generators"]["entries"]["generator"]
            ["calls"] == str(UINT64_MAX))
    module.generator.reset()
    load_native_checkpoint(exhausted, module)
    assert module.generator.calls == UINT64_MAX
    with pytest.raises(RuntimeError, match="exhausted"):
        module(NativeTensor.from_array(X))
    assert module.generator.calls == UINT64_MAX


@needs_native
def test_a_high_bit_seed_survives_exactly(tmp_path):
    for seed in (2 ** 63, 2 ** 63 + 1, 0xFFFFFFFFFFFFFFFE, 12297829382473034410):
        module = NativeDropout(0.5, seed=seed)
        path = tmp_path / f"seed-{seed}.npz"
        save_native_checkpoint(path, module)
        module.generator.reseed(0)
        load_native_checkpoint(path, module)
        assert module.generator.seed == seed


@needs_native
def test_buffers_optimizer_and_generators_restore_together(tmp_path):
    """The four state families in one archive, drifted apart and put
    back together."""
    from tensorforge.experimental import NativeBatchNorm1d

    class Mixed(NativeModule):
        def __init__(self):
            super().__init__()
            self.linear = NativeLinear(4, 4, seed=2)
            self.norm = NativeBatchNorm1d(4)
            self.drop = NativeDropout(0.5, seed=31)

        def forward(self, x):
            return self.drop(self.norm(self.linear(x)))

    model = Mixed()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _advance(model, steps=2, optimizer=optimizer)
    path = tmp_path / "mixed.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    saved = _fingerprint(model, optimizer)
    buffer_names = [name for name, _ in model.named_buffers()]
    assert buffer_names == ["norm.running_mean", "norm.running_var"]

    _advance(model, steps=3, optimizer=optimizer)
    model.drop.generator.reseed(99)
    load_native_checkpoint(path, model, optimizer=optimizer)
    restored = _fingerprint(model, optimizer)
    for (name, values, _, _), (n2, v2, _, _) in zip(saved["parameters"],
                                                    restored["parameters"]):
        assert name == n2
        assert np.array_equal(values, v2), name
    assert restored["generators"] == saved["generators"]
    assert restored["steps"] == saved["steps"]
    _close(model, optimizer)


# ======================================================================
# 3. Strict topology validation
# ======================================================================


def _topology_cases(source, tmp_path):
    """Every §10.5 generator-section corruption, as (name, path)."""
    cases = []

    def add(name, mutate, raw=None):
        cases.append((name, _tamper(
            source, tmp_path / f"bad-{name}.npz", mutate, raw_manifest=raw,
        )))

    def section(manifest):
        return json.loads(json.dumps(manifest["generators"]))

    add("null-section", lambda m: {**m, "generators": None})
    add("wrong-section-type", lambda m: {**m, "generators": []})
    add("missing-entries",
        lambda m: {**m, "generators": {k: v for k, v in section(m).items()
                                       if k != "entries"}})
    add("missing-aliases",
        lambda m: {**m, "generators": {k: v for k, v in section(m).items()
                                       if k != "aliases"}})
    add("missing-keys",
        lambda m: {**m, "generators": {k: v for k, v in section(m).items()
                                       if k != "keys"}})
    add("extra-field",
        lambda m: {**m, "generators": {**section(m), "extra": 1}})

    def malformed_key(m):
        s = section(m)
        s["keys"] = [""] + s["keys"][1:]
        return {**m, "generators": s}

    add("malformed-canonical-name", malformed_key)

    def duplicate_key(m):
        s = section(m)
        s["keys"] = [s["keys"][0], s["keys"][0]]
        return {**m, "generators": s}

    add("duplicate-canonical-key", duplicate_key)

    def entries_out_of_order(m):
        s = section(m)
        s["entries"] = {k: s["entries"][k] for k in reversed(s["keys"])}
        return {**m, "generators": s}

    add("entries-out-of-order", entries_out_of_order)

    def alias_to_absent(m):
        s = section(m)
        s["aliases"] = {**s["aliases"], "drop_b.generator": "nowhere.generator"}
        return {**m, "generators": s}

    add("alias-target-absent", alias_to_absent)

    def missing_self_alias(m):
        s = section(m)
        s["aliases"] = {k: v for k, v in s["aliases"].items()
                        if k != "drop_c.generator"}
        return {**m, "generators": s}

    add("canonical-absent-from-aliases", missing_self_alias)

    def not_self_mapped(m):
        s = section(m)
        s["aliases"]["drop_c.generator"] = "drop_a.generator"
        return {**m, "generators": s}

    add("canonical-not-self-mapped", not_self_mapped)

    def collapsed_aliases(m):
        """Every path aliased onto one canonical entry — which would
        leave the other canonical entry referenced by nothing."""
        s = section(m)
        s["aliases"] = {"drop_a.generator": "drop_a.generator",
                        "drop_b.generator": "drop_a.generator",
                        "drop_c.generator": "drop_a.generator"}
        return {**m, "generators": s}

    add("collapsed-aliases", collapsed_aliases)

    def missing_alias_path(m):
        s = section(m)
        s["aliases"] = {k: v for k, v in s["aliases"].items()
                        if k != "drop_b.generator"}
        return {**m, "generators": s}

    add("missing-alias-path", missing_alias_path)

    def unexpected_alias_path(m):
        s = section(m)
        s["aliases"]["drop_z.generator"] = "drop_a.generator"
        return {**m, "generators": s}

    add("unexpected-alias-path", unexpected_alias_path)

    def malformed_alias_path(m):
        s = section(m)
        s["aliases"]["a..b"] = "drop_a.generator"
        return {**m, "generators": s}

    add("malformed-alias-path", malformed_alias_path)

    def saved_independent(m):
        """The archive says drop_a and drop_b are independent; the live
        model shares one generator."""
        s = section(m)
        s["keys"] = ["drop_a.generator", "drop_b.generator",
                     "drop_c.generator"]
        s["entries"] = {
            "drop_a.generator": s["entries"]["drop_a.generator"],
            "drop_b.generator": dict(s["entries"]["drop_a.generator"]),
            "drop_c.generator": s["entries"]["drop_c.generator"],
        }
        s["aliases"] = {"drop_a.generator": "drop_a.generator",
                        "drop_b.generator": "drop_b.generator",
                        "drop_c.generator": "drop_c.generator"}
        return {**m, "generators": s}

    add("saved-independent-live-shared", saved_independent)

    def unknown_algorithm(m):
        s = section(m)
        s["entries"]["drop_a.generator"]["algorithm"] = "numpy.pcg64"
        return {**m, "generators": s}

    add("unsupported-algorithm", unknown_algorithm)

    def wrong_algorithm_version(m):
        s = section(m)
        s["entries"]["drop_a.generator"]["algorithm_version"] = 2
        return {**m, "generators": s}

    add("unsupported-algorithm-version", wrong_algorithm_version)

    def bad_entry_fields(m):
        s = section(m)
        s["entries"]["drop_a.generator"] = {"seed": "1", "calls": "1"}
        return {**m, "generators": s}

    add("entry-missing-fields", bad_entry_fields)

    for label, value in [
        ("negative-seed", "-1"),
        ("leading-zero-seed", "007"),
        ("decimal-point-seed", "7.0"),
        ("exponent-seed", "1e3"),
        ("plus-seed", "+7"),
        ("space-seed", " 7 "),
        ("hex-seed", "0x1f"),
        ("underscore-seed", "1_000"),
        ("above-uint64-seed", str(2 ** 64)),
        ("empty-seed", ""),
    ]:
        def make(v=value):
            def mutate(m):
                s = section(m)
                s["entries"]["drop_a.generator"]["seed"] = v
                return {**m, "generators": s}
            return mutate

        add(label, make())

    def numeric_seed(m):
        s = section(m)
        s["entries"]["drop_a.generator"]["seed"] = 11
        return {**m, "generators": s}

    add("json-numeric-seed", numeric_seed)

    def bool_calls(m):
        s = section(m)
        s["entries"]["drop_a.generator"]["calls"] = True
        return {**m, "generators": s}

    add("bool-calls", bool_calls)

    def above_range_calls(m):
        s = section(m)
        s["entries"]["drop_a.generator"]["calls"] = str(2 ** 64)
        return {**m, "generators": s}

    add("above-uint64-calls", above_range_calls)
    return cases


@needs_native
def test_every_topology_corruption_fails_before_any_live_change(tmp_path):
    model = SharedStreamModel()
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    _advance(model, steps=2, optimizer=optimizer)
    source = tmp_path / "good.npz"
    save_native_checkpoint(source, model, optimizer=optimizer)

    before = _fingerprint(model, optimizer)
    cases = _topology_cases(source, tmp_path)
    assert len(cases) >= 25, "the corruption matrix shrank"
    for name, path in cases:
        with pytest.raises((ValueError, TypeError)) as info:
            load_native_checkpoint(path, model, optimizer=optimizer)
        assert "load_native_checkpoint()" in str(info.value), name
        _assert_unchanged(model, before, optimizer, label=name)
    # ...and the same pair recovers completely on the good archive.
    load_native_checkpoint(source, model, optimizer=optimizer)
    _close(model, optimizer)


@needs_native
def test_a_duplicate_alias_path_is_rejected(tmp_path):
    """``json`` keeps the last duplicate silently, which would turn "this
    archive names one path twice with two targets" into "this archive is
    fine". The loader rejects the repeat instead."""
    model = SharedStreamModel()
    source = tmp_path / "dup.npz"
    save_native_checkpoint(source, model)
    manifest = _manifest_of(source)
    text = json.dumps(manifest)
    doubled = text.replace(
        '"drop_b.generator": "drop_a.generator"',
        '"drop_b.generator": "drop_a.generator", '
        '"drop_b.generator": "drop_c.generator"',
        1,
    )
    assert doubled != text
    path = _tamper(source, tmp_path / "duplicated.npz", raw_manifest=doubled)
    with pytest.raises(ValueError, match="repeats the object key"):
        load_native_checkpoint(path, model)
    _close(model)


@needs_native
def test_saved_shared_versus_live_independent_is_rejected(tmp_path):
    """The archive shares two paths; the live model has two independent
    generators. Restoring the states alone would silently change the
    model's stochastic behavior, so it fails."""
    shared_model = SharedStreamModel()
    path = tmp_path / "shared.npz"
    save_native_checkpoint(path, shared_model)

    class Independent(NativeModule):
        def __init__(self):
            super().__init__()
            self.linear = NativeLinear(4, 4, seed=1)
            self.drop_a = NativeDropout(0.5, seed=11)
            self.drop_b = NativeDropout(0.5, seed=11)   # NOT shared
            self.drop_c = NativeDropout(0.25, seed=22)

        def forward(self, x):
            return self.drop_c(self.drop_b(self.drop_a(self.linear(x))))

    live = Independent()
    before = _fingerprint(live)
    with pytest.raises(ValueError) as info:
        load_native_checkpoint(path, live)
    message = str(info.value)
    assert "drop_b.generator" in message
    assert ("topology" in message or "do not match" in message)
    _assert_unchanged(live, before)
    _close(shared_model)
    _close(live)


@needs_native
def test_a_renamed_or_reordered_registration_is_rejected(tmp_path):
    """A shared generator's canonical name is the first path the
    traversal reaches, so swapping the registration order changes it —
    and the load fails naming both rather than remapping by state."""
    model = SharedStreamModel()
    path = tmp_path / "order.npz"
    save_native_checkpoint(path, model)

    class Swapped(NativeModule):
        def __init__(self):
            super().__init__()
            self.linear = NativeLinear(4, 4, seed=1)
            shared = NativeGenerator(11)
            self.drop_b = NativeDropout(0.5, generator=shared)   # first now
            self.drop_a = NativeDropout(0.5, generator=shared)
            self.drop_c = NativeDropout(0.25, seed=22)

        def forward(self, x):
            return self.drop_c(self.drop_a(self.drop_b(self.linear(x))))

    swapped = Swapped()
    assert [n for n, _ in swapped.named_generators()] == [
        "drop_b.generator", "drop_c.generator"
    ]
    before = _fingerprint(swapped)
    with pytest.raises(ValueError) as info:
        load_native_checkpoint(path, swapped)
    message = str(info.value)
    assert "drop_a.generator" in message and "drop_b.generator" in message
    _assert_unchanged(swapped, before)
    _close(model)
    _close(swapped)


@needs_native
def test_the_comparison_uses_a_real_traversal_not_a_supplied_list(tmp_path):
    """Registering another generator after the save makes the archive
    incomplete — detected from the live model, with nothing supplied by
    the caller."""
    model = SharedStreamModel()
    path = tmp_path / "before.npz"
    save_native_checkpoint(path, model)
    model.extra = NativeGenerator(77)
    before = _fingerprint(model)
    with pytest.raises(ValueError, match="extra"):
        load_native_checkpoint(path, model)
    _assert_unchanged(model, before)
    _close(model)


# ======================================================================
# 4. Version-1 compatibility
# ======================================================================


@needs_native
def test_version_1_loads_into_a_generator_free_model(tmp_path):
    model = _plain_model()
    _advance(model, steps=1)
    source = tmp_path / "v2.npz"
    save_native_checkpoint(source, model)
    legacy = _as_version_1(source, tmp_path / "v1.npz")
    assert _manifest_of(legacy)["format_version"] == 1
    assert "generators" not in _manifest_of(legacy)

    fresh = _plain_model(seed=9)
    load_native_checkpoint(legacy, fresh)
    for (_, saved), (_, live) in zip(model.named_parameters(),
                                     fresh.named_parameters()):
        assert np.array_equal(saved.to_numpy(), live.to_numpy())
    _close(model)
    _close(fresh)


@needs_native
def test_version_1_with_an_optimizer_and_metadata_still_works(tmp_path):
    model = _plain_model()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _advance(model, steps=2, optimizer=optimizer)
    source = tmp_path / "v2.npz"
    save_native_checkpoint(source, model, optimizer=optimizer,
                           metadata={"epoch": 4})
    legacy = _as_version_1(source, tmp_path / "v1.npz")
    metadata = load_native_checkpoint(legacy, model, optimizer=optimizer)
    assert metadata == {"epoch": 4}
    _close(model, optimizer)


@needs_native
def test_version_1_into_a_generator_bearing_model_fails_atomically(tmp_path):
    """No seed and no counter is ever fabricated — not zero, not fresh
    entropy, not the generator's current value. The load fails naming the
    generators the archive cannot supply."""
    plain = _plain_model()
    source = tmp_path / "v2.npz"
    save_native_checkpoint(source, plain, metadata={"m": 1})
    legacy = _as_version_1(source, tmp_path / "v1.npz")

    # A generator-bearing model with the same tensor state keys.
    class WithGenerator(NativeModule):
        def __init__(self):
            super().__init__()
            self.inner = _plain_model(seed=3)
            self.drop = NativeDropout(0.5, seed=44)

        def forward(self, x):
            return self.drop(self.inner(x))

    model = WithGenerator()
    # Line the key space up so only the generator rule can fail.
    keys = list(model.state_dict())
    assert keys == ["inner.0.weight", "inner.0.bias",
                    "inner.2.weight", "inner.2.bias"]
    renamed = _tamper(
        legacy, tmp_path / "v1-renamed.npz",
        lambda m: {**m, "model": {
            "keys": keys,
            "entries": {new: m["model"]["entries"][old] for new, old
                        in zip(keys, m["model"]["keys"])},
        }},
    )
    before = _fingerprint(model)
    with pytest.raises(ValueError) as info:
        load_native_checkpoint(renamed, model)
    message = str(info.value)
    assert "version 1" in message
    assert "drop.generator" in message
    _assert_unchanged(model, before)
    # Nothing was invented: the live stream is exactly what it was.
    assert model.drop.generator.seed == 44
    assert model.drop.generator.calls == 0
    _close(plain)
    _close(model)


@needs_native
def test_version_2_generators_into_a_generator_free_model_fails(tmp_path):
    model = SharedStreamModel()
    path = tmp_path / "withgen.npz"
    save_native_checkpoint(path, model)

    class NoGenerators(NativeModule):
        def __init__(self):
            super().__init__()
            self.linear = NativeLinear(4, 4, seed=1)

        def forward(self, x):
            return self.linear(x)

    live = NoGenerators()
    # Match the model key space, so only the generator rule can fire.
    renamed = _tamper(
        path, tmp_path / "renamed.npz",
        lambda m: {**m, "model": {
            "keys": ["linear.weight", "linear.bias"],
            "entries": {"linear.weight": m["model"]["entries"]["linear.weight"],
                        "linear.bias": m["model"]["entries"]["linear.bias"]},
        }},
    )
    before = _fingerprint(live)
    with pytest.raises(ValueError, match="registers none"):
        load_native_checkpoint(renamed, live)
    _assert_unchanged(live, before)
    _close(model)
    _close(live)


@needs_native
def test_a_v1_manifest_carrying_a_generator_section_is_rejected(tmp_path):
    model = SharedStreamModel()
    source = tmp_path / "src.npz"
    save_native_checkpoint(source, model)
    hybrid = _tamper(source, tmp_path / "hybrid.npz",
                     lambda m: {**m, "format_version": 1})
    with pytest.raises(ValueError, match="format version 1"):
        load_native_checkpoint(hybrid, model)
    _close(model)


@needs_native
@pytest.mark.parametrize("version", [0, 3, 99, -1])
def test_unsupported_versions_fail_before_any_live_state(tmp_path, version):
    model = SharedStreamModel()
    source = tmp_path / "src.npz"
    save_native_checkpoint(source, model)
    bad = _tamper(source, tmp_path / f"v{version}.npz",
                  lambda m: {**m, "format_version": version})
    before = _fingerprint(model)
    with pytest.raises(ValueError, match="format_version"):
        load_native_checkpoint(bad, model)
    _assert_unchanged(model, before)
    _close(model)


# ======================================================================
# 5. Whole-checkpoint transaction and rollback
# ======================================================================


@needs_native
@pytest.mark.parametrize(
    "seam", ["_commit_optimizer", "_commit_generators",
             "_reach_commit_boundary"],
)
@pytest.mark.parametrize(
    "error", [RuntimeError, KeyboardInterrupt, MemoryError],
)
def test_an_injected_commit_failure_rolls_back_all_four_families(
    tmp_path, monkeypatch, live_storages, seam, error,
):
    """§10.7 Phase 3. A failure at any commit position — including a
    deliverable ``KeyboardInterrupt``, which is explicitly **not** an
    exception to atomicity — restores the model, its buffers, the
    optimizer, and every generator, preserves every identity, moves no
    parameter version, and returns native live storage to baseline."""
    model = SharedStreamModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _advance(model, steps=2, optimizer=optimizer)
    path = tmp_path / "txn.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)

    # Drift away from the archive so a silent no-op cannot pass.
    _advance(model, steps=2, optimizer=optimizer)
    model.drop_a.generator.reseed(4242)
    gc.collect()
    before = _fingerprint(model, optimizer)
    baseline = len(live_storages)

    def boom(*args, **kwargs):
        raise error(f"injected {seam}")

    monkeypatch.setattr(transaction, seam, boom)
    with pytest.raises(error):
        load_native_checkpoint(path, model, optimizer=optimizer)
    monkeypatch.undo()

    _assert_unchanged(model, before, optimizer, label=f"{seam}/{error}")
    gc.collect()
    assert len(live_storages) == baseline, seam
    # ...and the very same pair loads cleanly afterwards.
    load_native_checkpoint(path, model, optimizer=optimizer)
    _close(model, optimizer)


@needs_native
def test_a_model_commit_failure_commits_nothing_else(tmp_path, monkeypatch,
                                                     live_storages):
    """The model commits first, so its own failure must leave the
    optimizer and the generators untouched — there is nothing to unwind,
    and nothing may have run ahead."""
    model = SharedStreamModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _advance(model, steps=2, optimizer=optimizer)
    path = tmp_path / "model-commit.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    _advance(model, steps=1, optimizer=optimizer)
    model.drop_c.generator.reseed(31337)
    gc.collect()
    before = _fingerprint(model, optimizer)
    baseline = len(live_storages)

    def boom(*args, **kwargs):
        raise RuntimeError("injected model commit failure")

    monkeypatch.setattr(transaction, "_commit_model", boom)
    with pytest.raises(RuntimeError, match="injected model commit"):
        load_native_checkpoint(path, model, optimizer=optimizer)
    monkeypatch.undo()

    _assert_unchanged(model, before, optimizer)
    gc.collect()
    assert len(live_storages) == baseline
    _close(model, optimizer)


@needs_native
def test_a_staging_failure_commits_nothing(tmp_path, monkeypatch,
                                           live_storages):
    """§10.7 Phase 2: everything that can allocate happens before the
    commit, and a failure there closes it all."""
    model = SharedStreamModel()
    optimizer = NativeSGD(model.parameters(), lr=0.1)
    _advance(model, steps=1)
    path = tmp_path / "stage.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    _advance(model, steps=1)
    gc.collect()
    before = _fingerprint(model, optimizer)
    baseline = len(live_storages)

    real = NativeTensor.from_array
    calls = {"n": 0}

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError("forced staging failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(NativeTensor, "from_array", staticmethod(failing))
    with pytest.raises(MemoryError):
        load_native_checkpoint(path, model, optimizer=optimizer)
    monkeypatch.undo()

    _assert_unchanged(model, before, optimizer)
    gc.collect()
    assert len(live_storages) == baseline
    _close(model, optimizer)


@needs_native
def test_a_rollback_snapshot_failure_commits_nothing(tmp_path, monkeypatch,
                                                    live_storages):
    """The rollback snapshots are themselves staged, so failing to build
    one aborts before the commit rather than committing unprotected."""
    model = SharedStreamModel()
    _advance(model, steps=1)
    path = tmp_path / "snapshot.npz"
    save_native_checkpoint(path, model)
    _advance(model, steps=1)
    gc.collect()
    before = _fingerprint(model)
    baseline = len(live_storages)

    real = NativeModule.state_dict
    seen = {"n": 0}

    def failing(self):
        seen["n"] += 1
        if seen["n"] == 1:
            raise MemoryError("forced rollback-snapshot failure")
        return real(self)

    monkeypatch.setattr(NativeModule, "state_dict", failing)
    with pytest.raises(MemoryError):
        load_native_checkpoint(path, model)
    monkeypatch.undo()

    _assert_unchanged(model, before)
    gc.collect()
    assert len(live_storages) == baseline
    _close(model)


@needs_native
def test_generator_validation_failure_leaves_the_optimizer_untouched(
    tmp_path,
):
    """No optimizer state is partially restored when generator
    validation fails: the check is prevalidation, so nothing has run."""
    model = SharedStreamModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _advance(model, steps=2, optimizer=optimizer)
    source = tmp_path / "src.npz"
    save_native_checkpoint(source, model, optimizer=optimizer)
    _advance(model, steps=1, optimizer=optimizer)
    before = _fingerprint(model, optimizer)

    bad = _tamper(
        source, tmp_path / "bad-generator.npz",
        lambda m: {**m, "generators": {
            **m["generators"],
            "entries": {**m["generators"]["entries"],
                        "drop_c.generator": {
                            **m["generators"]["entries"]["drop_c.generator"],
                            "seed": "not-a-number"}},
        }},
    )
    with pytest.raises(ValueError, match="canonical decimal"):
        load_native_checkpoint(bad, model, optimizer=optimizer)
    _assert_unchanged(model, before, optimizer)
    _close(model, optimizer)


@needs_native
def test_optimizer_validation_failure_leaves_generators_untouched(tmp_path):
    model = SharedStreamModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _advance(model, steps=2, optimizer=optimizer)
    source = tmp_path / "src.npz"
    save_native_checkpoint(source, model, optimizer=optimizer)
    model.drop_a.generator.reseed(5150)
    before = _fingerprint(model, optimizer)

    bad = _tamper(source, tmp_path / "bad-optimizer.npz",
                  lambda m: {**m, "optimizer": {**m["optimizer"],
                                                "lr": -1.0}})
    with pytest.raises(ValueError):
        load_native_checkpoint(bad, model, optimizer=optimizer)
    _assert_unchanged(model, before, optimizer)
    assert model.drop_a.generator.seed == 5150
    _close(model, optimizer)


@needs_native
def test_the_transaction_seams_are_private_and_unexported():
    import tensorforge
    import tensorforge.experimental as experimental

    # The module is private: importable inside the package, absent from
    # the public export list, and never re-exported at top level.
    assert "_native_checkpoint_transaction" not in experimental.__all__
    assert not any(name.startswith("_") for name in experimental.__all__)
    for absent in ("commit_checkpoint", "CheckpointPlan", "ModelRollback",
                   "OptimizerRollback", "GeneratorRollback"):
        assert absent not in experimental.__all__
        assert not hasattr(tensorforge, absent)
    # The seams exist for tests, and are module-private.
    for seam in ("_commit_model", "_commit_optimizer", "_commit_generators",
                 "_reach_commit_boundary", "_rollback_model",
                 "_rollback_optimizer", "_rollback_generators"):
        assert callable(getattr(transaction, seam)), seam
    # No new public checkpoint entry point.
    public = sorted(
        name for name in dir(native_checkpoint)
        if not name.startswith("_") and callable(getattr(native_checkpoint,
                                                         name))
    )
    assert "save_native_checkpoint" in public
    assert "load_native_checkpoint" in public
    for absent in ("save_generator_state", "load_generator_state",
                   "upgrade_checkpoint", "migrate_checkpoint"):
        assert absent not in public, absent


# ======================================================================
# 6. Save behavior and destination atomicity
# ======================================================================


@needs_native
def test_a_save_during_an_active_reservation_is_refused(tmp_path):
    """A generator whose next index has been decided but not committed
    has no single honest state to record."""
    model = SharedStreamModel()
    good = tmp_path / "good.npz"
    save_native_checkpoint(good, model)
    original_bytes = good.read_bytes()

    token = model.drop_a.generator._reserve_call()
    before = _fingerprint(model)
    with pytest.raises(RuntimeError, match="reservation"):
        save_native_checkpoint(good, model)
    assert good.read_bytes() == original_bytes, "the destination changed"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["good.npz"]
    _assert_unchanged(model, before)
    assert model.drop_a.generator._has_active_reservation() is True

    model.drop_a.generator._abandon_call(token)
    save_native_checkpoint(good, model)          # recovers immediately
    _close(model)


@needs_native
def test_a_save_and_a_load_during_a_construction_claim_are_refused(
    tmp_path, monkeypatch,
):
    """The **claim** window too, not just a published reservation: an
    index that has been decided against the current seed but whose token
    does not exist yet is exactly as ambiguous.

    Reached by running inside token construction — the one moment when a
    claim stands and no reservation is published — which is the same
    window §3.6 makes addressable for finalizer reentry."""
    from tensorforge.experimental import native_generator

    model = SharedStreamModel()
    good = tmp_path / "good.npz"
    save_native_checkpoint(good, model)
    original_bytes = good.read_bytes()
    outcome = {}

    real_token = native_generator._ReservationToken

    class ProbingToken(real_token):
        def __init__(self, generator, serial, index):
            # A claim stands and nothing is published: prove it.
            assert generator._claim_serial == serial
            assert generator._active_serial == 0
            for label, attempt in (
                ("save", lambda: save_native_checkpoint(good, model)),
                ("load", lambda: load_native_checkpoint(good, model)),
            ):
                try:
                    attempt()
                except BaseException as error:
                    outcome[label] = error
                else:
                    outcome[label] = None
            super().__init__(generator, serial, index)

    monkeypatch.setattr(native_generator, "_ReservationToken", ProbingToken)
    token = model.drop_c.generator._reserve_call()
    monkeypatch.undo()

    for label in ("save", "load"):
        assert isinstance(outcome[label], RuntimeError), (label, outcome)
        assert "reservation" in str(outcome[label]), label
    assert good.read_bytes() == original_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["good.npz"]

    model.drop_c.generator._abandon_call(token)
    save_native_checkpoint(good, model)          # both recover
    load_native_checkpoint(good, model)
    _close(model)


@needs_native
def test_a_generator_manifest_failure_leaves_the_destination_intact(
    tmp_path, monkeypatch,
):
    model = SharedStreamModel()
    path = tmp_path / "dest.npz"
    save_native_checkpoint(path, model, metadata={"which": "first"})
    original_bytes = path.read_bytes()
    before = _fingerprint(model)

    def boom(*args, **kwargs):
        raise RuntimeError("injected generator-manifest failure")

    monkeypatch.setattr(native_checkpoint, "_generator_section", boom)
    with pytest.raises(RuntimeError, match="injected generator-manifest"):
        save_native_checkpoint(path, model, metadata={"which": "second"})
    monkeypatch.undo()

    assert path.read_bytes() == original_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dest.npz"]
    assert _manifest_of(path)["metadata"] == {"which": "first"}
    _assert_unchanged(model, before)
    _close(model)


@needs_native
def test_a_failing_archive_write_leaves_generators_and_destination_intact(
    tmp_path, monkeypatch,
):
    model = SharedStreamModel()
    path = tmp_path / "dest.npz"
    save_native_checkpoint(path, model)
    original_bytes = path.read_bytes()
    before = _fingerprint(model)

    def failing_savez(*args, **kwargs):
        raise OSError("forced archive-write failure")

    monkeypatch.setattr(np, "savez", failing_savez)
    with pytest.raises(OSError, match="forced archive-write"):
        save_native_checkpoint(path, model)
    monkeypatch.undo()

    assert path.read_bytes() == original_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dest.npz"]
    _assert_unchanged(model, before)
    assert all(not g._has_active_reservation() for g in model.generators())
    _close(model)


@needs_native
def test_a_save_never_advances_or_mutates_a_generator(tmp_path):
    model = SharedStreamModel()
    _advance(model, steps=1)
    before = model.generator_state_dict()
    for index in range(3):
        save_native_checkpoint(tmp_path / f"s{index}.npz", model)
        assert model.generator_state_dict() == before
        assert all(not g._has_active_reservation() for g in model.generators())
    _close(model)


# ======================================================================
# 7. Reservations and concurrency
# ======================================================================


@needs_native
def test_a_load_during_an_active_reservation_is_refused(tmp_path):
    model = SharedStreamModel()
    path = tmp_path / "reserved.npz"
    save_native_checkpoint(path, model)
    _advance(model, steps=1)
    before = _fingerprint(model)

    token = model.drop_a.generator._reserve_call()
    with pytest.raises(RuntimeError, match="reservation"):
        load_native_checkpoint(path, model)
    _assert_unchanged(model, before)
    assert model.drop_a.generator._has_active_reservation() is True
    model.drop_a.generator._abandon_call(token)
    load_native_checkpoint(path, model)          # recovers
    _close(model)


@needs_native
def test_a_reservation_on_an_untouched_generator_does_not_block(tmp_path):
    """Only the *targets* matter: a reservation on a generator this model
    does not register is irrelevant."""
    model = SharedStreamModel()
    path = tmp_path / "other.npz"
    save_native_checkpoint(path, model)
    outsider = NativeGenerator(5)
    token = outsider._reserve_call()
    load_native_checkpoint(path, model)
    outsider._abandon_call(token)
    _close(model)


@needs_native
def test_a_concurrent_reservation_never_overlaps_a_load(tmp_path):
    """Two outcomes only: the reservation wins the lock and the load
    rejects without mutating anything, or it waits and observes the
    finished state. Never a half-loaded generator."""
    model = SharedStreamModel()
    _advance(model, steps=1)
    path = tmp_path / "race.npz"
    save_native_checkpoint(path, model)
    saved = model.generator_state_dict()
    _advance(model, steps=2)

    results = []
    start = threading.Barrier(2, timeout=10)

    def loader():
        start.wait()
        for _ in range(40):
            try:
                load_native_checkpoint(path, model)
                results.append("loaded")
            except RuntimeError as error:
                results.append(f"refused: {error}")

    def reserver():
        start.wait()
        generator = model.drop_a.generator
        for _ in range(40):
            try:
                token = generator._reserve_call()
            except RuntimeError:
                continue
            generator._abandon_call(token)

    threads = [threading.Thread(target=loader),
               threading.Thread(target=reserver)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive(), "the load/reserve pair deadlocked"

    # Every generator holds a complete, self-consistent state — never a
    # half-applied one — and abandoned reservations advanced nothing.
    for name, state in model.generator_state_dict().items():
        assert set(state) == {"algorithm", "algorithm_version", "seed",
                              "calls"}
        assert state["seed"] == saved[name]["seed"], name
    assert all(not g._has_active_reservation() for g in model.generators())
    assert results, "the loader never ran"
    _close(model)


@needs_native
def test_two_concurrent_overlapping_loads_do_not_deadlock(tmp_path):
    """Both loads take the same global identity-ordered generator lock
    sequence, so overlapping targets reached through different modules
    cannot form a cycle."""
    shared = NativeGenerator(11)

    def build(order):
        module = NativeModule()
        drops = {
            "a": NativeDropout(0.5, generator=shared),
            "b": NativeDropout(0.5, generator=shared),
        }
        for name in order:
            setattr(module, name, drops[name])
        module.own = NativeDropout(0.5, seed=hash(order[0]) % 1000)
        return module

    first = build(("a", "b"))
    second = build(("b", "a"))
    path_one = tmp_path / "one.npz"
    path_two = tmp_path / "two.npz"
    save_native_checkpoint(path_one, first)
    save_native_checkpoint(path_two, second)

    errors = []
    start = threading.Barrier(2, timeout=10)

    def load(path, model):
        try:
            start.wait()
            for _ in range(30):
                load_native_checkpoint(path, model)
        except BaseException as error:       # pragma: no cover - diagnostic
            errors.append(error)

    threads = [
        threading.Thread(target=load, args=(path_one, first)),
        threading.Thread(target=load, args=(path_two, second)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive(), "two concurrent loads deadlocked"
    assert errors == [], errors
    assert shared.seed == 11


@needs_native
def test_a_concurrent_generator_state_load_does_not_deadlock(tmp_path):
    """A checkpoint load and a ``load_generator_state_dict`` over the
    same generators take one order, so they serialize rather than
    deadlock."""
    model = SharedStreamModel()
    path = tmp_path / "concurrent.npz"
    save_native_checkpoint(path, model)
    state = model.generator_state_dict()

    errors = []
    start = threading.Barrier(2, timeout=10)

    def checkpoint_loads():
        try:
            start.wait()
            for _ in range(30):
                load_native_checkpoint(path, model)
        except BaseException as error:       # pragma: no cover - diagnostic
            errors.append(error)

    def state_loads():
        try:
            start.wait()
            for _ in range(30):
                model.load_generator_state_dict(state)
        except BaseException as error:       # pragma: no cover - diagnostic
            errors.append(error)

    threads = [threading.Thread(target=checkpoint_loads),
               threading.Thread(target=state_loads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive(), "checkpoint/state load deadlocked"
    assert errors == [], errors
    assert model.generator_state_dict() == state
    _close(model)


@needs_native
def test_the_checkpoint_takes_only_generator_locks(monkeypatch, tmp_path):
    """The transaction adds no new lock order: the only locks a load ever
    holds are generator locks, taken inside the shared multi-generator
    transaction in its own global identity order."""
    from tensorforge.experimental import native_generator

    model = SharedStreamModel()
    path = tmp_path / "locks.npz"
    save_native_checkpoint(path, model)

    orders = []
    original = native_generator._ordered_targets

    def recording(generators):
        ordered = original(generators)
        orders.append([id(g) for g in ordered])
        return ordered

    monkeypatch.setattr(native_generator, "_ordered_targets", recording)
    load_native_checkpoint(path, model)
    monkeypatch.undo()

    assert orders, "no generator lock sequence was taken"
    for order in orders:
        assert order == sorted(order), "not the global id() order"
    _close(model)


# ======================================================================
# 8. Existing graphs are independent of every load
# ======================================================================


@needs_native
def test_a_pre_load_dropout_graph_keeps_its_original_mask(tmp_path,
                                                          live_storages):
    """A saved multiplier mask is private native state owned by graph
    history and reachable from no registry, so no phase of a load touches
    it: the pre-load graph's gradient is still its **original** mask, the
    backward redraws nothing, and it consumes no generator call."""
    module = NativeDropout(0.5, seed=61)
    x = NativeTensor.from_array(X, requires_grad=True)
    y = module(x)
    mask = y._graph_resources[0].to_numpy().copy()
    calls_after_forward = module.generator.calls
    assert calls_after_forward == 1

    path = tmp_path / "graph.npz"
    save_native_checkpoint(path, module)
    # Move the stream somewhere completely different, then restore it.
    module.generator.reseed(987654321)
    module(x).close()
    load_native_checkpoint(path, module)
    assert module.generator.calls == calls_after_forward

    # The graph built before all of that is untouched.
    assert np.array_equal(y._graph_resources[0].to_numpy(), mask)
    grad = NativeTensor.from_array(np.ones_like(X))
    y.backward(gradient=grad)
    assert np.array_equal(x.grad.to_numpy(), mask)
    # Backward consumed no generator call and reserved nothing.
    assert module.generator.calls == calls_after_forward
    assert module.generator._has_active_reservation() is False
    grad.close()
    y.close()
    x.close()


@needs_native
def test_a_failed_load_leaves_a_pre_load_graph_identical(tmp_path,
                                                         monkeypatch,
                                                         live_storages):
    module = NativeDropout(0.5, seed=67)
    x = NativeTensor.from_array(X, requires_grad=True)
    y = module(x)
    mask = y._graph_resources[0].to_numpy().copy()
    path = tmp_path / "failed.npz"
    save_native_checkpoint(path, module)
    module.generator.reseed(11223344)

    def boom(*args, **kwargs):
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(transaction, "_commit_generators", boom)
    with pytest.raises(RuntimeError, match="injected commit"):
        load_native_checkpoint(path, module)
    monkeypatch.undo()

    assert module.generator.seed == 11223344     # rolled back to pre-load
    assert np.array_equal(y._graph_resources[0].to_numpy(), mask)
    grad = NativeTensor.from_array(np.ones_like(X))
    y.backward(gradient=grad)
    assert np.array_equal(x.grad.to_numpy(), mask)
    grad.close()
    y.close()
    x.close()


@needs_native
def test_graph_resources_release_exactly_once_around_a_load(tmp_path,
                                                            live_storages):
    module = NativeDropout(0.5, seed=71)
    path = tmp_path / "lifetime.npz"
    save_native_checkpoint(path, module)
    baseline = len(live_storages)

    x = NativeTensor.from_array(X, requires_grad=True)
    y = module(x)
    assert len(live_storages) > baseline
    load_native_checkpoint(path, module)
    grad = NativeTensor.from_array(np.ones_like(X))
    y.backward(gradient=grad)
    assert y._graph_resources == ()
    grad.close()
    y.close()
    x.grad.close()
    x.close()
    gc.collect()
    assert len(live_storages) == baseline


@needs_native
def test_generator_loading_moves_no_parameter_version(tmp_path):
    """Loading generator state touches no tensor and stales no graph;
    the version movement in a checkpoint load is the **model** section's,
    exactly as before G5."""
    model = SharedStreamModel()
    _advance(model, steps=1)
    path = tmp_path / "versions.npz"
    save_native_checkpoint(path, model)
    versions = [p.version for p in model.parameters()]
    load_native_checkpoint(path, model)
    assert [p.version for p in model.parameters()] == [
        version + 1 for version in versions
    ]
    # ...and a pure generator-state load moves none at all.
    model.load_generator_state_dict(model.generator_state_dict())
    assert [p.version for p in model.parameters()] == [
        version + 1 for version in versions
    ]
    _close(model)


# ======================================================================
# 9. Separation and scope
# ======================================================================


@needs_native
def test_no_python_or_numpy_global_rng_is_captured(tmp_path):
    """Design §11.1: reproducibility is exact for the state actually
    captured, and full-program determinism is not claimed."""
    import random

    model = SharedStreamModel()
    path = tmp_path / "scope.npz"
    save_native_checkpoint(path, model)
    text = json.dumps(_manifest_of(path)).lower()
    for banned in ("mt19937", "pcg64", "random_state", "rng_state",
                   "numpy_state", "python_random", "shuffle", "epoch_order",
                   "dataloader", "scheduler", "augment"):
        assert banned not in text, banned

    # A load leaves both global RNGs exactly where they were.
    random.seed(1234)
    np.random.seed(4321)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    load_native_checkpoint(path, model)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    _close(model)


@needs_native
def test_the_source_captures_no_foreign_random_state():
    source = native_checkpoint.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for banned in ("np.random", "numpy.random", "import random",
                   "getstate", "set_state", "mt19937", "secrets."):
        assert banned not in text, banned


@needs_native
def test_the_capability_boundary_is_exactly_what_g5_moved():
    from tensorforge.experimental import native_checkpoint as module

    # Moved: the format version and the state-support reporting.
    assert module._FORMAT_VERSION == 2
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",
    )
    assert cpp.backend_info()["state_support"] == cpp.STATE_SUPPORT
    # Unmoved: everything else.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    dropout_symbols = [name for name in cpp._CHECKED_KERNELS
                       if "dropout" in name or "random" in name]
    assert dropout_symbols == ["tf_core_dropout_forward"]
    for inventory in (cpp.AUTOGRAD_OPS, cpp.TENSOR_CORE_OPS, cpp.RAW_KERNELS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS):
        assert "checkpoint_generator_state" not in inventory


@needs_native
def test_g5_added_no_kernel_abi_symbol_or_stable_change():
    """G5 is Python persistence: no C++, no C ABI symbol, no ctypes
    declaration, no Core method, no operation, no module, no export."""
    import tensorforge
    import tensorforge.nn as nn

    for absent in ("dropout_backward", "generator_forward", "rng_forward"):
        assert absent not in cpp.TENSOR_CORE_OPS, absent
    for absent in ("tf_core_generator_state", "tf_core_checkpoint",
                   "tf_core_dropout_backward"):
        assert absent not in cpp._CHECKED_KERNELS, absent
    # The stable line is untouched and still has its own RNG capture.
    assert hasattr(nn, "Dropout")
    assert hasattr(tensorforge, "save_checkpoint")
    assert not hasattr(tensorforge, "save_native_checkpoint")
    assert not hasattr(tensorforge, "NativeGenerator")


@needs_native
def test_g9_and_later_milestones_have_not_begun():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    for absent in ("tests/test_native_phase_g.py",):
        assert not (repo_root / absent).exists(), absent
    # No result artifact of any kind is written by this milestone.
    assert not (repo_root / "benchmark_results").exists()


@needs_native
def test_live_storage_returns_to_baseline_across_a_full_cycle(
    tmp_path, live_storages,
):
    baseline = len(live_storages)
    model = SharedStreamModel()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _advance(model, steps=2, optimizer=optimizer)
    path = tmp_path / "cycle.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    load_native_checkpoint(path, model, optimizer=optimizer)
    after_cycle = len(live_storages)

    for _ in range(3):
        save_native_checkpoint(path, model, optimizer=optimizer)
        load_native_checkpoint(path, model, optimizer=optimizer)
    gc.collect()
    assert len(live_storages) == after_cycle

    optimizer.close()
    for _, parameter in model.named_parameters():
        parameter.close()
    gc.collect()
    assert len(live_storages) == baseline
