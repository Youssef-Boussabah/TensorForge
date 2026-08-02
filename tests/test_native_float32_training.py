"""Integrated native float32 training and exact deterministic resume
(Phase I, milestone I9).

``examples/native_float32_training.py`` runs one deep native model —

    Conv2d(1 -> 4, 3x3, pad 1) -> BatchNorm2d(4) -> ReLU -> MaxPool2d(2)
      -> Dropout(p) -> Flatten -> Linear(36 -> 8) -> BatchNorm1d(8)
      -> ReLU -> LayerNorm(8) -> Dropout(p) -> Linear(8 -> 3)
      -> NativeCrossEntropyLoss, with NativeAdam

— twice at **each** supported dtype, and proves the interrupted-and-resumed
run bitwise identical to the uninterrupted one.

These tests verify the deterministic dataset and batch schedule, the
architecture and every state family it carries, the shared-generator alias
topology, run-to-run bit-identity, exact uninterrupted-versus-resumed
equivalence through one **format-version 3** checkpoint at both dtypes, the
first-resumed-step gradient claim, the next-Dropout-mask proof, the
BatchNorm evaluation-snapshot independence proof, the explicit external
loop-progress metadata and its strict validation, cleanup and live-storage
lifecycle, and the I9 scope boundary.

**Every equality here is exact, over raw IEEE-754 bit patterns** — a
``uint32`` view at float32 and a ``uint64`` view at float64. Never a
tolerance, never ``allclose``, and **never a comparison between the two
dtypes**: each is proved only against itself (design §18.3).

Selector: python -m pytest -q -k native_float32_training
"""

import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeGenerator,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint

from examples.native_float32_training import (
    ALIAS_GENERATOR_KEY,
    BATCH_SIZE,
    CANONICAL_GENERATOR_KEY,
    CHANNELS,
    CONV_CHANNELS,
    CONV_DROPOUT_NAME,
    DEFAULT_LR,
    DENSE_DROPOUT_NAME,
    DROPOUT_CALLS_PER_STEP,
    DROPOUT_P,
    EXPECTED_GENERATOR_ALIASES,
    FRESH_GENERATOR_SEED,
    GENERATOR_SEED,
    HEIGHT,
    HIDDEN_FEATURES,
    NUM_BATCHES,
    NUM_CLASSES,
    POOLED_FEATURES,
    PROGRESS_FIELDS,
    REQUIRED,
    RUN_DTYPES,
    SAMPLES,
    SPLIT_STEP,
    TOTAL_STEPS,
    WIDTH,
    NativeFloat32Classifier,
    alias_topology,
    batch_index_for_step,
    bits,
    build_batches,
    build_dataset,
    build_loss,
    build_model,
    build_optimizer,
    evaluate,
    failed_checks,
    generator_state,
    host_images,
    main,
    model_bits,
    model_dtypes,
    native_input,
    optimizer_bits,
    progress_metadata,
    run_dtype_proof,
    run_uninterrupted,
    validated_progress,
)

EXAMPLE = (Path(__file__).resolve().parent.parent / "examples"
           / "native_float32_training.py")

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)

pytestmark = needs_native


@pytest.fixture()
def live_storages(monkeypatch):
    """The ids of every open NativeStorage — a real live-allocation count,
    so a lifecycle test can prove the count returns exactly to its baseline
    instead of trusting collection."""
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


def _close_model(model, optimizer=None):
    if optimizer is not None:
        optimizer.close()
    for parameter in model.parameters():
        parameter.close()
    for buffer in model.buffers():
        buffer.close()


# ==========================================================================
# 1. The fixed task
# ==========================================================================


def test_the_dataset_is_deterministic_exact_and_free_of_global_rng():
    """Every value is a quarter or an eighth, so the same nested list is
    exactly representable in **both** binary32 and binary64 — which is why
    neither dtype's run is a rounded version of the other's."""
    images, targets = build_dataset()
    again, again_targets = build_dataset()
    assert images == again and targets == again_targets
    assert images is not again                       # independent objects
    assert len(images) == SAMPLES == len(targets)
    assert targets == [index % NUM_CLASSES for index in range(SAMPLES)]
    array = np.asarray(images, dtype=np.float64)
    assert array.shape == (SAMPLES, CHANNELS, HEIGHT, WIDTH)
    assert np.isfinite(array).all()
    # Exactly representable at binary32: narrowing and widening is identity.
    narrowed = array.astype(np.float32).astype(np.float64)
    assert np.array_equal(narrowed.view(np.uint64), array.view(np.uint64))
    # Every value is a multiple of an eighth.
    assert np.array_equal(array * 8.0, np.round(array * 8.0))
    # Both classes' rows really differ, so the task is not degenerate.
    assert len({tuple(np.ravel(image)) for image in images}) == SAMPLES


def test_the_dataset_ignores_the_global_numpy_rng():
    np.random.seed(1)
    first = build_dataset()
    np.random.seed(2)
    assert build_dataset() == first


def test_the_batch_schedule_is_a_pure_function_of_the_step():
    assert [batch_index_for_step(step) for step in range(9)] == [
        0, 1, 2, 0, 1, 2, 0, 1, 2]
    with pytest.raises(TypeError):
        batch_index_for_step(True)
    with pytest.raises(TypeError):
        batch_index_for_step(1.0)
    with pytest.raises(ValueError):
        batch_index_for_step(-1)


def test_the_split_step_is_neither_the_first_nor_the_last_step():
    """A resume at step 0 proves nothing (there is no state yet) and one at
    the last step proves almost nothing. It is also deliberately not a
    multiple of ``NUM_BATCHES``, so the resume lands mid-cycle in the batch
    schedule and a loop that restarted at batch 0 would diverge."""
    assert 0 < SPLIT_STEP < TOTAL_STEPS - 1
    assert SPLIT_STEP % NUM_BATCHES != 0
    assert TOTAL_STEPS % NUM_BATCHES == 0     # every batch used equally often


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_host_images_are_physically_the_run_dtype(dtype):
    images, _ = build_dataset()
    array = host_images(images, dtype)
    assert array.dtype == np.dtype(dtype)
    assert array.shape == (SAMPLES, CHANNELS, HEIGHT, WIDTH)


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_native_input_uses_the_public_constructor_and_the_asked_for_dtype(
    dtype
):
    """Ingress is the **public** ``NativeTensor.from_array`` with an
    explicit ``dtype``, never a private ``_typed*`` entry point and never
    an inference from the host array."""
    images, _ = build_dataset()
    tensor = native_input(host_images(images, dtype), dtype)
    try:
        assert isinstance(tensor, NativeTensor)
        assert tensor.dtype == dtype
        assert tensor.to_numpy().dtype == np.dtype(dtype)
    finally:
        tensor.close()
    source = EXAMPLE.read_text(encoding="utf-8")
    for private in ("_typed_from_array", "_typed_zeros", "_typed_full",
                    "_typed(", "_trusted_dtype", "_from_core",
                    "_normalize_internal_dtype"):
        assert private not in source, private


# ==========================================================================
# 2. The architecture and its state families
# ==========================================================================


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_every_state_owning_child_carries_the_run_dtype(dtype):
    model = build_model(dtype)
    try:
        assert model.dtype == dtype
        tags = model_dtypes(model)
        assert set(tags.values()) == {dtype}
        # All four state families are present and named.
        names = set(tags)
        assert {"conv.weight", "conv.bias", "hidden.weight", "output.weight",
                "batch_norm2d.gamma", "batch_norm1d.beta",
                "layer_norm.weight"} <= names
        assert {"batch_norm2d.running_mean", "batch_norm2d.running_var",
                "batch_norm1d.running_mean",
                "batch_norm1d.running_var"} <= names
    finally:
        _close_model(model)


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_stateless_children_hold_no_dtype_of_their_own(dtype):
    """ReLU, MaxPool2d, Flatten and both Dropouts own no dtype-bearing
    numeric state, so they take no dtype argument and must not gain one —
    a second authority there could disagree with the data."""
    model = build_model(dtype)
    try:
        for name in ("relu1", "pool", "relu2", "flatten",
                     CONV_DROPOUT_NAME, DENSE_DROPOUT_NAME):
            child = getattr(model, name)
            assert not hasattr(child, "dtype"), name
            assert child.parameters() == [], name
    finally:
        _close_model(model)


def test_the_two_dropouts_share_one_registered_generator_object():
    """Genuine object identity, not equal state: the topology is what the
    checkpoint records and re-validates, and two independent generators
    would consume two streams rather than one interleaved one."""
    model = build_model("float32")
    try:
        first = getattr(model, CONV_DROPOUT_NAME).generator
        second = getattr(model, DENSE_DROPOUT_NAME).generator
        assert first is second
        assert isinstance(first, NativeGenerator)
        # Identity-deduplicated, so the canonical walk reports it once...
        canonical = [name for name, _ in model.named_generators()]
        assert canonical == [CANONICAL_GENERATOR_KEY]
        # ...while both registered paths map to it.
        topology = alias_topology(model)
        assert topology["aliases"] == EXPECTED_GENERATOR_ALIASES
        assert topology["shared"] is True
        assert ALIAS_GENERATOR_KEY in topology["aliases"]
        assert (topology["aliases"][ALIAS_GENERATOR_KEY]
                == CANONICAL_GENERATOR_KEY)
    finally:
        _close_model(model)


def test_the_shapes_line_up_through_the_whole_network():
    model = build_model("float32")
    images, targets = build_dataset()
    x = native_input(host_images(images, "float32")[:BATCH_SIZE], "float32")
    try:
        model.train()
        logits = model(x)
        try:
            assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
            assert logits.dtype == "float32"
        finally:
            logits.close()
        assert POOLED_FEATURES == CONV_CHANNELS * (HEIGHT // 2) * (WIDTH // 2)
        assert model.hidden.weight.shape == (POOLED_FEATURES, HIDDEN_FEATURES)
        assert model.output.weight.shape == (HIDDEN_FEATURES, NUM_CLASSES)
    finally:
        x.close()
        _close_model(model)


def test_a_fresh_model_starts_from_deliberately_different_seeds():
    """The restore target must not be able to match by accident: every
    initialization seed **and** the generator seed differ."""
    standard = build_model("float32")
    fresh = build_model("float32", fresh=True)
    try:
        assert (generator_state(standard)["seed"] == GENERATOR_SEED
                != FRESH_GENERATOR_SEED == generator_state(fresh)["seed"])
        assert model_bits(standard, "float32") != model_bits(fresh, "float32")
        # ...and the fresh one still has the required topology.
        assert alias_topology(fresh)["aliases"] == EXPECTED_GENERATOR_ALIASES
    finally:
        _close_model(standard)
        _close_model(fresh)


# ==========================================================================
# 3. The bit-comparison mechanism itself
# ==========================================================================


@pytest.mark.parametrize("dtype, width", [("float64", 8), ("float32", 4)])
def test_bits_reads_raw_patterns_and_refuses_the_wrong_width(dtype, width):
    """The helper the whole proof rests on. It must distinguish values a
    tolerance would not, and it must refuse an array of the other dtype
    rather than converting — otherwise "the values matched" could quietly
    mean "the values matched after a conversion this runtime never
    performs"."""
    array = np.array([0.0, -0.0, 1.0], dtype=dtype)
    pattern = bits(array, dtype)
    assert len(pattern) == 3
    assert pattern[0] != pattern[1]          # +0.0 and -0.0 are distinguished
    assert array[0] == array[1]              # ...though == cannot tell
    assert np.dtype(dtype).itemsize == width
    other = "float32" if dtype == "float64" else "float64"
    with pytest.raises(TypeError):
        bits(array, other)


def test_bits_separates_neighbouring_floats():
    """One ULP apart is a real difference, and the comparison must see it."""
    value = np.float32(1.0)
    nudged = np.nextafter(value, np.float32(2.0))
    assert float(value) != float(nudged)
    assert bits(np.array([value]), "float32") != bits(
        np.array([nudged]), "float32")


# ==========================================================================
# 4. Determinism and the training contract
# ==========================================================================


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_two_independent_runs_are_bit_identical(dtype):
    first = run_uninterrupted(dtype)
    second = run_uninterrupted(dtype)
    for key in ("loss_bits", "parameters", "final_train_logit_bits",
                "split_step_gradients", "generator", "topology",
                "final_eval"):
        assert first[key] == second[key], key
    assert first["optimizer"] == second["optimizer"]


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_run_is_nontrivial_and_leaves_finite_state(dtype):
    run = run_uninterrupted(dtype)
    assert run["parameters_changed"] is True
    assert run["gradients_cleared"] is True
    assert all(np.isfinite(value) for value in run["losses"])
    assert len(run["losses"]) == TOTAL_STEPS
    # Adam stepped once per parameter per step.
    assert run["optimizer"]["step_counts"] == [TOTAL_STEPS] * len(
        run["optimizer"]["step_counts"])
    # The generator advanced by exactly two calls per step — one per
    # training-mode Dropout forward — and evaluation consumed none.
    assert run["calls_equal_expected"] is True
    assert (run["generator"]["calls"]
            == TOTAL_STEPS * DROPOUT_CALLS_PER_STEP)
    assert run["eval_consumed_no_call"] is True
    # The running buffers really moved.
    assert any(run["parameters"][name] != [0] * len(run["parameters"][name])
               for name in run["parameters"] if "running_mean" in name)


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_every_gradient_and_every_value_is_at_the_run_dtype(dtype):
    run = run_uninterrupted(dtype)
    assert set(run["dtypes"].values()) == {dtype}
    assert run["split_step_gradients"], "no gradients were captured"
    # ``bits`` refuses a mismatched width, so a captured gradient reaching
    # this point at all is itself the dtype proof; the count is the shape.
    assert set(run["split_step_gradients"]) == {
        name for name in run["dtypes"] if "running_" not in name}


def test_the_two_dtypes_are_genuinely_different_runs():
    """The negative control for the whole file: nothing here compares the
    two dtypes, and this proves there *is* something to keep apart. If a
    float32 run silently computed in binary64 and narrowed, the widened
    float32 losses would equal the float64 ones exactly."""
    narrow = run_uninterrupted("float32")
    wide = run_uninterrupted("float64")
    assert [float(value) for value in narrow["losses"]] != [
        float(value) for value in wide["losses"]]
    assert narrow["dtypes"] != wide["dtypes"]


# ==========================================================================
# 5. Progress metadata — carried explicitly, validated strictly
# ==========================================================================


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_progress_metadata_is_json_compatible_and_exact(dtype):
    metadata = progress_metadata(SPLIT_STEP, dtype)
    assert set(PROGRESS_FIELDS) <= set(metadata)
    assert metadata["training_step"] == SPLIT_STEP
    assert metadata["next_batch_index"] == batch_index_for_step(SPLIT_STEP)
    assert metadata["run_dtype"] == dtype
    assert json.loads(json.dumps(metadata)) == metadata


@pytest.mark.parametrize("bad, error", [
    (True, TypeError), (1.5, TypeError), ("5", TypeError), (-1, ValueError),
])
def test_progress_metadata_validates_completed_steps(bad, error):
    with pytest.raises(error):
        progress_metadata(bad, "float32")


def test_progress_metadata_rejects_an_unknown_run_dtype():
    with pytest.raises(ValueError):
        progress_metadata(SPLIT_STEP, "float16")


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_validated_progress_accepts_what_it_produces(dtype):
    metadata = progress_metadata(SPLIT_STEP, dtype)
    assert validated_progress(metadata, dtype) == (
        SPLIT_STEP, batch_index_for_step(SPLIT_STEP))


@pytest.mark.parametrize("field", PROGRESS_FIELDS)
def test_validated_progress_rejects_a_missing_field_rather_than_defaulting(
    field
):
    """A resume that silently restarted from step 0 would still "work",
    would still converge, and would be a different run — so it must be an
    error, not a fallback."""
    metadata = progress_metadata(SPLIT_STEP, "float32")
    del metadata[field]
    with pytest.raises(ValueError):
        validated_progress(metadata, "float32")


def test_validated_progress_rejects_a_schedule_inconsistency():
    metadata = progress_metadata(SPLIT_STEP, "float32")
    metadata["next_batch_index"] += 1
    with pytest.raises(ValueError):
        validated_progress(metadata, "float32")


def test_validated_progress_rejects_the_other_dtypes_archive():
    """There is no dtype conversion at load and none may be added, so a
    float32 run must refuse a float64 archive with a clear message rather
    than a shape or dtype error from deeper down."""
    metadata = progress_metadata(SPLIT_STEP, "float64")
    with pytest.raises(ValueError, match="run_dtype"):
        validated_progress(metadata, "float32")


@pytest.mark.parametrize("step", [-1, TOTAL_STEPS + 1, 10 ** 6])
def test_validated_progress_rejects_out_of_range_steps(step):
    metadata = {"training_step": step, "run_dtype": "float32",
                "next_batch_index": batch_index_for_step(abs(step))}
    with pytest.raises(ValueError):
        validated_progress(metadata, "float32")


# ==========================================================================
# 6. The exact-resume proof, per dtype
# ==========================================================================


@pytest.fixture(scope="module")
def proofs():
    """One integrated proof per dtype, shared across the assertions below.

    Module-scoped because each proof runs the whole schedule four times
    over; every one of them closes its own state, and the lifecycle tests
    build their own so nothing here is measured through a shared fixture."""
    return {dtype: run_dtype_proof(dtype) for dtype in RUN_DTYPES}


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_integrated_proof_passes_every_required_check(dtype, proofs):
    """The whole gate, by name, so a new claim cannot be added to the
    example's output without being added to the check list."""
    assert failed_checks(proofs[dtype]) == []


@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("claim", REQUIRED)
def test_each_required_claim_holds_individually(dtype, claim, proofs):
    """Also asserted one at a time, so a failure names the claim that broke
    instead of the aggregate."""
    assert proofs[dtype][claim] is True, claim


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_resume_reproduces_every_state_family_exactly(dtype, proofs):
    proof = proofs[dtype]
    for claim in ("losses_match", "loss_bits_match", "prefix_matches",
                  "suffix_matches", "first_resumed_loss_matches",
                  "parameters_match", "buffers_match", "moments_match",
                  "counters_match", "optimizer_matches", "generator_matches",
                  "topology_matches", "final_train_logits_match",
                  "final_eval_matches", "predictions_match"):
        assert proof[claim] is True, claim


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_first_resumed_step_produces_equal_gradients(dtype, proofs):
    """Gradients are **not** checkpointed, so the contractual claim is that
    the first resumed forward/backward *produces* gradients equal to the
    corresponding uninterrupted step, compared at that step and before
    either optimizer commits. Nothing here claims they were restored."""
    proof = proofs[dtype]
    assert proof["split_gradients_match"] is True
    assert proof["gradients_nonempty"] is True
    # The claim is about the split step specifically, which is the first
    # step the resumed run executes.
    assert proof["resumed_step"] == SPLIT_STEP


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_fresh_destination_shares_nothing_with_the_saved_run(dtype,
                                                                 proofs):
    proof = proofs[dtype]
    assert proof["fresh_started_different"] is True
    assert proof["fresh_generator"] != proof["saved_generator"]
    assert proof["restored_generator"] == proof["saved_generator"]
    assert proof["identities_preserved"] is True
    assert proof["mode_not_serialized"] is True
    assert proof["load_restored_topology"] is True
    assert proof["topology_is_expected"] is True


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_checkpoint_is_version_three_and_declares_every_dtype(dtype,
                                                                  tmp_path):
    """Read off a **real** archive the proof's own save path produces."""
    model = build_model(dtype)
    optimizer = build_optimizer(model)
    images, targets = build_dataset()
    batches = build_batches(images, targets, dtype)
    path = str(tmp_path / f"integrated_{dtype}.npz")
    try:
        from examples.native_float32_training import train_step

        loss_fn = build_loss()
        for step in range(2):
            train_step(model, loss_fn, optimizer, batches, step, dtype)
        save_native_checkpoint(path, model, optimizer=optimizer,
                               metadata=progress_metadata(2, dtype))
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(bytes(archive["manifest"]).decode("utf-8"))
            arrays = {name: archive[name].dtype for name in archive.files
                      if name != "manifest"}
        assert manifest["format"] == native_checkpoint._FORMAT
        assert manifest["format_version"] == 3
        assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
        # Every model entry declares its dtype, and every declaration is
        # this run's.
        for entry in manifest["model"]["entries"].values():
            assert entry["dtype"] == dtype
        for entry in manifest["optimizer"]["m"] + manifest["optimizer"]["v"]:
            assert entry["dtype"] == dtype
        # ...and the stored payloads physically agree with the manifest.
        assert set(arrays.values()) == {np.dtype(dtype)}
        # The generator section carries the topology, not just the counter.
        section = manifest["generators"]
        assert section["keys"] == [CANONICAL_GENERATOR_KEY]
        assert section["aliases"] == EXPECTED_GENERATOR_ALIASES
        # uint64 fields are canonical decimal strings, never JSON numbers.
        entry = section["entries"][CANONICAL_GENERATOR_KEY]
        assert isinstance(entry["seed"], str)
        assert isinstance(entry["calls"], str)
        assert int(entry["calls"]) == 2 * DROPOUT_CALLS_PER_STEP
    finally:
        for inputs, _ in batches:
            inputs.close()
        _close_model(model, optimizer)


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_a_resume_that_ignores_the_metadata_diverges(dtype, tmp_path):
    """The negative control for the whole resume proof: restarting the
    schedule at step 0 instead of at the validated metadata step produces a
    **different** run. Without this, "the losses matched" could mean the
    schedule position never mattered.

    ``SPLIT_STEP`` is deliberately not a multiple of ``NUM_BATCHES``, which
    is exactly what makes this control non-vacuous: resuming at step 5
    takes batches 2, 0, 1, … while restarting at step 0 takes 0, 1, 2, …,
    so the two sequences genuinely differ. At a split that *was* a multiple
    the two would coincide and this test would pass without proving
    anything — asserted below so that property cannot quietly lapse."""
    from examples.native_float32_training import train_step

    assert SPLIT_STEP % NUM_BATCHES != 0, (
        "the control is vacuous at a split that is a multiple of the "
        "batch count: restarting at 0 would replay the same batches"
    )
    loss_fn = build_loss()
    images, targets = build_dataset()
    batches = build_batches(images, targets, dtype)
    model = build_model(dtype)
    optimizer = build_optimizer(model)
    fresh = fresh_optimizer = None
    path = str(tmp_path / f"divergence_{dtype}.npz")
    try:
        correct = []
        for step in range(TOTAL_STEPS):
            if step == SPLIT_STEP:
                # After SPLIT_STEP *completed* steps, exactly as the example
                # saves it.
                save_native_checkpoint(
                    path, model, optimizer=optimizer,
                    metadata=progress_metadata(SPLIT_STEP, dtype))
            _, pattern = train_step(model, loss_fn, optimizer, batches, step,
                                    dtype)
            if step >= SPLIT_STEP:
                correct.append(pattern)

        fresh = build_model(dtype, fresh=True)
        fresh_optimizer = build_optimizer(fresh)
        load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
        wrong = []
        # Deliberately the wrong schedule position: start from 0.
        for step in range(TOTAL_STEPS - SPLIT_STEP):
            _, pattern = train_step(fresh, loss_fn, fresh_optimizer, batches,
                                    step, dtype)
            wrong.append(pattern)
        assert len(wrong) == len(correct)
        assert wrong != correct
        # ...and resuming at the *validated* step reproduces it exactly,
        # so the divergence above is the schedule and nothing else.
        _close_model(fresh, fresh_optimizer)
        fresh = build_model(dtype, fresh=True)
        fresh_optimizer = build_optimizer(fresh)
        metadata = load_native_checkpoint(path, fresh,
                                          optimizer=fresh_optimizer)
        resumed_step, _ = validated_progress(metadata, dtype)
        assert resumed_step == SPLIT_STEP
        right = [train_step(fresh, loss_fn, fresh_optimizer, batches, step,
                            dtype)[1]
                 for step in range(resumed_step, TOTAL_STEPS)]
        assert right == correct
    finally:
        for inputs, _ in batches:
            inputs.close()
        _close_model(model, optimizer)
        if fresh is not None:
            _close_model(fresh, fresh_optimizer)


# ==========================================================================
# 7. The next-Dropout-mask proof
# ==========================================================================


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_next_dropout_mask_after_a_resume_is_identical(dtype, proofs):
    """Restoration reproduces the next *stochastic event*, not merely the
    generator's serialized integer fields."""
    mask = proofs[dtype]["next_mask"]
    assert mask["mask_bits_match"] is True
    assert mask["calls_match"] is True
    for side in ("uninterrupted", "resumed"):
        assert mask[side]["dtype"] == dtype
        # Non-degenerate: an all-kept mask would match vacuously.
        assert mask[side]["non_degenerate"] is True
        assert 0 < mask[side]["dropped"] < BATCH_SIZE * HIDDEN_FEATURES
        assert mask[side]["dropped"] + mask[side]["kept"] == (
            BATCH_SIZE * HIDDEN_FEATURES)
        # The all-ones probe means the output *is* the multiplier.
        assert mask[side]["kept_value_is_inverted_scale"] is True
        assert mask[side]["consumed_exactly_one_call"] is True


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_shared_alias_path_observes_the_same_advanced_generator(dtype,
                                                                    proofs):
    """The topology, proved by identity and by the counter both moving —
    which is what distinguishes a restored *sharing relationship* from two
    generators that merely hold equal numbers."""
    mask = proofs[dtype]["next_mask"]
    assert mask["both_aliases_shared"] is True
    for side in ("uninterrupted", "resumed"):
        assert mask[side]["alias_is_same_object"] is True
        assert mask[side]["alias_sees_advanced_calls"] is True


def test_a_generator_that_was_not_restored_produces_a_different_mask():
    """The negative control for the mask proof: a model whose generator was
    left at the fresh seed draws a different pattern, so "the masks
    matched" is evidence of restoration rather than of the mask being
    insensitive to the stream."""
    dtype = "float32"
    restored = build_model(dtype)
    other = build_model(dtype, fresh=True)
    probe_values = np.ones((BATCH_SIZE, HIDDEN_FEATURES),
                           dtype=np.dtype(dtype))
    patterns = []
    try:
        for model in (restored, other):
            dropout = getattr(model, CONV_DROPOUT_NAME)
            model.train()
            probe = native_input(probe_values, dtype)
            output = dropout(probe)
            try:
                patterns.append(bits(output.to_numpy(), dtype))
            finally:
                output.close()
                probe.close()
        assert patterns[0] != patterns[1]
    finally:
        _close_model(restored)
        _close_model(other)


# ==========================================================================
# 8. Evaluation and the BatchNorm snapshot family
# ==========================================================================


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_eval_graph_is_independent_of_the_buffers_it_was_built_from(
    dtype, proofs
):
    """The fourth graph-owned saved-resource family. A training forward
    advances all four running buffers underneath an already-built
    evaluation graph, and that graph's backward is unchanged — it answered
    for the forward it recorded, not for the buffers as they are now."""
    snapshots = proofs[dtype]["eval_snapshots"]
    assert snapshots["both_independent"] is True
    assert snapshots["both_advanced_buffers"] is True
    assert snapshots["gradients_match"] is True
    for side in ("uninterrupted", "resumed"):
        assert snapshots[side]["gradients_cleared"] is True
        assert snapshots[side]["control_gradients"], side


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_evaluation_consumes_no_generator_call(dtype, proofs):
    assert proofs[dtype]["eval_consumed_no_call"] is True


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_evaluation_restores_the_callers_previous_mode(dtype):
    model = build_model(dtype)
    loss_fn = build_loss()
    images, targets = build_dataset()
    full = native_input(host_images(images, dtype), dtype)
    try:
        model.train()
        evaluate(model, loss_fn, full, targets, dtype)
        assert model.training is True
        model.eval()
        evaluate(model, loss_fn, full, targets, dtype)
        assert model.training is False
    finally:
        full.close()
        _close_model(model)


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_four_graph_resource_families_are_scoped_honestly(dtype):
    """The claim the example makes, checked rather than repeated: three
    families ride a **training** graph, and the BatchNorm evaluation
    snapshots exist only on an **evaluation** graph — because training-mode
    BatchNorm normalizes with the batch's own statistics and takes no
    snapshot at all. So they are exercised *across* the run, not
    simultaneously in one graph, and the example says exactly that."""
    model = build_model(dtype)
    loss_fn = build_loss()
    images, targets = build_dataset()
    x = native_input(host_images(images, dtype)[:BATCH_SIZE], dtype)
    try:
        # A training graph carries Dropout masks, MaxPool2d winners, and
        # cross-entropy probabilities. It advances the buffers **in the
        # forward** rather than snapshotting them, which is precisely why
        # the fourth family is absent here — there is nothing to snapshot
        # when the batch's own statistics are what normalizes.
        before = {name: buffer.to_numpy().copy()
                  for name, buffer in model.named_buffers()}
        model.train()
        logits = model(x)
        loss = loss_fn(logits, targets[:BATCH_SIZE])
        for name, buffer in model.named_buffers():
            assert not np.array_equal(buffer.to_numpy(), before[name]), name
        loss.backward()
        for parameter in model.parameters():
            grad = parameter.grad
            parameter.zero_grad()
            grad.close()
        loss.close()
        logits.close()
    finally:
        x.close()
        _close_model(model)
    source = EXAMPLE.read_text(encoding="utf-8")
    flattened = " ".join(source.split())
    assert "across" in flattened and "coexist" in flattened, (
        "the example must state whether the four families coexist in one "
        "graph or are exercised across the run"
    )


# ==========================================================================
# 9. Ownership and lifecycle
# ==========================================================================


def test_a_complete_integrated_proof_returns_storage_to_baseline(
    live_storages
):
    """Both dtypes, both runs each, plus the mask and snapshot proofs —
    and the live count returns exactly to where it started. Explicit
    ``close()`` is the release mechanism; the collection below only settles
    the two documented reference cycles (``zero_grad`` dropping gradient
    objects, and the Python-managed graph holding parents through backward
    closures)."""
    gc.collect()
    baseline = len(live_storages)
    for dtype in RUN_DTYPES:
        run_dtype_proof(dtype)
    gc.collect()
    assert len(live_storages) == baseline


def test_repeated_uninterrupted_runs_return_to_baseline(live_storages):
    gc.collect()
    baseline = len(live_storages)
    for _ in range(2):
        run_uninterrupted("float32")
        gc.collect()
        assert len(live_storages) == baseline


@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_the_helpers_return_no_live_native_object(dtype):
    """Every public helper that represents a completed run returns plain
    Python values only, so a test can hold the result without holding
    native storage."""
    def _scan(value, path="result"):
        for banned in (NativeTensor, cpp.NativeStorage,
                       cpp.NativeTensorCore, NativeGenerator, NativeAdam):
            assert not isinstance(value, banned), f"{path} is a {banned}"
        if isinstance(value, dict):
            for key, item in value.items():
                _scan(item, f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                _scan(item, f"{path}[{index}]")

    _scan(run_uninterrupted(dtype))


def test_snapshot_helpers_close_every_caller_owned_tensor(live_storages):
    """``state_dict()`` and the optimizer's ``state_dict()`` both return
    **caller-owned** snapshots; a reporting helper must never leak them."""
    model = build_model("float32")
    optimizer = build_optimizer(model)
    try:
        gc.collect()
        baseline = len(live_storages)
        for _ in range(3):
            model_bits(model, "float32")
            model_dtypes(model)
            optimizer_bits(optimizer, "float32")
            alias_topology(model)
            generator_state(model)
        gc.collect()
        assert len(live_storages) == baseline
    finally:
        _close_model(model, optimizer)


# ==========================================================================
# 10. The example as a program
# ==========================================================================


def test_example_runs_as_a_subprocess_and_exits_zero():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True, text=True,
        cwd=str(EXAMPLE.resolve().parent.parent),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exact deterministic resume at float32: yes" in result.stdout
    assert "exact deterministic resume at float64: yes" in result.stdout
    assert "live native storage baseline / final: 0 / 0" in result.stdout


def test_the_example_leaves_no_checkpoint_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dtype_proof("float32")
    assert list(tmp_path.iterdir()) == []


def test_the_example_claims_no_timing_and_imports_no_test_framework():
    source = EXAMPLE.read_text(encoding="utf-8")
    for banned in ("import pytest", "import unittest", "time.perf_counter",
                   "timeit", "speedup", "faster than", "benchmark("):
        assert banned not in source, banned
    for dependency in ("import torch", "import pandas", "sklearn",
                       "matplotlib", "requests", "urllib"):
        assert dependency not in source, dependency


def test_the_example_makes_no_cross_dtype_numerical_claim():
    """§18.3: a float32 run does not need to produce the same numbers as a
    float64 run, and no part of the proof may depend on it. Checked
    structurally — no comparison in the file may span the two records."""
    source = EXAMPLE.read_text(encoding="utf-8")
    flattened = " ".join(source.split())
    assert "never compared" in flattened or "only against itself" in flattened
    # Scanned as *calls*, not as words: the file's own prose says it never
    # uses ``allclose``, and a word-level scan would fail on that sentence
    # while missing ``np.allclose(...)`` written any other way.
    for banned in ("allclose(", "isclose(", "atol=", "rtol=", "approx("):
        assert banned not in source, banned


def test_the_example_states_what_the_checkpoint_does_not_capture():
    source = EXAMPLE.read_text(encoding="utf-8")
    flattened = " ".join(source.split()).lower()
    for absent in ("data-loader", "batch order", "global rng"):
        assert absent in flattened, absent


def test_main_runs_and_reports_both_dtypes(capsys):
    main()
    out = capsys.readouterr().out
    for dtype in RUN_DTYPES:
        assert f"run dtype: {dtype}" in out
        assert f"exact deterministic resume at {dtype}: yes" in out
    assert "version 3" in out
    assert "uint32 at float32" in out


# ==========================================================================
# 11. The I9 scope boundary
# ==========================================================================


def test_i9_moved_the_public_registry_and_nothing_else():
    from tensorforge.experimental import native_optimizer_state

    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert native_optimizer_state.FORMAT_VERSION == 1


def test_the_example_defines_no_public_training_api():
    """It is an executable correctness proof, not a framework: nothing in
    it is importable as a TensorForge capability."""
    import tensorforge.experimental as experimental

    for name in ("NativeFloat32Classifier", "run_dtype_proof",
                 "run_uninterrupted", "native_float32_training"):
        assert not hasattr(experimental, name), name
        assert name not in experimental.__all__, name


def test_the_stable_framework_is_untouched():
    import tensorforge

    for name in ("NativeFloat32Classifier", "NativeTensor", "NativeAdam"):
        assert not hasattr(tensorforge, name), name
