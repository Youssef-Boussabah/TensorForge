"""Tests for the native normalization benchmark harness (Phase F,
milestone F7).

These validate the benchmark's *behavior* — the nine-case inventory, the
correctness-before-timing rule, the reference labelling (including why
the three ``NativeBatchNorm2d`` cases are ``native_only`` for timing), the
JSON schema, cleanup and ownership, the NumPy boundary of the timed
training step, and the CLI — with **no timing or threshold assertions of
any kind**. Benchmark durations are hardware-specific measurements, never
pass/fail criteria, and nothing here depends on one path taking more or
less time than another. Importing the benchmark module must run nothing.

Selector: python -m pytest -q -k native_normalization_benchmark
"""

import gc
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from benchmarks import benchmark_native_normalization as bench

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = (REPO_ROOT / "benchmarks"
                  / "benchmark_native_normalization.py")
TEST_FILE = Path(__file__)

SMOKE = {"warmup": 1, "repetitions": 3, "smoke": True}

EXPECTED_CASES = (
    "layernorm_forward",
    "layernorm_backward",
    "batchnorm1d_training_forward",
    "batchnorm1d_eval_forward",
    "batchnorm1d_backward",
    "batchnorm2d_training_forward",
    "batchnorm2d_eval_forward",
    "batchnorm2d_backward",
    "normalized_training_step",
)

BATCHNORM2D_CASES = (
    "batchnorm2d_training_forward",
    "batchnorm2d_eval_forward",
    "batchnorm2d_backward",
)

STABLE_CASES = tuple(name for name in EXPECTED_CASES
                     if name not in BATCHNORM2D_CASES)


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open."""
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


def _by_name(payload):
    return {record["case"]: record for record in payload["cases"]}


# --------------------------------------------------------------------------
# Import safety and the case registry
# --------------------------------------------------------------------------

def test_importing_the_module_runs_nothing(capsys):
    import importlib

    importlib.reload(bench)
    assert capsys.readouterr().out == ""


def test_case_inventory_is_exactly_the_f7_set():
    assert tuple(bench.CASES) == EXPECTED_CASES
    categories = {name: spec["category"] for name, spec in bench.CASES.items()}
    assert categories == {
        "layernorm_forward": "layer_normalization",
        "layernorm_backward": "layer_normalization",
        "batchnorm1d_training_forward": "batch_normalization_1d",
        "batchnorm1d_eval_forward": "batch_normalization_1d",
        "batchnorm1d_backward": "batch_normalization_1d",
        "batchnorm2d_training_forward": "batch_normalization_2d",
        "batchnorm2d_eval_forward": "batch_normalization_2d",
        "batchnorm2d_backward": "batch_normalization_2d",
        "normalized_training_step": "training_step",
    }
    # Each case names the operation it actually measures.
    assert "NativeLayerNorm" in bench.CASES["layernorm_forward"]["operation"]
    assert "backward()" in bench.CASES["layernorm_backward"]["operation"]
    for name in ("batchnorm1d_training_forward", "batchnorm1d_backward"):
        assert "NativeBatchNorm1d" in bench.CASES[name]["operation"], name
    assert "training mode" in (
        bench.CASES["batchnorm1d_training_forward"]["operation"]
    )
    assert "evaluation mode" in (
        bench.CASES["batchnorm1d_eval_forward"]["operation"]
    )
    for name in BATCHNORM2D_CASES:
        assert "NativeBatchNorm2d" in bench.CASES[name]["operation"], name
    step = bench.CASES["normalized_training_step"]["operation"]
    for piece in ("zero_grad", "BatchNorm1d", "LayerNorm", "NativeMSELoss",
                  "backward", "NativeAdam.step()"):
        assert piece in step, piece
    # The benchmark stays scoped: no unrelated or unshipped case sneaks in.
    for name in bench.CASES:
        for banned in ("dropout", "batchnorm3d", "checkpoint", "state_dict",
                       "cuda", "float32"):
            assert banned not in name.lower(), (name, banned)


def test_every_case_declares_deterministic_configurations_and_a_seed():
    seeds = set()
    for name, spec in bench.CASES.items():
        assert set(spec["configurations"]) == {"full", "smoke"}, name
        for variant, config in spec["configurations"].items():
            shape = config["shape"]
            assert shape and all(isinstance(d, int) and d > 0 for d in shape), (
                name, variant
            )
        smoke = spec["configurations"]["smoke"]["shape"]
        full = spec["configurations"]["full"]["shape"]
        assert len(smoke) == len(full), name
        assert all(s <= f for s, f in zip(smoke, full)), name
        assert isinstance(spec["seed"], int)
        assert not isinstance(spec["seed"], bool)
        seeds.add(spec["seed"])
        assert spec["notes"].strip(), name
        assert isinstance(spec["eps"], float) and spec["eps"] > 0, name
    # Distinct seeds, so no two cases silently share their sampled data.
    assert len(seeds) == len(bench.CASES)


def test_layernorm_configurations_exercise_multi_axis_normalization():
    for name in ("layernorm_forward", "layernorm_backward"):
        spec = bench.CASES[name]
        for variant, config in spec["configurations"].items():
            normalized = config["normalized_shape"]
            assert len(normalized) == 2, (name, variant)
            assert config["shape"][-2:] == normalized, (name, variant)


def test_batchnorm2d_smoke_shape_uses_unequal_channel_and_spatial_dims():
    """Unequal C/H/W in smoke mode means an accidental channel/spatial
    broadcast mistake cannot hide behind a coincidental shape match."""
    for name in BATCHNORM2D_CASES:
        _n, c, h, w = bench.CASES[name]["configurations"]["smoke"]["shape"]
        assert len({c, h, w}) == 3, name
        assert bench.CASES[name]["reduction_axes"] == (0, 2, 3), name


def test_the_training_step_reuses_the_f6_architecture_and_dataset():
    from examples import native_normalization_training as example

    spec = bench.CASES["normalized_training_step"]
    # One real F6 iteration, not a scaling study: the shapes match.
    assert (spec["configurations"]["full"]["shape"]
            == spec["configurations"]["smoke"]["shape"] == (8, 2))
    assert spec["momentum"] == example.MOMENTUM
    assert bench.DEFAULT_LR == example.DEFAULT_LR
    inputs, targets = bench.build_dataset()
    assert inputs == example.X_VALUES and targets == example.Y_VALUES
    assert len(inputs) == spec["configurations"]["full"]["shape"][0]


def test_reference_labels_are_honest():
    allowed = {bench.STABLE, bench.NATIVE_ONLY}
    for name, spec in bench.CASES.items():
        assert spec["reference_type"] in allowed, name
        assert spec["reference_detail"].strip(), name
        assert spec["correctness_reference"].strip(), name
    # The six cases with a real stable counterpart say which module it is.
    for name in STABLE_CASES:
        spec = bench.CASES[name]
        assert spec["reference_type"] == bench.STABLE, name
        assert "tensorforge" in spec["reference_detail"], name
    # ...and the three BatchNorm2d cases are native_only, because the
    # stable line has no public BatchNorm2d.
    for name in BATCHNORM2D_CASES:
        spec = bench.CASES[name]
        assert spec["reference_type"] == bench.NATIVE_ONLY, name
        detail = spec["reference_detail"].lower()
        assert "no public batchnorm2d" in detail, name
        assert "misleading" in detail, name
        # ...while still naming a real correctness oracle.
        assert spec["correctness_reference"].strip(), name
    from tensorforge import nn

    assert not hasattr(nn, "BatchNorm2d"), (
        "a public stable BatchNorm2d now exists; the native_only timing "
        "label for the BatchNorm2d cases must be revisited"
    )


def test_the_batchnorm2d_backward_oracle_is_described_and_scoped():
    """The transformed stable computation is a correctness oracle only —
    the case must say so, and must not present it as a timed reference."""
    spec = bench.CASES["batchnorm2d_backward"]
    oracle = spec["correctness_reference"].lower()
    assert "batchnorm1d" in oracle
    assert "(n*h*w, c)" in oracle
    assert "nchw" in oracle
    assert "never timed" in oracle
    assert "misleading" in spec["notes"].lower()


# --------------------------------------------------------------------------
# The smoke run and the payload schema
# --------------------------------------------------------------------------

@needs_native
def test_smoke_run_produces_the_documented_schema():
    payload = bench.run_benchmark(**SMOKE)
    assert set(payload) == {"benchmark", "version", "mode", "environment",
                            "cases"}
    assert payload["benchmark"] == "tensorforge.native_normalization"
    assert payload["version"] == "1.0"
    assert payload["mode"] == "smoke"
    environment = payload["environment"]
    for key in ("python_version", "platform", "machine", "processor",
                "tensorforge_version", "native_backend", "dtype", "device",
                "scope", "timer", "primary_statistic", "warmup",
                "repetitions", "training_step_repetitions",
                "backward_repetitions", "timestamp"):
        assert key in environment, key
    assert environment["dtype"] == "float64" and environment["device"] == "cpu"
    assert environment["timer"] == "time.perf_counter_ns"
    assert environment["primary_statistic"] == "median"
    backend = environment["native_backend"]
    for key in ("name", "tensor_core", "available", "native_autograd",
                "stable_framework_integration"):
        assert key in backend, key
    assert backend["available"] is True
    assert backend["native_autograd"] is True
    assert backend["stable_framework_integration"] is False
    assert "normalization" in environment["scope"]
    # Every case appears exactly once, in the declared order.
    assert [record["case"] for record in payload["cases"]] == list(
        EXPECTED_CASES
    )
    for record in payload["cases"]:
        for key in ("case", "category", "operation", "mode", "configuration",
                    "shape", "normalized_shape", "reduction_axes", "eps",
                    "momentum", "seed", "reference_type", "reference_detail",
                    "correctness_reference", "correctness", "warmup",
                    "sample_count", "native", "reference",
                    "native_to_reference_ratio", "notes"):
            assert key in record, (record["case"], key)


@needs_native
def test_smoke_configurations_are_used_in_smoke_mode():
    payload = bench.run_benchmark(**SMOKE)
    for record in payload["cases"]:
        spec = bench.CASES[record["case"]]
        assert tuple(record["shape"]) == spec["configurations"]["smoke"]["shape"]
    full = bench.run_benchmark(cases=["batchnorm1d_eval_forward"], warmup=1,
                               repetitions=1, smoke=False)
    assert tuple(full["cases"][0]["shape"]) == (256, 64)
    assert full["mode"] == "full"


@needs_native
def test_sample_and_warmup_counts_match_the_requested_mode():
    payload = bench.run_benchmark(**SMOKE)
    for record in payload["cases"]:
        assert record["warmup"] == SMOKE["warmup"]
        assert record["sample_count"] == SMOKE["repetitions"]
        assert record["native"]["sample_count"] == record["sample_count"]
    assert payload["environment"]["warmup"] == SMOKE["warmup"]
    assert payload["environment"]["repetitions"] == SMOKE["repetitions"]
    # The declared defaults, and the per-case caps, are honest.
    assert bench.SMOKE_DEFAULTS == {"warmup": 1, "repetitions": 3}
    assert bench.DEFAULTS["warmup"] >= 3
    assert 10 <= bench.DEFAULTS["repetitions"] <= 20
    assert bench.TRAINING_STEP_REPETITIONS <= bench.DEFAULTS["repetitions"]
    assert bench.BACKWARD_REPETITIONS <= bench.DEFAULTS["repetitions"]


@needs_native
def test_case_specific_repetition_caps_are_reported_accurately():
    """When more repetitions are requested than a heavy case allows, the
    cap is applied *and* reported — never silently."""
    requested = 12
    payload = bench.run_benchmark(warmup=1, repetitions=requested, smoke=True)
    environment = payload["environment"]
    assert environment["repetitions"] == requested
    assert environment["training_step_repetitions"] == (
        bench.TRAINING_STEP_REPETITIONS
    )
    assert environment["backward_repetitions"] == bench.BACKWARD_REPETITIONS
    for record in payload["cases"]:
        expected = min(requested,
                       bench.CASES[record["case"]].get("repetitions",
                                                       requested))
        assert record["sample_count"] == expected, record["case"]
        assert len(record["native"]["samples_s"]) == expected, record["case"]


# --------------------------------------------------------------------------
# Timing record structure. Nothing here compares a duration or a ratio
# against a nonzero constant.
# --------------------------------------------------------------------------

@needs_native
def test_timing_fields_are_finite_non_negative_and_complete():
    payload = bench.run_benchmark(**SMOKE)
    for record in payload["cases"]:
        native = record["native"]
        assert native["units"] == "seconds_per_call"
        assert len(native["samples_s"]) == native["sample_count"]
        assert native["sample_count"] == record["sample_count"]
        for value in native["samples_s"]:
            assert math.isfinite(value) and value >= 0.0
        assert native["min_s"] <= native["median_s"] <= native["max_s"]
        assert native["spread_s"] == pytest.approx(
            native["max_s"] - native["min_s"]
        )
        assert math.isfinite(native["relative_spread"])
        assert native["relative_spread"] >= 0.0
        assert native["median_s"] == pytest.approx(
            float(np.median(native["samples_s"]))
        )
        reference = record["reference"]
        if record["reference_type"] == bench.NATIVE_ONLY:
            assert reference is None, record["case"]
            assert record["native_to_reference_ratio"] is None, record["case"]
        else:
            assert reference is not None, record["case"]
            assert len(reference["samples_s"]) == reference["sample_count"]
            assert reference["sample_count"] == record["sample_count"]
            assert reference["min_s"] <= reference["median_s"] <= reference["max_s"]
            ratio = record["native_to_reference_ratio"]
            assert math.isfinite(ratio) and ratio > 0.0


@needs_native
def test_no_sample_is_discarded_and_one_call_is_timed_per_sample():
    calls = []
    samples = bench.measure(
        prepare=lambda: calls.append("prepare"),
        run=lambda _state: calls.append("run"),
        cleanup=lambda _state, _result: calls.append("cleanup"),
        warmup=2, repetitions=5,
    )
    assert len(samples) == 5
    assert calls.count("run") == 7          # 2 warm-up + 5 measured
    assert calls.count("prepare") == 7      # ...each with its own setup
    assert calls.count("cleanup") == 7      # ...and its own cleanup
    assert calls[:3] == ["prepare", "run", "cleanup"]


# --------------------------------------------------------------------------
# The correctness gates: every case passes, and each gates what it promises
# --------------------------------------------------------------------------

@needs_native
def test_every_case_reports_a_passed_correctness_gate():
    payload = bench.run_benchmark(**SMOKE)
    for record in payload["cases"]:
        correctness = record["correctness"]
        assert correctness["status"] == "passed", record["case"]
        checks = correctness["checks"]
        assert len(checks) >= 5, record["case"]
        assert all(isinstance(check, str) and check for check in checks)
        assert len(set(checks)) == len(checks), record["case"]
        error = correctness["max_abs_error"]
        assert isinstance(error, float) and math.isfinite(error)
        assert error >= 0.0


@needs_native
def test_layernorm_gates_cover_multi_axis_parity_and_gradients():
    payload = bench.run_benchmark(
        cases=["layernorm_forward", "layernorm_backward"], **SMOKE
    )
    records = _by_name(payload)
    forward = records["layernorm_forward"]["correctness"]
    for expected in ("numpy_formula_parity", "stable_parity",
                     "mode_independent", "no_parameter_mutation",
                     "stateless"):
        assert expected in forward["checks"], expected
    assert math.isfinite(forward["native_vs_stable_max_abs_error"])
    backward = records["layernorm_backward"]["correctness"]
    for expected in ("input_gradient_present", "affine_gradients_present",
                     "gradient_shapes", "stable_parity", "graph_released",
                     "no_graph_resource_survives"):
        assert expected in backward["checks"], expected
    for key in ("input_gradient_max_abs_error", "weight_gradient_max_abs_error",
                "bias_gradient_max_abs_error"):
        assert math.isfinite(backward[key]), key


@needs_native
def test_batchnorm1d_gates_cover_running_state_and_mode_behavior():
    payload = bench.run_benchmark(
        cases=["batchnorm1d_training_forward", "batchnorm1d_eval_forward",
               "batchnorm1d_backward"],
        **SMOKE,
    )
    records = _by_name(payload)
    training = records["batchnorm1d_training_forward"]["correctness"]
    for expected in ("population_statistics_parity", "running_mean_parity",
                     "running_var_parity", "both_buffers_advanced",
                     "parameter_versions_unchanged", "identities_preserved",
                     "stable_parity", "no_input_mutation"):
        assert expected in training["checks"], expected
    assert math.isfinite(training["running_state_max_abs_error"])
    evaluation = records["batchnorm1d_eval_forward"]["correctness"]
    for expected in ("numpy_formula_parity", "no_running_state_mutation",
                     "graph_holds_snapshots_not_buffers",
                     "snapshots_own_their_storage", "stable_parity"):
        assert expected in evaluation["checks"], expected
    assert evaluation["adopted_snapshot_count"] >= 2
    backward = records["batchnorm1d_backward"]["correctness"]
    for expected in ("reference_parity", "no_buffer_gradients",
                     "backward_does_not_advance_state", "graph_released"):
        assert expected in backward["checks"], expected
    for key in ("input_gradient_max_abs_error", "gamma_gradient_max_abs_error",
                "beta_gradient_max_abs_error"):
        assert math.isfinite(backward[key]), key


@needs_native
def test_batchnorm2d_gates_are_real_despite_having_no_timed_reference():
    payload = bench.run_benchmark(cases=list(BATCHNORM2D_CASES), **SMOKE)
    records = _by_name(payload)
    training = records["batchnorm2d_training_forward"]["correctness"]
    for expected in ("population_statistics_parity", "running_mean_parity",
                     "running_var_parity", "both_buffers_advanced",
                     "channelwise_affine", "identities_preserved"):
        assert expected in training["checks"], expected
    assert math.isfinite(training["channelwise_affine_max_abs_error"])
    # No stable comparison is claimed for a native_only case.
    assert "stable_parity" not in training["checks"]
    evaluation = records["batchnorm2d_eval_forward"]["correctness"]
    for expected in ("numpy_formula_parity", "no_running_state_mutation",
                     "graph_holds_snapshots_not_buffers",
                     "owning_contiguous_output"):
        assert expected in evaluation["checks"], expected
    assert "stable_parity" not in evaluation["checks"]
    backward = records["batchnorm2d_backward"]["correctness"]
    for expected in ("reference_parity", "no_buffer_gradients",
                     "backward_does_not_advance_state", "graph_released"):
        assert expected in backward["checks"], expected
    for key in ("input_gradient_max_abs_error", "gamma_gradient_max_abs_error",
                "beta_gradient_max_abs_error"):
        assert math.isfinite(backward[key]), key
    for record in payload["cases"]:
        assert record["reference"] is None
        assert record["native_to_reference_ratio"] is None


@needs_native
def test_the_batchnorm2d_oracle_really_is_equivalent():
    """The transformation the BatchNorm2d gate relies on, checked here in
    the open: reducing the ``(N*H*W, C)`` sample matrix over axis 0 is the
    NCHW reduction over N, H, and W."""
    values = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    matrix = bench._nchw_to_samples(values)
    assert matrix.shape == (2 * 4 * 5, 3)
    assert np.array_equal(bench._samples_to_nchw(matrix, values.shape), values)
    assert np.allclose(matrix.mean(axis=0), values.mean(axis=(0, 2, 3)))
    assert np.allclose(matrix.var(axis=0), values.var(axis=(0, 2, 3)))


@needs_native
def test_the_training_step_gate_covers_parameters_state_and_cleanup():
    payload = bench.run_benchmark(cases=["normalized_training_step"], **SMOKE)
    step = payload["cases"][0]["correctness"]
    for expected in ("scalar_finite_loss", "all_gradients_present",
                     "buffers_excluded_from_optimizer",
                     "running_state_advanced", "optimizer_state_advanced",
                     "parameter_updated", "graph_released",
                     "no_batchnorm_graph_resource_survives",
                     "transients_closed", "stable_loss_parity",
                     "stable_gradient_parity", "stable_parameter_parity",
                     "stable_running_state_parity",
                     "no_checkpoint_or_reporting_work"):
        assert expected in step["checks"], expected
    assert step["updated_parameters"] == [
        "batch_norm.beta", "batch_norm.gamma", "hidden.bias", "hidden.weight",
        "layer_norm.bias", "layer_norm.weight", "output.bias", "output.weight",
    ]
    for key in ("loss_abs_error", "gradient_max_abs_error",
                "running_state_max_abs_error"):
        assert math.isfinite(step[key]), key


@needs_native
def test_correctness_tolerances_are_agreement_bounds_not_budgets():
    for name in ("FORWARD_ATOL", "GRADIENT_ATOL", "STATE_ATOL", "LOSS_ATOL",
                 "PARAMETER_ATOL"):
        value = getattr(bench, name)
        assert isinstance(value, float)
        assert 0 < value < 1e-6, name


# --------------------------------------------------------------------------
# The deliberately broken gate: timing must never begin
# --------------------------------------------------------------------------

@needs_native
def test_a_finite_but_wrong_native_result_aborts_before_timing(monkeypatch):
    """The gate is real: a native forward that returns a correctly shaped,
    finite, but numerically wrong result must raise before any timing
    happens and publish nothing."""
    from tensorforge.experimental import NativeLayerNorm, NativeTensor

    timed = []
    original_measure = bench.measure

    def tracking_measure(*args, **kwargs):
        timed.append(args)
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(bench, "measure", tracking_measure)
    # Same shape, entirely finite, and not layer normalization: only the
    # reference comparison can catch it.
    monkeypatch.setattr(
        NativeLayerNorm, "forward",
        lambda self, x: NativeTensor.from_array(np.zeros(x.shape)),
    )
    with pytest.raises(AssertionError, match="differs from the reference"):
        bench.run_benchmark(cases=["layernorm_forward"], **SMOKE)
    assert timed == [], "timing ran despite a failed correctness gate"


@needs_native
def test_a_wrong_batchnorm_running_update_aborts_before_timing(monkeypatch):
    """The stateful half of the same rule: correct output, correctly
    shaped and finite running statistics, wrong momentum blend."""
    from tensorforge.experimental import native_batchnorm

    timed = []
    original_measure = bench.measure
    original_blend = native_batchnorm._NativeBatchNorm._blend

    def tracking_measure(*args, **kwargs):
        timed.append(args)
        return original_measure(*args, **kwargs)

    def wrong_blend(self, current, statistic, like, track):
        # A finite, correctly shaped, but wrong replacement value: the
        # batch statistic with no momentum blending at all.
        detached = track(statistic.detach())
        return track(detached.reshape((self.num_features,)))

    monkeypatch.setattr(bench, "measure", tracking_measure)
    monkeypatch.setattr(native_batchnorm._NativeBatchNorm, "_blend",
                        wrong_blend)
    with pytest.raises(AssertionError, match="differs from the reference"):
        bench.run_benchmark(cases=["batchnorm1d_training_forward"], **SMOKE)
    assert timed == [], "timing ran despite a failed correctness gate"


@needs_native
def test_a_non_finite_native_result_is_caught_by_the_gate(monkeypatch):
    from tensorforge.experimental import NativeLayerNorm, NativeTensor

    timed = []
    original_measure = bench.measure
    monkeypatch.setattr(
        bench, "measure",
        lambda *a, **k: (timed.append(a), original_measure(*a, **k))[1],
    )
    monkeypatch.setattr(
        NativeLayerNorm, "forward",
        lambda self, x: NativeTensor.from_array(np.full(x.shape, np.inf)),
    )
    with pytest.raises(AssertionError, match="not finite"):
        bench.run_benchmark(cases=["layernorm_forward"], **SMOKE)
    assert timed == []


@needs_native
def test_cli_reports_a_correctness_failure_with_a_nonzero_exit(monkeypatch,
                                                              capsys):
    def broken_build(config, spec):
        raise AssertionError("synthetic gate failure")

    monkeypatch.setitem(bench.CASES["batchnorm2d_eval_forward"], "build",
                        broken_build)
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--smoke", "--case", "batchnorm2d_eval_forward"])
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "correctness gate failed" in captured.err
    assert "synthetic gate failure" in captured.err
    assert captured.out == ""      # no timing published for a failed case


def test_the_gate_runs_before_the_timer_structurally():
    """Reading the harness: ``check()`` is called before ``measure`` in
    ``_measure_case``, so a raising gate can never reach a timer."""
    import inspect

    source = inspect.getsource(bench._measure_case)
    assert source.index('case["check"]()') < source.index("measure(")


# --------------------------------------------------------------------------
# CLI, JSON, and the human report
# --------------------------------------------------------------------------

@needs_native
def test_cli_json_smoke_output_parses_and_keeps_stdout_clean(capsys):
    bench.main(["--smoke", "--json", "--warmup", "1", "--repetitions", "2"])
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert output.lstrip().startswith("{")     # JSON only, no banner
    assert payload["benchmark"] == "tensorforge.native_normalization"
    assert payload["mode"] == "smoke"
    assert [record["case"] for record in payload["cases"]] == list(
        EXPECTED_CASES
    )
    # Round-trips: every value is JSON-native, nothing custom leaked in.
    assert json.loads(json.dumps(payload)) == payload


@needs_native
def test_cli_human_output_carries_the_local_characterization_disclaimer(capsys):
    bench.main(["--smoke", "--warmup", "1", "--repetitions", "2"])
    output = capsys.readouterr().out
    assert "TensorForge native normalization benchmark" in output
    assert "{" not in output       # no JSON wrapper in human mode
    for case in EXPECTED_CASES:
        assert case in output, case
    # Whitespace-normalized, so the guard survives line rewrapping.
    lowered = re.sub(r"\s+", " ", output.lower())
    for phrase in ("local characterization only",
                   "not a performance contract",
                   "one machine, one build, and one workload",
                   "not cross-machine",
                   "observations",
                   "correctness is gated before timing",
                   "timing is never a pass/fail criterion",
                   "no test or ci job"):
        assert phrase in lowered, phrase
    # No marketing and no speed verdict, anywhere in the report.
    for banned in ("faster", "fastest", "slower", "speedup", "outperform",
                   "beats", "wins", "competitive", "production-ready",
                   "pytorch", "gpu"):
        assert banned not in lowered, banned
    # native_only cases publish no ratio in the human report either.
    for line in output.splitlines():
        if line.startswith("batchnorm2d_"):
            assert "n/a" in line, line


@needs_native
def test_single_case_selection():
    payload = bench.run_benchmark(cases=["batchnorm2d_backward"], **SMOKE)
    assert [record["case"] for record in payload["cases"]] == [
        "batchnorm2d_backward"
    ]
    assert payload["cases"][0]["reduction_axes"] == [0, 2, 3]
    assert payload["cases"][0]["reference_type"] == bench.NATIVE_ONLY


def test_cli_rejects_unknown_and_invalid_arguments():
    with pytest.raises(SystemExit):
        bench.main(["--case", "nope"])
    with pytest.raises(SystemExit):
        bench.main(["--warmup", "not-a-number"])
    with pytest.raises(ValueError, match="unknown case"):
        bench.run_benchmark(cases=["nope"], **SMOKE)
    with pytest.raises(ValueError, match="unknown case"):
        bench.run_benchmark(cases=["batchnorm3d_forward"], **SMOKE)


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "3", None])
def test_non_positive_or_non_int_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        bench.run_benchmark(warmup=bad, smoke=True)
    with pytest.raises(ValueError):
        bench.run_benchmark(repetitions=bad, smoke=True)


def test_unbuilt_backend_follows_the_benchmark_convention(monkeypatch, capsys):
    monkeypatch.setattr(cpp, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not built"):
        bench.run_benchmark(**SMOKE)
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--smoke"])
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "not built" in captured.err
    assert captured.out == ""


# --------------------------------------------------------------------------
# No result file is ever written
# --------------------------------------------------------------------------

@needs_native
def test_running_the_benchmark_writes_no_files():
    watched = (REPO_ROOT, REPO_ROOT / "benchmarks", REPO_ROOT / "docs")
    before = {path: {entry.name for entry in path.iterdir()}
              for path in watched}
    bench.run_benchmark(cases=["layernorm_forward",
                               "normalized_training_step"], **SMOKE)
    for path in watched:
        assert {entry.name for entry in path.iterdir()} == before[path], path
    for pattern in ("*.json", "*.csv", "*.png", "*.svg", "benchmark*.*"):
        assert not list((REPO_ROOT / "benchmarks").glob(pattern)) or (
            pattern == "benchmark*.*"
        ), pattern
    assert not list(REPO_ROOT.glob("*.json"))
    assert not list(REPO_ROOT.glob("*.csv"))
    assert not list((REPO_ROOT / "benchmarks").glob("*.json"))
    assert not list((REPO_ROOT / "benchmarks").glob("*.csv"))


@needs_native
def test_the_cli_writes_no_files(capsys):
    before = {entry.name for entry in REPO_ROOT.iterdir()}
    bench.main(["--smoke", "--json", "--case", "batchnorm1d_eval_forward"])
    capsys.readouterr()
    assert {entry.name for entry in REPO_ROOT.iterdir()} == before


def test_the_benchmark_opens_no_file_and_imports_no_writer():
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    for banned in ("open(", "Path(", "os.makedirs", "savefig", "to_csv",
                   "csv.writer", "np.save", "json.dump(", "matplotlib",
                   "tempfile", "shutil"):
        assert banned not in source, banned
    # json.dumps (a string) is fine; json.dump (a file) is not.
    assert "json.dumps(payload)" in source


# --------------------------------------------------------------------------
# Ownership: repeated runs leak no native storage
# --------------------------------------------------------------------------

@needs_native
def test_repeated_smoke_runs_do_not_grow_live_native_storage(live_storages):
    """Each run builds its own modules, inputs, and graphs and releases
    them when the case closes, so the live count returns to the same
    baseline every time — the invariant is that nothing accumulates."""
    cases = ["layernorm_backward", "batchnorm1d_training_forward",
             "batchnorm2d_eval_forward", "normalized_training_step"]
    bench.run_benchmark(cases=cases, **SMOKE)
    gc.collect()
    baseline = len(live_storages)
    for _ in range(3):
        bench.run_benchmark(cases=cases, **SMOKE)
        gc.collect()
        assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("case", list(EXPECTED_CASES))
def test_each_case_returns_live_storage_to_its_baseline(case, live_storages):
    bench.run_benchmark(cases=[case], **SMOKE)
    gc.collect()
    baseline = len(live_storages)
    bench.run_benchmark(cases=[case], **SMOKE)
    gc.collect()
    assert len(live_storages) == baseline


@needs_native
def test_the_training_step_case_closes_its_parameters_and_buffers():
    """There is no ``NativeModule.close()``, so the case must close both
    traversals explicitly — proved by inspecting the objects it left."""
    spec = bench.CASES["normalized_training_step"]
    case = spec["build"](spec["configurations"]["smoke"], spec)
    state = case["native_prepare"]()
    model, optimizer = state
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    assert parameters and buffers
    result = case["native_run"](state)
    assert all(parameter.grad is not None for parameter in parameters)
    case["native_cleanup"](state, result)
    case["close"]()
    # Both traversals were closed, and the gradients were released before
    # their parameters were (reading .grad afterwards would raise).
    assert all(parameter.closed for parameter in parameters)
    assert all(buffer.closed for buffer in buffers)
    assert all(parameter._grad is None for parameter in parameters)
    assert optimizer.closed
    assert result[0].closed and result[1].closed


@needs_native
def test_a_failed_gate_still_releases_the_case(live_storages, monkeypatch):
    """The ``finally: case["close"]()`` in ``_measure_case`` is real: an
    aborted case leaks nothing."""
    from tensorforge.experimental import NativeLayerNorm, NativeTensor

    bench.run_benchmark(cases=["layernorm_forward"], **SMOKE)
    gc.collect()
    baseline = len(live_storages)
    monkeypatch.setattr(
        NativeLayerNorm, "forward",
        lambda self, x: NativeTensor.from_array(np.zeros(x.shape)),
    )
    with pytest.raises(AssertionError):
        bench.run_benchmark(cases=["layernorm_forward"], **SMOKE)
    monkeypatch.undo()
    gc.collect()
    assert len(live_storages) == baseline


# --------------------------------------------------------------------------
# The NumPy boundary of the timed training step
# --------------------------------------------------------------------------

# Everything the timed native path must not touch: NumPy's numerical
# routines, and every route by which tensor *data* could cross into a host
# buffer.
_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "sum", "divide", "true_divide",
    "add", "subtract", "multiply", "matmul", "mean", "var", "std",
    "negative", "power", "square", "copyto", "sqrt", "reciprocal",
)
_DATA_NUMPY = ("empty", "frombuffer")


@needs_native
def test_the_timed_training_step_callable_stays_native(monkeypatch):
    """The exact callable the benchmark times — zero_grad, train(), the
    normalized forward (including the BatchNorm running-statistics
    update), LayerNorm, MSE, backward, and the NativeAdam step — runs to
    completion with NumPy's numerical routines and every tensor-data
    conversion route armed. Setup happens before the tripwire is armed."""
    from tensorforge.experimental import NativeTensor

    spec = bench.CASES["normalized_training_step"]
    case = spec["build"](spec["configurations"]["smoke"], spec)
    state = case["native_prepare"]()          # setup is outside the timer
    model, _optimizer = state
    running_before = model.batch_norm.running_mean.to_numpy().copy()

    def tripwire(*args, **kwargs):
        raise AssertionError("the timed training step reached NumPy")

    for name in _NUMERICAL_NUMPY + _DATA_NUMPY:
        monkeypatch.setattr(np, name, tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "from_array",
                        staticmethod(tripwire))
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "from_array", staticmethod(tripwire))
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "copy_from", tripwire)
    monkeypatch.setattr(NativeTensor, "to_numpy", tripwire)

    result = case["native_run"](state)         # <- the timed region

    # The tripwire really can fire from right here.
    with pytest.raises(AssertionError, match="reached NumPy"):
        result[1].to_numpy()
    monkeypatch.undo()

    prediction, loss = result
    assert math.isfinite(float(loss.to_numpy()))
    assert loss.shape == ()
    # The whole step really happened: gradients, the running update, and
    # the optimizer's own state all advanced.
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert not np.array_equal(
        model.batch_norm.running_mean.to_numpy(), running_before
    )
    assert list(state[1].step_counts) == [1] * len(list(model.parameters()))
    case["native_cleanup"](state, result)
    case["close"]()
    assert loss.closed and prediction.closed


@needs_native
def test_the_benchmark_never_reaches_checkpoint_or_reporting_helpers():
    """The timed training step does no checkpoint I/O and no reporting
    work, so the harness never even imports those names."""
    for forbidden in bench._FORBIDDEN_TRAINING_STEP_WORK:
        assert not hasattr(bench, forbidden), forbidden
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    for banned in ("save_native_checkpoint(", "load_native_checkpoint(",
                   "native_accuracy(", "run_training(", "run_resume_proof(",
                   "example.main(", "build_model("):
        assert banned not in source, banned


# --------------------------------------------------------------------------
# No timing threshold, no speed claim
# --------------------------------------------------------------------------

def test_the_benchmark_defines_no_timing_threshold():
    """Structural guard: no timing value is ever compared against a fixed
    duration, and no threshold constant exists."""
    banned_tokens = ("assert_faster", "max_seconds", "min_speedup",
                     "time_budget", "timing_threshold", "max_duration",
                     "performance_gate", "min_throughput", "speed_limit",
                     "required_speedup", "performance_budget")
    lowered = BENCHMARK_FILE.read_text(encoding="utf-8").lower()
    for banned in banned_tokens:
        assert banned not in lowered, banned
    # No module-level threshold constant hides in the benchmark: the only
    # floats it carries are correctness tolerances and module arguments.
    allowed_floats = {
        "FORWARD_ATOL", "GRADIENT_ATOL", "STATE_ATOL", "LOSS_ATOL",
        "PARAMETER_ATOL",        # correctness tolerances
        "EPS", "MOMENTUM", "TRAINING_MOMENTUM", "DEFAULT_LR",  # module args
    }
    for name in dir(bench):
        if name.startswith("_"):
            continue
        value = getattr(bench, name)
        if isinstance(value, float):
            assert name in allowed_floats, f"{name} looks like a threshold"


def test_no_source_in_this_pair_compares_a_measured_duration():
    """The measured statistics may only be checked for finiteness,
    ordering, and non-negativity relative to each other — never against a
    numeric constant, which is what a hidden performance gate looks
    like."""
    # A real comparison has whitespace around its operator; a format spec
    # like "{ratio:>8}" does not, which keeps report formatting out of the
    # scan.
    pattern = re.compile(
        r"(median_s|min_s|max_s|spread_s|samples_s|relative_spread|"
        r"native_to_reference_ratio|ratio)[\"'\]\s]{0,3}\s+[<>]=?\s*[0-9.]+"
    )
    for path in (BENCHMARK_FILE, TEST_FILE):
        offenders = [
            match.group(0) for match in pattern.finditer(
                path.read_text(encoding="utf-8")
            )
            # `>= 0.0` / `> 0.0` are non-negativity checks, not thresholds.
            if not re.search(r"[<>]=?\s*0(\.0)?$", match.group(0))
        ]
        assert offenders == [], (path.name, offenders)


def test_no_test_here_adjudicates_between_two_measured_paths():
    """No test compares one measured statistic against another — the
    harness measures, it never declares a winner — and no test times
    anything of its own."""
    source = TEST_FILE.read_text(encoding="utf-8")
    # Ordering *within* one record (min <= median <= max) is an invariant,
    # not a verdict; what is forbidden is ordering the native path against
    # the reference path.
    verdict = re.compile(
        r"assert[^\n]*\bnative\b[^\n]*[<>][^\n]*\breference\b"
        r"|assert[^\n]*\breference\b[^\n]*[<>][^\n]*\bnative\b"
    )
    assert verdict.search(source) is None
    # The tests take no clock reading of their own: no timing module is
    # imported at all, so the only mention of the timer anywhere here is
    # the reported metadata string. (Top-of-line imports only, so this
    # guard cannot match its own assertion text.)
    imported = re.findall(r"^(?:import|from)\s+(\w+)", source, re.M)
    for module_name in ("time", "timeit", "datetime", "cProfile"):
        assert module_name not in imported, module_name
    assert "pytest.approx" in source          # ordering/identity only


def test_documentation_commits_no_normalization_timing_number():
    """Benchmark numbers are machine-specific characterizations, so no
    document may publish a normalization timing, ratio, or throughput as
    a project promise."""
    surfaces = ["README.md", "CLAUDE.md"] + [
        f"docs/{name}" for name in (
            "native_normalization_design.md", "native_support_matrix.md",
            "roadmap.md", "release_history.md", "backend_experiments.md",
            "project_summary.md", "architecture.md",
        )
    ]
    for surface in surfaces:
        text = (REPO_ROOT / surface).read_text(encoding="utf-8")
        # A raw character window, not a sentence one: the file name itself
        # contains a period, which would truncate a "[^.]" span.
        chunks = [text[max(0, match.start() - 400):match.end() + 600]
                  for match in re.finditer("benchmark_native_normalization",
                                           text)]
        assert chunks, surface
        for chunk in chunks:
            assert not re.search(
                r"\d+(\.\d+)?\s*(us|ms|µs|ns|microseconds|milliseconds|"
                r"seconds)\b", chunk, re.I
            ), (surface, chunk[:160])
            assert not re.search(r"\bx faster|\bspeedup\b|\d+(\.\d+)?x\b",
                                 chunk, re.I), (surface, chunk[:160])


def test_no_committed_benchmark_result_artifact_exists():
    for pattern in ("*.json", "*.csv", "*.png", "*.svg", "*.npz"):
        assert not list((REPO_ROOT / "benchmarks").glob(pattern)), pattern
    assert not list(REPO_ROOT.glob("benchmark*.json"))
    assert not list((REPO_ROOT / "docs").glob("*normalization*.json"))


def test_ci_asserts_no_benchmark_duration():
    workflow = (REPO_ROOT / ".github" / "workflows"
                / "tests.yml").read_text(encoding="utf-8")
    assert "benchmark_native_normalization" not in workflow


# --------------------------------------------------------------------------
# Scope boundaries: F7 adds no capability
# --------------------------------------------------------------------------

def test_f7_changes_no_capability_inventory():
    import tensorforge.experimental as experimental

    assert set(experimental.__all__) == {
        "NativeTensor", "NativeParameter", "NativeParameterRegistry",
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeMSELoss", "NativeSGD", "NativeAdam",
        "save_native_checkpoint", "load_native_checkpoint",
        "NativeCrossEntropyLoss", "native_accuracy",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    }
    assert cpp.RAW_KERNELS == (
        "elementwise_add", "elementwise_subtract", "elementwise_multiply",
        "elementwise_divide", "relu", "matmul", "matmul_tiled",
    )
    assert cpp.TENSOR_CORE_KERNELS == (
        "relu", "add", "subtract", "multiply", "matmul",
    )
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "save_native_checkpoint", "load_native_checkpoint",
    )
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # No normalization operation, Core method, or kernel appeared.
    for name in ("layer_norm", "batch_norm", "layernorm", "batchnorm",
                 "normalization"):
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
    # ...and the benchmark itself is registered nowhere.
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                      cpp.STATE_SUPPORT, cpp.UNSUPPORTED):
        for banned in ("benchmark", "normalization_benchmark",
                       "characterization", "training_step"):
            assert not [n for n in inventory if banned in n.lower()], banned


def test_f7_adds_no_kernel_abi_declaration_or_checkpoint_change():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 1
    for absent in ("tf_core_layer_norm", "tf_core_batch_norm",
                   "tf_core_normalize", "tf_core_running_update"):
        assert absent not in cpp._CHECKED_KERNELS, absent
    # The benchmark declares no runtime surface of its own.
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    for banned in ("import ctypes", "CDLL(", "restype", "argtypes",
                   "_CHECKED_KERNELS", "def tf_core", "NativeTensorCore(",
                   "_from_op("):
        assert banned not in source, banned
    assert cpp.backend_info()["stable_framework_integration"] is False


def test_f7_touches_no_production_normalization_source():
    """F7 is measurement only: the harness must not monkeypatch, subclass,
    or otherwise reach into the shipped normalization implementation."""
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    for banned in ("setattr(", "_NativeBatchNorm", "replace_native_state",
                   "monkeypatch", "copy_value_", "register_buffer(",
                   "_native_state"):
        assert banned not in source, banned


def test_the_phase_f_integration_file_is_still_f8s_work():
    """F7 ships a benchmark and its test — not the cross-cutting Phase-F
    integration file, which is F8's deliverable."""
    assert not (REPO_ROOT / "tests" / "test_native_phase_f.py").exists()
    assert not list((REPO_ROOT / "examples").glob("*normalization_benchmark*"))
