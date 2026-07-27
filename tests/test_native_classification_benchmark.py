"""Tests for the native classification benchmark harness (Phase E,
milestone E9).

These validate the benchmark's *behavior* — the case inventory, the
correctness-before-timing rule, the JSON schema, reference labelling,
cleanup, the NumPy boundary of the timed training step, and the CLI —
with **no timing or threshold assertions of any kind**. Benchmark
durations are hardware-specific measurements, never pass/fail criteria,
and nothing here depends on one machine being faster than another.
Importing the benchmark module must run nothing.

Selector: python -m pytest -q -k native_classification_benchmark
"""

import gc
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from benchmarks import benchmark_native_classification as bench

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = (REPO_ROOT / "benchmarks"
                  / "benchmark_native_classification.py")

SMOKE = {"warmup": 1, "repetitions": 3, "smoke": True}

EXPECTED_CASES = (
    "exp_forward",
    "log_forward",
    "softmax_forward",
    "log_softmax_forward",
    "cross_entropy_forward",
    "cross_entropy_backward",
    "classification_training_step",
)


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


# --------------------------------------------------------------------------
# Registry and import safety
# --------------------------------------------------------------------------

def test_importing_the_module_runs_nothing(capsys):
    import importlib

    importlib.reload(bench)
    assert capsys.readouterr().out == ""


def test_case_inventory_is_exactly_the_e9_set():
    assert tuple(bench.CASES) == EXPECTED_CASES
    categories = {name: spec["category"] for name, spec in bench.CASES.items()}
    assert categories == {
        "exp_forward": "stable_math",
        "log_forward": "stable_math",
        "softmax_forward": "probability_transform",
        "log_softmax_forward": "probability_transform",
        "cross_entropy_forward": "loss_forward",
        "cross_entropy_backward": "loss_backward",
        "classification_training_step": "training_step",
    }
    # The seven required operations, each named in its own case.
    assert "exp" in bench.CASES["exp_forward"]["operation"]
    assert "log()" in bench.CASES["log_forward"]["operation"]
    assert "softmax(axis=-1)" in bench.CASES["softmax_forward"]["operation"]
    assert "log_softmax" in bench.CASES["log_softmax_forward"]["operation"]
    assert "cross_entropy" in bench.CASES["cross_entropy_forward"]["operation"]
    assert "backward" in bench.CASES["cross_entropy_backward"]["operation"]
    step = bench.CASES["classification_training_step"]["operation"]
    for piece in ("zero_grad", "NativeCrossEntropyLoss", "backward",
                  "NativeAdam.step()"):
        assert piece in step, piece


def test_every_case_declares_deterministic_shapes_and_a_seed():
    for name, spec in bench.CASES.items():
        assert set(spec["shapes"]) == {"full", "smoke"}, name
        for variant, shape in spec["shapes"].items():
            assert shape and all(isinstance(d, int) and d > 0 for d in shape), (
                name, variant
            )
        smoke, full = spec["shapes"]["smoke"], spec["shapes"]["full"]
        assert len(smoke) == len(full), name
        assert all(s <= f for s, f in zip(smoke, full)), name
        assert isinstance(spec["seed"], int) and not isinstance(spec["seed"], bool)
        assert spec["notes"].strip(), name


def test_reference_labels_are_honest():
    allowed = {bench.STABLE, bench.NUMPY, bench.NATIVE_ONLY}
    for name, spec in bench.CASES.items():
        assert spec["reference_type"] in allowed, name
        assert spec["reference_detail"].strip(), name
    # log_softmax has no direct stable counterpart, so it is labelled
    # `numpy` and says why — and it must not silently use softmax().log().
    log_softmax = bench.CASES["log_softmax_forward"]
    assert log_softmax["reference_type"] == bench.NUMPY
    detail = log_softmax["reference_detail"].lower()
    assert "log-sum-exp" in detail
    assert "softmax().log()" in detail and "not used" in detail
    # The stable-labelled cases really do name a stable-line operation.
    for name in ("exp_forward", "log_forward", "softmax_forward",
                 "cross_entropy_forward", "cross_entropy_backward"):
        spec = bench.CASES[name]
        assert spec["reference_type"] == bench.STABLE, name
        assert "tensorforge" in spec["reference_detail"], name


# --------------------------------------------------------------------------
# The smoke run, the schema, and the correctness gates
# --------------------------------------------------------------------------

@needs_native
def test_smoke_run_produces_the_documented_schema():
    payload = bench.run_benchmark(**SMOKE)
    assert set(payload) == {"benchmark", "version", "mode", "environment",
                            "cases"}
    assert payload["benchmark"] == "tensorforge.native_classification"
    assert payload["mode"] == "smoke"
    environment = payload["environment"]
    for key in ("python_version", "platform", "processor",
                "tensorforge_version", "native_backend", "dtype", "device",
                "scope", "warmup", "repetitions", "timer",
                "primary_statistic"):
        assert key in environment, key
    assert environment["dtype"] == "float64" and environment["device"] == "cpu"
    assert environment["timer"] == "time.perf_counter_ns"
    assert environment["primary_statistic"] == "median"
    assert environment["native_backend"]["available"] is True
    assert environment["native_backend"]["stable_framework_integration"] is False
    assert [record["case"] for record in payload["cases"]] == list(EXPECTED_CASES)


@needs_native
def test_every_case_reports_a_passed_correctness_gate():
    payload = bench.run_benchmark(**SMOKE)
    for record in payload["cases"]:
        correctness = record["correctness"]
        assert correctness["status"] == "passed", record["case"]
        assert correctness["checks"], record["case"]
        error = correctness["max_abs_error"]
        assert isinstance(error, float) and math.isfinite(error)
        assert error >= 0.0
    by_name = {record["case"]: record for record in payload["cases"]}
    # Each case gates what it actually promises.
    assert "no_input_mutation" in by_name["exp_forward"]["correctness"]["checks"]
    assert ("extreme_offset_stability"
            in by_name["softmax_forward"]["correctness"]["checks"])
    backward_checks = by_name["cross_entropy_backward"]["correctness"]["checks"]
    for expected in ("gradient_present", "gradient_shape", "reference_parity"):
        assert expected in backward_checks, expected
    step = by_name["classification_training_step"]["correctness"]
    for expected in ("finite_loss", "parameter_updated",
                     "optimizer_state_advanced", "graph_released",
                     "transients_closed", "reference_parity"):
        assert expected in step["checks"], expected
    # The step really moved every trainable parameter.
    assert step["updated_parameters"] == [
        "conv.bias", "conv.weight", "linear.bias", "linear.weight"
    ]


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
        reference = record["reference"]
        if record["reference_type"] == bench.NATIVE_ONLY:
            assert reference is None
            assert record["native_to_reference_ratio"] is None
        else:
            assert reference is not None, record["case"]
            assert len(reference["samples_s"]) == reference["sample_count"]
            assert reference["min_s"] <= reference["median_s"] <= reference["max_s"]
            ratio = record["native_to_reference_ratio"]
            assert math.isfinite(ratio) and ratio > 0.0


@needs_native
def test_sample_and_warmup_counts_match_the_requested_mode():
    payload = bench.run_benchmark(**SMOKE)
    for record in payload["cases"]:
        assert record["warmup"] == SMOKE["warmup"]
        assert record["sample_count"] == SMOKE["repetitions"]
    # Full-mode defaults are declared, and the training step honestly
    # reports its own smaller repetition count.
    assert bench.SMOKE_DEFAULTS == {"warmup": 1, "repetitions": 3}
    assert bench.DEFAULTS["warmup"] >= 3
    assert 10 <= bench.DEFAULTS["repetitions"] <= 20
    assert bench.TRAINING_STEP_REPETITIONS <= bench.DEFAULTS["repetitions"]


@needs_native
def test_single_case_selection():
    payload = bench.run_benchmark(cases=["softmax_forward"], **SMOKE)
    assert [record["case"] for record in payload["cases"]] == ["softmax_forward"]
    assert payload["cases"][0]["axis"] == -1


@needs_native
def test_smoke_shapes_are_used_in_smoke_mode():
    payload = bench.run_benchmark(cases=["cross_entropy_forward"], **SMOKE)
    assert tuple(payload["cases"][0]["shape"]) == (16, 8)
    full = bench.run_benchmark(cases=["cross_entropy_forward"], warmup=1,
                               repetitions=2, smoke=False)
    assert tuple(full["cases"][0]["shape"]) == (128, 32)


@needs_native
def test_a_failing_correctness_gate_aborts_before_timing(monkeypatch):
    """The gate is real: if the native operation returns a wrong result,
    the case must raise before any timing happens and publish nothing."""
    from tensorforge.experimental import NativeTensor

    timed = []
    original_measure = bench.measure

    def tracking_measure(*args, **kwargs):
        timed.append(args)
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(bench, "measure", tracking_measure)
    # A genuinely wrong native exp: same shape and finite, but not the
    # exponential — only the reference comparison can catch it.
    monkeypatch.setattr(
        NativeTensor, "exp",
        lambda self: NativeTensor.from_array(np.zeros(self.shape)),
    )
    with pytest.raises(AssertionError, match="differs from the reference"):
        bench.run_benchmark(cases=["exp_forward"], **SMOKE)
    assert timed == [], "timing ran despite a failed correctness gate"


@needs_native
def test_a_non_finite_result_is_caught_by_the_gate(monkeypatch):
    from tensorforge.experimental import NativeTensor

    monkeypatch.setattr(
        NativeTensor, "log",
        lambda self: NativeTensor.from_array(np.full(self.shape, np.inf)),
    )
    with pytest.raises(AssertionError, match="not finite"):
        bench.run_benchmark(cases=["log_forward"], **SMOKE)


@needs_native
def test_cli_reports_a_correctness_failure_with_a_nonzero_exit(monkeypatch,
                                                              capsys):
    def broken_build(shape, seed):
        raise AssertionError("synthetic gate failure")

    monkeypatch.setitem(bench.CASES["log_forward"], "build", broken_build)
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--smoke", "--case", "log_forward"])
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "correctness gate failed" in captured.err
    assert captured.out == ""      # no timing published for a failed case


# --------------------------------------------------------------------------
# CLI, JSON, and the human report
# --------------------------------------------------------------------------

@needs_native
def test_cli_json_smoke_output_parses_and_keeps_stdout_clean(capsys):
    bench.main(["--smoke", "--json", "--warmup", "1", "--repetitions", "2"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"] == "tensorforge.native_classification"
    assert payload["mode"] == "smoke"
    assert [record["case"] for record in payload["cases"]] == list(EXPECTED_CASES)
    # Round-trips: every value is JSON-native, nothing custom leaked in.
    assert json.loads(json.dumps(payload)) == payload


@needs_native
def test_cli_human_output_carries_the_local_characterization_disclaimer(capsys):
    bench.main(["--smoke", "--case", "exp_forward", "--warmup", "1",
                "--repetitions", "2"])
    output = capsys.readouterr().out
    assert "TensorForge native classification benchmark" in output
    assert "exp_forward" in output
    lowered = output.lower()
    assert "local characterization only" in lowered
    assert "not a performance contract" in lowered
    assert "not cross-machine comparable" in lowered
    assert "no test or ci job asserts any timing threshold" in lowered
    # No marketing and no speed verdict.
    for banned in ("faster", "fastest", "speedup", "outperform", "beats",
                   "pytorch", "gpu", "production-ready"):
        assert banned not in lowered, banned


def test_cli_rejects_unknown_and_invalid_arguments():
    with pytest.raises(SystemExit):
        bench.main(["--case", "nope"])
    with pytest.raises(SystemExit):
        bench.main(["--warmup", "not-a-number"])
    with pytest.raises(ValueError, match="unknown case"):
        bench.run_benchmark(cases=["nope"], **SMOKE)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_non_positive_counts_are_rejected(bad):
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
    assert "not built" in capsys.readouterr().err


@needs_native
def test_running_the_benchmark_writes_no_files():
    before = {path.name for path in REPO_ROOT.iterdir()}
    benchmarks_before = {path.name for path in (REPO_ROOT / "benchmarks").iterdir()}
    bench.run_benchmark(cases=["exp_forward"], **SMOKE)
    assert {path.name for path in REPO_ROOT.iterdir()} == before
    assert {path.name for path in (REPO_ROOT / "benchmarks").iterdir()} == (
        benchmarks_before
    )
    assert not list(REPO_ROOT.glob("*.json"))


# --------------------------------------------------------------------------
# Ownership and the NumPy boundary
# --------------------------------------------------------------------------

@needs_native
def test_repeated_smoke_runs_do_not_grow_live_native_storage(live_storages):
    """Each run creates its own persistent inputs and releases them when
    the case closes, so the live count returns to the same baseline every
    time — the invariant is that nothing accumulates across runs."""
    bench.run_benchmark(cases=["exp_forward", "cross_entropy_backward"],
                        **SMOKE)
    gc.collect()
    baseline = len(live_storages)
    for _ in range(3):
        bench.run_benchmark(cases=["exp_forward", "cross_entropy_backward"],
                            **SMOKE)
        gc.collect()
        assert len(live_storages) == baseline


@needs_native
def test_the_training_step_case_releases_its_model_and_optimizer(live_storages):
    gc.collect()
    baseline = len(live_storages)
    bench.run_benchmark(cases=["classification_training_step"], **SMOKE)
    gc.collect()
    assert len(live_storages) == baseline


# Everything the timed native path must not touch: NumPy numerical
# routines, and every route by which tensor *data* could cross into a host
# buffer. np.array/np.asarray stay available because scalar/metadata
# marshalling legitimately uses them (the E5-E8 precedent) — no tensor
# data passes through them here.
_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "logaddexp", "sum", "divide",
    "true_divide", "add", "subtract", "multiply", "matmul", "mean",
    "negative", "power", "copyto", "sqrt", "reciprocal", "take",
    "take_along_axis", "put", "put_along_axis", "where", "choose", "maximum",
)


@needs_native
def test_the_timed_training_step_callable_stays_native(monkeypatch):
    """The exact callable the benchmark times — zero_grad, forward, loss,
    backward, optimizer.step() — runs to completion with NumPy's
    numerical routines and the tensor-data conversion boundary armed."""
    from tensorforge.experimental import NativeTensor

    spec = bench.CASES["classification_training_step"]
    case = spec["build"](spec["shapes"]["smoke"], spec["seed"])
    state = case["native_prepare"]()          # setup is outside the timer

    def tripwire(*args, **kwargs):
        raise AssertionError("the timed training step reached NumPy")

    for name in _NUMERICAL_NUMPY:
        monkeypatch.setattr(np, name, tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", tripwire)
    monkeypatch.setattr(NativeTensor, "to_numpy", tripwire)

    result = case["native_run"](state)         # <- the timed region

    # The tripwire really can fire from right here.
    with pytest.raises(AssertionError, match="reached NumPy"):
        result[1].to_numpy()
    monkeypatch.undo()

    logits, loss = result
    assert math.isfinite(float(loss.to_numpy()))
    case["native_cleanup"](state, result)
    case["close"]()
    assert loss.closed and logits.closed


@needs_native
def test_the_benchmark_never_calls_native_accuracy(monkeypatch):
    """The reporting metric converts to the host on purpose, so it has no
    place in a timed measurement — the benchmark must not call it at
    all."""
    from tensorforge.experimental import native_metrics

    def tripwire(*args, **kwargs):
        raise AssertionError("the benchmark called native_accuracy")

    monkeypatch.setattr(native_metrics, "native_accuracy", tripwire)
    bench.run_benchmark(cases=["classification_training_step"], **SMOKE)
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    # Named only in prose that explains its exclusion, never called.
    assert "native_accuracy(" not in source


# --------------------------------------------------------------------------
# No speed assertion, no threshold
# --------------------------------------------------------------------------

def test_neither_the_benchmark_nor_its_tests_assert_a_speed():
    """Structural guard: no timing value is ever compared against a fixed
    duration, and no threshold constant exists."""
    banned_tokens = ("assert_faster", "max_seconds", "min_speedup",
                     "time_budget", "timing_threshold", "max_duration",
                     "performance_gate")
    lowered = BENCHMARK_FILE.read_text(encoding="utf-8").lower()
    for banned in banned_tokens:
        assert banned not in lowered, banned
    # No module-level threshold constant hides in the benchmark.
    for name in dir(bench):
        if name.startswith("_"):
            continue
        value = getattr(bench, name)
        if isinstance(value, float):
            # The only floats the module carries are correctness
            # tolerances and the E8 learning rate it reuses.
            assert name in ("FORWARD_ATOL", "LOSS_ATOL", "GRADIENT_ATOL",
                            "PARAMETER_ATOL", "DEFAULT_LR"), (
                f"{name} looks like a timing threshold"
            )
    # The tolerances that do exist are correctness tolerances only.
    for name in ("FORWARD_ATOL", "LOSS_ATOL", "GRADIENT_ATOL",
                 "PARAMETER_ATOL"):
        assert 0 < getattr(bench, name) < 1e-6, name


def test_no_test_in_this_file_compares_a_measured_duration():
    """The measured statistics may only be checked for finiteness,
    ordering, and non-negativity relative to each other — never against a
    numeric constant, which is what a hidden performance gate looks
    like."""
    import re

    # A real comparison has whitespace around its operator; a format spec
    # like "{ratio:>8}" does not, which keeps the report formatting out of
    # the scan.
    pattern = re.compile(
        r"(median_s|min_s|max_s|spread_s|samples_s|"
        r"native_to_reference_ratio|ratio)[\"'\]\s]{0,3}\s+[<>]=?\s*[0-9.]+"
    )
    for path in (BENCHMARK_FILE, Path(__file__)):
        offenders = [
            match.group(0) for match in pattern.finditer(
                path.read_text(encoding="utf-8")
            )
            # `>= 0.0` / `> 0.0` are non-negativity checks, not thresholds.
            if not re.search(r"[<>]=?\s*0(\.0)?$", match.group(0))
        ]
        assert offenders == [], (path.name, offenders)


# --------------------------------------------------------------------------
# Scope boundaries: E9 adds no capability
# --------------------------------------------------------------------------

def test_e9_adds_no_capability_inventory_entry():
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
        "NativeLayerNorm",     # Phase F, milestone F2 (unrelated to E9)
        "NativeBatchNorm1d",   # Phase F, milestone F3 (unrelated to E9)
        "NativeBatchNorm2d",   # Phase F, milestone F4 (unrelated to E9)
        "NativeDropout",       # Phase G, milestone G4 (unrelated to E9)
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    # "persistent_buffers" joined this tuple in Phase F milestone F1, as
    # reconciliation of a capability that predates Phase E entirely; E9
    # (a measurement-only milestone) contributed nothing here.
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers",
        "state_dict", "load_state_dict",
        "generator_state",   # Phase G, milestone G1 (in-memory only)
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",   # Phase G, milestone G5 (the file half)
    )
    # "cross_entropy" was the last autograd op when E9 landed and E9 added
    # nothing after it. The one entry that follows is Phase G milestone
    # G3's differentiable "dropout", which is unrelated to this benchmark.
    assert cpp.AUTOGRAD_OPS[-2] == "cross_entropy"
    assert cpp.AUTOGRAD_OPS[-1] == "dropout"
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                      cpp.UNSUPPORTED):
        for banned in ("benchmark", "classification_benchmark",
                       "training_step", "characterization"):
            assert not [n for n in inventory if banned in n.lower()], banned


def test_e9_adds_no_kernel_abi_operation_or_schema():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 2
    for absent in ("tf_core_benchmark", "tf_core_train_step",
                   "tf_core_accuracy", "tf_core_argmax"):
        assert absent not in cpp._CHECKED_KERNELS, absent
    # The benchmark defines no runtime surface of its own: it imports no
    # ctypes machinery, declares no ABI signature, and registers no
    # kernel. (The word "ctypes" appears only in prose explaining what the
    # measured times include, so the check is structural.)
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    for banned in ("import ctypes", "CDLL(", "restype", "argtypes",
                   "_CHECKED_KERNELS", "def tf_core", "NativeTensorCore("):
        assert banned not in source, banned
    assert cpp.backend_info()["stable_framework_integration"] is False


def test_the_benchmark_stays_separate_from_the_phase_e_closure_tests():
    """E10 closed the phase with a cross-cutting integration test. That
    file owns stack-level guarantees; this benchmark stays measurement
    only, and neither absorbs the other."""
    integration = REPO_ROOT / "tests" / "test_native_phase_e.py"
    assert integration.is_file()
    source = integration.read_text(encoding="utf-8")
    # The closure test may exercise the benchmark's correctness path, but
    # it must not time anything or assert a duration.
    assert "perf_counter" not in source
    assert "median_s\"] <" not in source and "median_s'] <" not in source
