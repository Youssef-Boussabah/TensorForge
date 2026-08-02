"""Contract tests for the dtype characterization harness (Phase I, I10).

These prove the harness's **structure and discipline**, never a speed.
There is deliberately no assertion here that one dtype is faster, that any
case meets a threshold, that a median falls in a range, or that the cases
rank in any order — a benchmark that asserted a duration would become a CI
job that fails on a number, which §9 of ``CLAUDE.md`` forbids outright.

What is asserted instead:

- the correctness gate runs **before** the timer, structurally and
  behaviourally, and a wrong result aborts before any timing exists;
- float32 and float64 are characterized **separately**, and no ratio
  between them appears anywhere in the payload or the report;
- no result file of any kind is written, and no CLI option could ask for
  one;
- the case inventory, families, gates, and configurations are
  deterministic;
- importing the module runs nothing.

Selector: python -m pytest -q -k native_dtype_benchmark
"""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_native_dtype as bench   # noqa: E402

from tensorforge.backends import cpp                     # noqa: E402

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "benchmark_native_dtype.py"

EXPECTED_FAMILIES = {
    "transfer", "elementwise", "reduction", "matmul", "cnn",
    "classification", "normalization", "dropout", "optimizer", "training",
    "control",
}


# ==========================================================================
# 1. Inventory and structure
# ==========================================================================


def test_importing_the_module_runs_nothing(capsys):
    """Importing a benchmark must never execute one."""
    import importlib

    importlib.reload(bench)
    assert capsys.readouterr().out == ""


def test_the_dtype_axis_is_exactly_the_two_supported_widths():
    """The harness measures what the runtime supports, and reads the
    order from the same contract: float64 first, because it is the
    default an omitted ``dtype`` selects."""
    assert bench.DTYPES == ("float64", "float32")
    assert bench.DTYPES == cpp.SUPPORTED_DTYPES
    assert set(bench.NUMPY_DTYPES) == set(bench.DTYPES)
    assert set(bench.TOLERANCES) == set(bench.DTYPES)


def test_every_family_is_populated_and_declared():
    assert set(bench.FAMILIES) == EXPECTED_FAMILIES
    populated = {spec["family"] for spec in bench.CASES.values()}
    assert populated == EXPECTED_FAMILIES, (
        f"declared but empty: {EXPECTED_FAMILIES - populated}")


def test_the_required_layer_coverage_is_present():
    """The layers I10 undertook to characterize, by name, so a case
    quietly disappearing shows up here rather than as a shorter table."""
    required = {
        "transfer": {"host_ingress", "host_egress", "contiguous_copy",
                     "strided_materialize"},
        "elementwise": {"elementwise_contiguous", "elementwise_broadcast",
                        "elementwise_small"},
        "reduction": {"reduction_contiguous", "reduction_strided"},
        "matmul": {"matmul_contiguous", "matmul_transposed_view"},
        "cnn": {"conv2d_forward", "conv2d_input_backward",
                "maxpool2d_forward"},
        "classification": {"softmax", "cross_entropy_forward"},
        "normalization": {"layernorm_step", "batchnorm_training_step"},
        "dropout": {"dropout_step"},
        "optimizer": {"sgd_step", "adam_step"},
        "training": {"training_step"},
        "control": {"control_identical", "control_twin"},
    }
    for family, names in required.items():
        present = {name for name, spec in bench.CASES.items()
                   if spec["family"] == family}
        assert names <= present, (family, names - present)


def test_every_case_declares_a_complete_deterministic_specification():
    seeds = set()
    for name, spec in bench.CASES.items():
        for field in ("family", "build", "seed", "operation", "gate",
                      "configurations", "notes"):
            assert field in spec, (name, field)
        assert callable(spec["build"]), name
        assert spec["family"] in bench.FAMILIES, name
        assert spec["gate"] in (bench.BITWISE, bench.TOLERANCE,
                                bench.SUMMATION_BOUND, bench.FINITE), name
        assert isinstance(spec["seed"], int), name
        assert set(spec["configurations"]) == {"full", "smoke"}, name
        assert spec["notes"].strip(), name
        seeds.add(spec["seed"])
    # One shared seed, deliberately: the control pair must have identical
    # inputs or it is not a control.
    assert bench.CASES["control_identical"]["seed"] == \
        bench.CASES["control_twin"]["seed"]
    assert len(seeds) == len(bench.CASES) - 1


def test_smoke_configurations_never_exceed_the_full_ones():
    for name, spec in bench.CASES.items():
        full = spec["configurations"]["full"]
        smoke = spec["configurations"]["smoke"]
        assert set(full) == set(smoke), name
        for key, value in full.items():
            if isinstance(value, tuple):
                assert np.prod(smoke[key]) <= np.prod(value), (name, key)
            else:
                assert smoke[key] <= value, (name, key)


def test_the_size_independent_case_keeps_one_shape():
    """``elementwise_small`` exists to show the Python-plus-ctypes floor,
    so a larger smoke shape would make it measure something else."""
    spec = bench.CASES["elementwise_small"]
    assert spec.get("size_independent") is True
    assert (spec["configurations"]["full"]
            == spec["configurations"]["smoke"])


def test_the_control_pair_is_genuinely_identical():
    """A control whose two halves differed would measure the difference
    instead of the noise."""
    first = bench.CASES["control_identical"]
    second = bench.CASES["control_twin"]
    for field in ("family", "build", "seed", "gate", "configurations"):
        assert first[field] == second[field], field


# ==========================================================================
# 2. The correctness gate runs before the timer
# ==========================================================================


@needs_native
def test_the_gate_runs_before_the_timer_structurally(monkeypatch):
    """If ``measure`` is reached at all, ``verify`` has already returned.
    Proved by making the timer explode: the exception must come from the
    timer, which means the gate ran and passed first."""
    order = []

    def tracking_measure(case, warmup, repetitions):
        order.append("measure")
        raise RuntimeError("timer reached")

    monkeypatch.setattr(bench, "measure", tracking_measure)
    with pytest.raises(RuntimeError, match="timer reached"):
        bench.run_case("elementwise_contiguous", "float32", 0, 1, smoke=True)
    assert order == ["measure"]


@needs_native
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_a_wrong_result_aborts_before_any_timing(dtype, monkeypatch):
    """The negative control that makes every gate load-bearing.

    The native result is perturbed, and the case must fail **in the gate**
    — before ``measure`` is entered, so no timing for it can exist. No
    repository source is modified to arrange this."""
    entered = []
    original = bench.measure

    def spy(case, warmup, repetitions):
        entered.append(True)
        return original(case, warmup, repetitions)

    monkeypatch.setattr(bench, "measure", spy)

    original_to_numpy = cpp.NativeTensorCore.to_numpy

    def wrong(self):
        return original_to_numpy(self) + np.asarray(
            1.0, dtype=bench.NUMPY_DTYPES[dtype])

    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", wrong)
    with pytest.raises(AssertionError):
        bench.run_case("elementwise_contiguous", dtype, 0, 1, smoke=True)
    assert entered == [], "timing was reached despite a failed gate"


@needs_native
def test_a_non_finite_result_is_caught_by_the_gate(monkeypatch):
    original_to_numpy = cpp.NativeTensorCore.to_numpy

    def poisoned(self):
        values = original_to_numpy(self)
        if values.size:
            values = values.copy()
            values.reshape(-1)[0] = np.inf
        return values

    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", poisoned)
    with pytest.raises(AssertionError):
        bench.run_case("conv2d_forward", "float32", 0, 1, smoke=True)


@needs_native
def test_a_wrong_dtype_result_is_caught_by_the_gate():
    """A gate that silently accepted the other width would let a case run
    at float64 while reporting float32 — the one mistake that would make
    this whole harness dishonest."""
    with pytest.raises(AssertionError, match="expected a float32 result"):
        bench.bits_of(np.ones(4, dtype=np.float64), "float32")
    with pytest.raises(AssertionError, match="expected float32"):
        bench.gate_tolerance(np.ones(4, dtype=np.float64),
                             np.ones(4, dtype=np.float32), "float32", "x")


@needs_native
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_every_case_reports_a_passed_gate_at_its_own_dtype(dtype):
    payload = bench.run_benchmark(dtypes=[dtype], smoke=True)
    assert payload["rows"], "the harness produced no rows"
    for row in payload["rows"]:
        assert row["dtype"] == dtype
        assert row["correctness"]["passed"] is True, row["case"]
        assert row["correctness"]["gate"] in (
            bench.BITWISE, bench.TOLERANCE, bench.SUMMATION_BOUND,
            bench.FINITE), row["case"]
        assert row["correctness"]["elements"] > 0, row["case"]


@needs_native
def test_each_gate_publishes_what_it_actually_compared():
    """Honest labelling: an inexact comparison publishes the bound it
    used, so a reader is never left to assume bit equality — and a bitwise
    one publishes no tolerance, because it had none."""
    payload = bench.run_benchmark(smoke=True)
    seen = set()
    for row in payload["rows"]:
        gate = row["correctness"]
        seen.add(gate["comparison"])
        if gate["comparison"] == bench.TOLERANCE:
            assert "rtol" in gate and "atol" in gate, row["case"]
            assert gate["rtol"] == bench.TOLERANCES[row["dtype"]]["rtol"]
        elif gate["comparison"] == bench.SUMMATION_BOUND:
            # The derived bound, the observed difference, the term count,
            # and the rule itself — all four, so the number can be checked
            # rather than trusted.
            for field in ("bound", "observed", "terms", "bound_rule"):
                assert field in gate, (row["case"], field)
            assert gate["observed"] <= gate["bound"], row["case"]
            assert gate["terms"] > 0, row["case"]
            assert gate["bound_rule"] == "2 * n * eps * max sum|terms|"
            assert "rtol" not in gate, row["case"]
        else:
            assert "rtol" not in gate, row["case"]
    # All four gates are genuinely exercised, so none is dead labelling.
    assert seen == {bench.BITWISE, bench.TOLERANCE, bench.SUMMATION_BOUND,
                    bench.FINITE}, seen


@needs_native
def test_the_summation_bound_is_derived_from_the_operands_not_tuned():
    """The bound scales with the accumulation length and the operand
    magnitudes, because that is what the error does. A constant would be
    the wrong instrument: it fails first on the output cell that happens
    to sum to nearly zero, which says nothing about either
    implementation."""
    expected = np.zeros(4, dtype=np.float32)
    got = np.full(4, 1e-4, dtype=np.float32)
    magnitudes = np.full(4, 1.0)
    # Too few terms to justify the observed difference: rejected.
    with pytest.raises(AssertionError, match="summation bound"):
        bench.gate_accumulated(got, expected, "float32", "probe", 4,
                               magnitudes)
    # The same difference over a long enough accumulation of large enough
    # terms is inside the bound, and the gate says so.
    result = bench.gate_accumulated(got, expected, "float32", "probe",
                                    4096, np.full(4, 1000.0))
    assert result["comparison"] == bench.SUMMATION_BOUND
    assert result["observed"] <= result["bound"]
    # float64's bound is far tighter than float32's for the same shape,
    # because eps is, so the gate is genuinely dtype-aware.
    narrow = bench.gate_accumulated(np.zeros(4, dtype=np.float32),
                                    np.zeros(4, dtype=np.float32),
                                    "float32", "probe", 100, magnitudes)
    wide = bench.gate_accumulated(np.zeros(4, dtype=np.float64),
                                  np.zeros(4, dtype=np.float64),
                                  "float64", "probe", 100, magnitudes)
    assert wide["bound"] < narrow["bound"]


# ==========================================================================
# 3. The two dtypes are measured separately and never divided
# ==========================================================================


@needs_native
def test_both_dtypes_are_measured_and_every_case_appears_at_each():
    payload = bench.run_benchmark(smoke=True)
    assert payload["dtypes"] == ["float64", "float32"]
    by_dtype = {}
    for row in payload["rows"]:
        by_dtype.setdefault(row["dtype"], set()).add(row["case"])
    assert set(by_dtype) == {"float64", "float32"}
    assert by_dtype["float64"] == by_dtype["float32"] == set(bench.CASES)


@needs_native
def test_no_cross_dtype_ratio_appears_anywhere_in_the_payload():
    """Design §10.4 forbids making a contract out of a float32/float64
    comparison, and this harness does not make one even informally. No
    key, and no field name, offers one."""
    payload = bench.run_benchmark(smoke=True)

    def every_key(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from every_key(value)
        elif isinstance(node, list):
            for item in node:
                yield from every_key(item)

    # Keys, compared **token by token** rather than as substrings:
    # "operation" and "configuration" both innocently contain "ratio", and
    # a test that tripped on those would be noise rather than a guardrail.
    banned_tokens = {"ratio", "speedup", "faster", "improvement",
                     "baseline", "versus"}
    for key in every_key(payload):
        tokens = set(key.lower().replace("-", "_").split("_"))
        overlap = tokens & banned_tokens
        assert not overlap, (key, overlap)
    # No *value* offers a cross-dtype comparison either.
    blob = json.dumps(payload).lower()
    for banned in ("speedup", "vs_float", "faster than", "x faster",
                   "improvement over"):
        assert banned not in blob, banned


@needs_native
def test_the_report_publishes_no_dtype_speed_claim():
    payload = bench.run_benchmark(smoke=True)
    report = bench.format_report(payload).lower()
    assert "float32" in report and "float64" in report
    for banned in ("faster", "speedup", "x speed", "outperform",
                   "beats", "wins"):
        assert banned not in report, banned
    # ...and it says out loud why there is no ratio.
    assert "no float32/float64 ratio" in report


@needs_native
def test_the_control_band_is_reported_and_is_not_a_gate():
    payload = bench.run_benchmark(smoke=True)
    band = bench.control_band(payload["rows"])
    assert set(band) == {"float64", "float32"}
    for dtype, value in band.items():
        assert value >= 0.0, dtype
        assert np.isfinite(value), dtype
    # It appears in the report as guidance, with no threshold attached.
    report = bench.format_report(payload)
    assert "Control band" in report
    assert "neutral" in report


# ==========================================================================
# 4. Timing hygiene — no assertion, no artifact
# ==========================================================================


@needs_native
def test_timing_fields_are_complete_finite_and_non_negative():
    payload = bench.run_benchmark(smoke=True)
    for row in payload["rows"]:
        for field in ("median_s", "iqr_s", "min_s", "max_s", "samples",
                      "relative_iqr", "spread_statistic"):
            assert field in row, (row["case"], field)
        assert row["median_s"] >= 0.0
        assert row["iqr_s"] >= 0.0
        assert row["min_s"] <= row["median_s"] <= row["max_s"]
        assert row["spread_statistic"] == "interquartile range"


@needs_native
def test_no_sample_is_discarded():
    payload = bench.run_benchmark(smoke=True, repetitions=3)
    for row in payload["rows"]:
        assert row["samples"] == 3, row["case"]


@needs_native
def test_a_low_round_count_is_marked_unpublishable():
    """Low round counts lie — Phase H recorded four separate cases that
    read as regressions at 7-9 rounds and as neutral-or-faster at 21-25.
    The harness says so rather than letting a reader quote a smoke run."""
    assert bench.run_benchmark(smoke=True)["publishable"] is False
    assert bench.run_benchmark(
        cases=["control_identical"], repetitions=9)["publishable"] is False
    assert bench.run_benchmark(
        cases=["control_identical"],
        repetitions=bench.PUBLISHABLE_MINIMUM)["publishable"] is True
    report = bench.format_report(bench.run_benchmark(smoke=True))
    assert "not publishable evidence" in report


def test_the_module_contains_no_timing_assertion_and_no_threshold():
    """Structural, over the harness's own source: a benchmark that grew a
    threshold would become a CI job that fails on a number."""
    source = inspect.getsource(bench)
    for banned in ("assert median", "assert row['median", "THRESHOLD",
                   "BUDGET", "max_duration", "min_speedup",
                   "assert elapsed", "assert samples[0] <"):
        assert banned not in source, banned


def test_the_harness_writes_no_result_file_of_any_kind():
    """No JSON file, no CSV, no database, no pickle, no cache. A committed
    number becomes a promise the project cannot keep across machines."""
    source = inspect.getsource(bench)
    for banned in ("open(", "Path.write", "write_text", "write_bytes",
                   "savez", "to_csv", "pickle", "shelve", "sqlite",
                   "mkdtemp", "NamedTemporary"):
        assert banned not in source, banned
    # ...and no CLI option could ask for one.
    parser = bench.build_parser()
    options = {action.dest for action in parser._actions}
    for banned in ("output", "out", "outfile", "results", "save", "report",
                   "csv", "path", "file"):
        assert banned not in options, banned
    assert "json" in options            # stdout only


@needs_native
def test_the_repetition_and_selection_arguments_are_validated():
    with pytest.raises(ValueError):
        bench.run_benchmark(repetitions=0)
    with pytest.raises(ValueError):
        bench.run_benchmark(warmup=-1)
    with pytest.raises(ValueError):
        bench.run_benchmark(cases=["no_such_case"])
    with pytest.raises(ValueError):
        bench.run_benchmark(dtypes=["float16"])


@needs_native
def test_case_and_family_selection_work():
    single = bench.run_benchmark(cases=["softmax"], smoke=True)
    assert {row["case"] for row in single["rows"]} == {"softmax"}
    assert {row["dtype"] for row in single["rows"]} == {"float64", "float32"}
    family = bench.run_benchmark(family="matmul", smoke=True)
    assert {row["case"] for row in family["rows"]} == {
        "matmul_contiguous", "matmul_transposed_view"}
    one = bench.run_benchmark(cases=["softmax"], dtypes=["float32"],
                              smoke=True)
    assert {row["dtype"] for row in one["rows"]} == {"float32"}


@needs_native
def test_the_environment_is_real_introspection_not_a_restatement():
    """A fabricated CPU model would make a published table worse than
    useless, so anything undeterminable is reported as ``None``."""
    info = bench.environment()
    assert info["python"] == ".".join(str(p) for p in sys.version_info[:3])
    assert info["numpy"] == np.__version__
    assert info["backend_supported_dtypes"] == list(cpp.SUPPORTED_DTYPES)
    assert info["backend_default_dtype"] == cpp.backend_info()["dtype"]
    assert info["backend_raw_kernel_dtypes"] == list(cpp.RAW_KERNEL_DTYPES)


@needs_native
def test_the_benchmark_uses_only_local_seeded_generators():
    """Touching the global NumPy RNG would make two runs build different
    inputs and would leak into anything else running in the process."""
    source = inspect.getsource(bench)
    assert "default_rng" in source
    for banned in ("np.random.seed", "np.random.rand", "np.random.randn",
                   "np.random.normal", "random.seed", "random.random"):
        assert banned not in source, banned

    before = np.random.get_state()[1][:8].copy()
    bench.run_benchmark(cases=["elementwise_contiguous"], smoke=True)
    assert np.array_equal(np.random.get_state()[1][:8], before)


@needs_native
def test_repeated_runs_produce_identical_correctness_metrics():
    """Deterministic inputs mean deterministic gates; only the timings may
    move between runs."""
    first = bench.run_benchmark(smoke=True)
    second = bench.run_benchmark(smoke=True)
    assert ([(r["case"], r["dtype"], r["correctness"]) for r in
             first["rows"]]
            == [(r["case"], r["dtype"], r["correctness"]) for r in
                second["rows"]])


@needs_native
def test_the_harness_leaks_no_native_storage(monkeypatch):
    """Setup, teardown, and every timed repetition return live native
    storage exactly to baseline."""
    import gc

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
    bench.run_benchmark(smoke=True)
    gc.collect()
    assert len(open_ids) == baseline


# ==========================================================================
# 5. The CLI
# ==========================================================================


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, str(BENCHMARK_PATH), *arguments],
        capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT))


@needs_native
def test_cli_smoke_json_parses_and_keeps_stdout_clean():
    result = run_cli("--smoke", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["harness"] == "benchmark_native_dtype"
    assert payload["milestone"] == "I10"
    assert payload["mode"] == "smoke"
    assert payload["publishable"] is False
    assert payload["rows"]


@needs_native
def test_cli_human_output_carries_the_local_characterization_disclaimer():
    result = run_cli("--smoke", "--case", "softmax")
    assert result.returncode == 0, result.stderr
    assert "No speed is asserted" in result.stdout
    assert "no result file is written" in result.stdout
    assert "measured separately" in result.stdout


@needs_native
def test_cli_writes_no_file_beside_the_repository(tmp_path):
    """Run from an empty working directory and prove nothing appeared."""
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_PATH), "--smoke", "--case",
         "control_identical"],
        capture_output=True, text=True, timeout=900, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == [], "the harness wrote a file"


@needs_native
def test_cli_rejects_an_unknown_case_without_polluting_stdout():
    result = run_cli("--case", "no_such_case")
    assert result.returncode != 0
    assert result.stdout.strip() == ""


@needs_native
def test_cli_refuses_case_and_family_together():
    result = run_cli("--case", "softmax", "--family", "matmul")
    assert result.returncode != 0
    assert result.stdout.strip() == ""


@needs_native
def test_cli_reports_a_correctness_failure_with_a_nonzero_exit(tmp_path):
    """A failed gate exits nonzero with clean stdout — arranged from
    outside, through a sitecustomize shim, so **no repository source is
    modified** to build the control."""
    shim = tmp_path / "sitecustomize.py"
    shim.write_text(
        "import numpy as np\n"
        "from tensorforge.backends import cpp\n"
        "_original = cpp.NativeTensorCore.to_numpy\n"
        "cpp.NativeTensorCore.to_numpy = (\n"
        "    lambda self: _original(self) + 1.0)\n",
        encoding="utf-8")
    import os
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_PATH), "--smoke", "--case",
         "elementwise_contiguous"],
        capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT),
        env=environment)
    assert result.returncode != 0
    assert result.stdout.strip() == ""
    assert "correctness gate failed" in result.stderr


# ==========================================================================
# 6. The harness adds no capability
# ==========================================================================


@needs_native
def test_the_harness_changed_no_registry_or_constant():
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["dtype"] == "float64"


def test_the_harness_measures_no_raw_kernel_at_float32():
    """The seven handle-free raw utility kernels take only ``double*``, so
    there is no float32 case to measure and none is invented."""
    source = inspect.getsource(bench)
    for banned in ("cpp.matmul(", "cpp.matmul_tiled(", "cpp.relu(",
                   "cpp.elementwise_"):
        assert banned not in source, banned


def test_the_harness_writes_no_checkpoint_during_a_timed_repetition():
    """File I/O is dominated by the filesystem rather than by TensorForge,
    and measuring it here would also make this harness write files."""
    source = inspect.getsource(bench)
    for banned in ("save_native_checkpoint", "load_native_checkpoint"):
        assert banned not in source, banned
