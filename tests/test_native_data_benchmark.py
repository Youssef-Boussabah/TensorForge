"""Contract tests for the data-pipeline characterization harness (J8).

These prove the harness's **structure and discipline**, never a speed.
There is deliberately no assertion anywhere that a case is fast, that a
median falls in a range, that one dtype beats the other, that a ratio
meets a bound, or that the cases rank in any order — a benchmark that
asserted a duration would become a CI job that fails on a number, which
§9 of ``CLAUDE.md`` forbids outright and which the J8 exit gate names as
the thing that must not exist anywhere in the repository.

What is asserted instead:

- the correctness gate runs **before** the timer, structurally and
  behaviourally, and every kind of wrong result aborts before any timing
  exists;
- float32 and float64 are characterized **separately**, and no ratio
  between them appears in the payload, the report, or the source;
- a ``native_only`` case publishes **no ratio**, and every published
  ratio has a same-dtype, same-inputs, independently written reference;
- setup, per-repetition state reset, and every ``close()`` happen outside
  the timed region, and every measured sample is retained;
- no result file of any kind is written, and no CLI option could ask for
  one;
- the case inventory, workloads, gates, seeds, and configurations are
  exact and deterministic;
- importing the module runs nothing at all;
- J8 shipped no production change, no optimization, and no new export.

**Every parser and every scanner here has a negative control**, driven
against text or payloads it must reject, so "nothing found" is evidence
rather than a dead regex.

Selector: python -m pytest -q tests/test_native_data_benchmark.py
"""

import ast
import gc
import getpass
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_native_data_pipeline as bench   # noqa: E402

from tensorforge.backends import cpp                             # noqa: E402
from tensorforge.experimental import (                           # noqa: E402
    NativeBatchSampler,
    NativeDataLoader,
    NativeTensor,
    NativeTensorDataset,
)

BENCHMARK_FILE = (REPO_ROOT / "benchmarks"
                  / "benchmark_native_data_pipeline.py")
BENCHMARK_SOURCE = BENCHMARK_FILE.read_text(encoding="utf-8")

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

# The exact, ordered case inventory J8 shipped. Written here independently
# of the harness, so a case appearing, disappearing, or moving family
# fails an equality rather than being absorbed silently.
EXPECTED_CASES = (
    "host_feature_gather_sequential",
    "host_feature_gather_shuffled",
    "host_feature_gather_duplicates",
    "dataset_target_batch_sequential",
    "dataset_target_batch_shuffled",
    "plan_sequential_exact",
    "plan_sequential_short_final",
    "plan_shuffled_reference",
    "plan_shuffled_large",
    "next_batch_indices_fresh",
    "next_batch_indices_mid_epoch",
    "permutation_cold_reference",
    "permutation_cold_later_epoch",
    "permutation_cold_large",
    "permutation_cache_hit",
    "feature_batch_small",
    "feature_batch_large",
    "feature_batch_shuffled",
    "feature_batch_image",
    "loader_next_batch",
)

EXPECTED_WORKLOADS = (
    "dataset_indexing",
    "batch_planning",
    "permutation_construction",
    "host_to_native_materialization",
    "loader_delivery",
)

# The four layers J8 undertook to isolate, named so a family quietly
# collapsing into another shows up here rather than as a shorter table.
REQUIRED_COVERAGE = {
    "dataset_indexing": {
        "host_feature_gather_sequential", "host_feature_gather_shuffled",
        "host_feature_gather_duplicates", "dataset_target_batch_sequential",
        "dataset_target_batch_shuffled",
    },
    "batch_planning": {
        "plan_sequential_exact", "plan_sequential_short_final",
        "plan_shuffled_reference", "plan_shuffled_large",
        "next_batch_indices_fresh", "next_batch_indices_mid_epoch",
    },
    "permutation_construction": {
        "permutation_cold_reference", "permutation_cold_later_epoch",
        "permutation_cold_large", "permutation_cache_hit",
    },
    "host_to_native_materialization": {
        "feature_batch_small", "feature_batch_large",
        "feature_batch_shuffled", "feature_batch_image",
    },
    "loader_delivery": {"loader_next_batch"},
}

ROOT_KEYS = {
    "benchmark", "benchmark_version", "schema_version", "mode",
    "selected_cases", "selected_workloads", "dtypes", "environment",
    "methodology", "cases", "disclaimer",
}

CASE_KEYS = {
    "case", "workload", "operation", "dtype", "configuration", "config",
    "feature_shape", "seed", "reference_type", "native_only",
    "reference_detail", "cache_state", "correctness", "timed_layer",
    "setup", "cleanup", "warmup", "statistics", "reference",
    "ratio_to_reference", "ratio_meaning", "notes",
}

STATISTICS_KEYS = {
    "sample_count", "samples_ns", "median_ns", "min_ns", "max_ns",
    "p25_ns", "p75_ns", "iqr_ns", "mean_ns", "relative_iqr",
    "headline_statistic", "spread_statistic", "units",
}

# The exact key set each gate publishes. A gate that started reporting
# less would make its own record unauditable.
GATE_KEYS = {
    "host_gather_bits": {"passed", "gate", "rows", "elements",
                         "duplicate_indices", "matched_feature_batch_bits",
                         "reference_agrees"},
    "target_batch_exact": {"passed", "gate", "rows", "elements",
                           "duplicate_indices", "read_only",
                           "reference_agrees"},
    "plan_exact": {"passed", "gate", "batches", "batch_size", "final_batch",
                   "short_final_batch", "shuffled", "cursor",
                   "checked_against_reference_vector", "position_unchanged"},
    "permutation_exact": {"passed", "gate", "length", "epoch", "cache_state",
                          "checked_against_reference_vector",
                          "position_unchanged"},
    "feature_batch_bits": {"passed", "gate", "rows", "elements",
                           "duplicate_indices", "device",
                           "owning_contiguous", "fresh_storage_per_call"},
    "delivery_transaction": {"passed", "gate", "rows", "batch_size",
                             "position_advanced_by_one_batch",
                             "restored_to_canonical_position"},
}

SMOKE = {"smoke": True}


def code_only(source):
    """The harness's **code**, with every docstring, comment, and string
    literal removed.

    Every substring ban below runs over this rather than over the raw
    file, on J6's recorded lesson: a naive scan fails on exactly the prose
    that documents the prohibition. This module's own harness says in
    words that it subtracts no timer overhead, discards no outlier, and
    uses no tolerance — and a checker that tripped on those sentences
    would be noise rather than a guardrail, and would push the next author
    to delete the explanation instead of the behaviour.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(tree)


BENCHMARK_CODE = code_only(BENCHMARK_SOURCE)


def test_the_code_only_scanner_can_actually_fail():
    """Negative control for every scan that runs over ``BENCHMARK_CODE``:
    prose must be stripped, and real code must survive."""
    stripped = code_only('"""A docstring naming np.allclose."""\n'
                         'VALUE = "a literal naming threading"\n'
                         'x = np.allclose(a, b)  # a comment naming outlier\n')
    assert "docstring" not in stripped
    assert "threading" not in stripped
    assert "outlier" not in stripped
    assert "np.allclose(a, b)" in stripped
    # ...and the real harness's code really did survive stripping.
    assert "def measure(" in BENCHMARK_CODE
    assert "time.perf_counter_ns()" in BENCHMARK_CODE


def _imported_modules(source):
    """Every top-level module name a source file imports."""
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


# ===========================================================================
# Shared instrumentation
# ===========================================================================


@pytest.fixture
def live_storages():
    """The ids of every open ``NativeStorage`` — the project's
    deterministic instrumentation for native-allocation lifetime, used
    unchanged since Phase C. **There is no public counter and J8 adds
    none.**

    Installed with an explicit save/restore rather than through
    ``monkeypatch``, on J7's recorded reason: a test that calls
    ``monkeypatch.undo()`` in the middle would otherwise silently
    uninstall the tracker with its injection and leave every live-storage
    assertion vacuous.
    """
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    cpp.NativeStorage.__init__ = tracked_init
    cpp.NativeStorage.close = tracked_close
    try:
        yield open_ids
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


def settled(live_storages):
    """The live-storage count after a collection. Collection *settles* the
    count; it is never the proof that anything was released — the harness
    closes what it owns explicitly."""
    gc.collect()
    return len(live_storages)


@needs_native
def test_the_live_storage_tracker_can_actually_notice(live_storages,
                                                      monkeypatch):
    """Negative control for the fixture, and therefore for every
    live-storage assertion below: it must survive a ``monkeypatch.undo()``
    and it must actually move when storage is allocated and released."""
    monkeypatch.setattr(bench, "SCHEMA_VERSION", 1)
    monkeypatch.undo()
    baseline = settled(live_storages)
    tensor = NativeTensor.from_array(np.zeros((2, 2)))
    assert settled(live_storages) == baseline + 1
    tensor.close()
    assert settled(live_storages) == baseline


class SpyTimer:
    """A ``measure`` replacement that records whether timing was reached."""

    def __init__(self, original=None):
        self.calls = []
        self.original = original

    def __call__(self, prepare, run, cleanup, warmup, repetitions):
        self.calls.append((warmup, repetitions))
        if self.original is None:
            raise RuntimeError("timer reached")
        return self.original(prepare, run, cleanup, warmup, repetitions)


# ===========================================================================
# 1. Import safety
# ===========================================================================


def test_importing_the_module_runs_nothing(capsys, live_storages):
    """Importing a benchmark must never execute one: no output, no native
    allocation, no dataset, no CLI parsing, and no file."""
    import importlib

    before = settled(live_storages)
    tree_before = _tree_fingerprint()
    rng_before = np.random.get_state()[1][:8].copy()
    importlib.reload(bench)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert settled(live_storages) == before
    assert np.array_equal(np.random.get_state()[1][:8], rng_before)
    assert _tree_fingerprint() == tree_before


def test_the_module_executes_only_through_main():
    """The only execution path is ``main()`` under a ``__main__`` guard —
    nothing at module scope runs a benchmark, parses an argument, prints,
    or builds a dataset."""
    tree = ast.parse(BENCHMARK_SOURCE)
    guards = [node for node in tree.body
              if isinstance(node, ast.If)
              and ast.unparse(node.test) == "__name__ == '__main__'"]
    assert len(guards) == 1, "there is no single __main__ guard"
    assert [ast.unparse(statement) for statement in guards[0].body] == [
        "main()"]
    # Only statements that actually execute at import time. A function or
    # class body runs when it is called, not when the module is read.
    called = []
    for node in tree.body:
        if node in guards or isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                               ast.Import, ast.ImportFrom)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                called.append(ast.unparse(inner.func))
    for banned in ("run_benchmark", "run_case", "measure", "main",
                   "build_parser", "parse_args", "print", "environment",
                   "NativeTensorDataset", "NativeBatchSampler",
                   "NativeDataLoader", "np.random.default_rng"):
        assert banned not in called, banned
    # ...and the scan really did look at something: the module-scope
    # bootstrap it is allowed to have is present in what it collected.
    assert "sys.path.insert" in called


def test_the_module_scope_scanner_can_actually_fail():
    """Negative control: the scanner must see a module-scope call when one
    is present."""
    planted = ast.parse("import os\nrun_benchmark()\n")
    called = [ast.unparse(inner.func)
              for node in planted.body
              for inner in ast.walk(node)
              if isinstance(inner, ast.Call)]
    assert "run_benchmark" in called


# ===========================================================================
# 2. Identity and schema
# ===========================================================================


def test_the_benchmark_identity_is_exact():
    assert bench.BENCHMARK_NAME == "tensorforge.native_data_pipeline"
    assert bench.BENCHMARK_VERSION == "1.0"
    assert bench.SCHEMA_VERSION == 1
    assert isinstance(bench.SCHEMA_VERSION, int)
    assert not isinstance(bench.SCHEMA_VERSION, bool)


def test_the_identity_is_not_a_package_export():
    """A benchmark is a measurement tool, never a capability. Nothing here
    may reach the public surface of either line."""
    import tensorforge
    import tensorforge.experimental as experimental

    for name in ("BENCHMARK_NAME", "BENCHMARK_VERSION", "SCHEMA_VERSION",
                 "CASES", "WORKLOADS", "run_benchmark", "measure",
                 "benchmark_native_data_pipeline"):
        assert name not in tensorforge.__all__, name
        assert name not in experimental.__all__, name
        assert not hasattr(experimental, name), name
    assert len(experimental.__all__) == 25
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "benchmark_native_data_pipeline" not in text, path.name
        assert "BENCHMARK_NAME" not in text, path.name


def test_the_dtype_axis_is_exactly_the_two_supported_widths():
    assert bench.DTYPES == ("float64", "float32")
    assert bench.DTYPES == cpp.SUPPORTED_DTYPES
    assert set(bench.NUMPY_DTYPES) == set(bench.DTYPES)
    assert set(bench.BIT_DTYPES) == set(bench.DTYPES)


@needs_native
def test_the_payload_root_and_case_schema_are_exact():
    payload = bench.run_benchmark(cases=["plan_sequential_exact"], **SMOKE)
    assert set(payload) == ROOT_KEYS
    assert payload["benchmark"] == bench.BENCHMARK_NAME
    assert payload["benchmark_version"] == bench.BENCHMARK_VERSION
    assert payload["schema_version"] == bench.SCHEMA_VERSION
    assert payload["mode"] == "smoke"
    for record in payload["cases"]:
        assert set(record) == CASE_KEYS, record["case"]
        assert set(record["statistics"]) == STATISTICS_KEYS, record["case"]
        gate = record["correctness"]["gate"]
        assert set(record["correctness"]) == GATE_KEYS[gate], record["case"]


@needs_native
def test_the_payload_round_trips_through_json_as_plain_python():
    payload = bench.run_benchmark(cases=["feature_batch_small"],
                                  dtypes=["float32"], **SMOKE)
    text = json.dumps(payload)
    assert json.loads(text) == payload
    _require_plain(payload)


def _require_plain(node, path="payload"):
    """Every value is a plain JSON type — no NumPy scalar, no tuple, no
    ndarray, and nothing callable."""
    if isinstance(node, dict):
        for key, value in node.items():
            assert type(key) is str, f"{path}: non-str key {key!r}"
            _require_plain(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _require_plain(value, f"{path}[{index}]")
    else:
        assert type(node) in (str, int, float, bool, type(None)), (
            f"{path}: {type(node).__name__}")


def test_the_plain_json_checker_can_actually_fail():
    """Negative control: a NumPy scalar or a tuple must be caught."""
    with pytest.raises(AssertionError):
        _require_plain({"value": np.float64(1.0)})
    with pytest.raises(AssertionError):
        _require_plain({"value": (1, 2)})
    with pytest.raises(AssertionError):
        _require_plain({1: "int key"})
    _require_plain({"ok": [1, 2.0, "three", True, None]})


# ===========================================================================
# 3. Case inventory
# ===========================================================================


def test_the_case_registry_is_exact_and_ordered():
    assert tuple(bench.CASES) == EXPECTED_CASES
    assert len(bench.CASES) == 20


def test_the_workload_registry_is_exact_and_every_family_is_populated():
    assert bench.WORKLOADS == EXPECTED_WORKLOADS
    populated = {spec["workload"] for spec in bench.CASES.values()}
    assert populated == set(EXPECTED_WORKLOADS), (
        f"declared but empty: {set(EXPECTED_WORKLOADS) - populated}")
    for workload, names in REQUIRED_COVERAGE.items():
        present = {name for name, spec in bench.CASES.items()
                   if spec["workload"] == workload}
        assert present == names, (workload, present ^ names)


def test_the_four_required_layers_are_separate_families():
    """J8's four questions must stay four measurements. A composed
    delivery case may exist beside them; it may not replace one."""
    for required in ("dataset_indexing", "batch_planning",
                     "permutation_construction",
                     "host_to_native_materialization"):
        assert required in bench.WORKLOADS
        assert bench.cases_for_workloads([required]), required
    # The composition is declared as its own family, not folded into one
    # of the four.
    assert bench.CASES["loader_next_batch"]["workload"] == "loader_delivery"


def test_every_case_declares_a_complete_auditable_specification():
    required_fields = (
        "workload", "label", "operation", "build", "gate", "reference_type",
        "native_only", "reference_detail", "ratio_meaning",
        "correctness_reference", "seed", "sampler_seed", "dtypes",
        "feature_shape", "classes", "shuffle", "drop_last", "cache_state",
        "fixed_configuration", "configurations", "setup", "timed",
        "cleanup", "notes",
    )
    for name, spec in bench.CASES.items():
        for field in required_fields:
            assert field in spec, (name, field)
        assert spec["label"] == name, name
        assert callable(spec["build"]), name
        assert spec["workload"] in bench.WORKLOADS, name
        assert spec["gate"] in bench.GATES, name
        assert spec["reference_type"] in bench.REFERENCE_TYPES, name
        assert spec["native_only"] is (
            spec["reference_type"] == bench.NATIVE_ONLY), name
        assert spec["cache_state"] in (None,) + bench.CACHE_STATES, name
        assert tuple(spec["dtypes"]) == bench.DTYPES, name
        assert isinstance(spec["seed"], int) and not isinstance(
            spec["seed"], bool), name
        assert spec["classes"] >= 1, name
        assert len(spec["feature_shape"]) >= 1, name
        assert all(dimension >= 1 for dimension in spec["feature_shape"]), name
        for text_field in ("operation", "reference_detail",
                           "correctness_reference", "setup", "timed",
                           "cleanup", "notes"):
            assert spec[text_field].strip(), (name, text_field)
        assert set(spec["configurations"]) == set(bench.CONFIGURATIONS), name


def test_every_case_carries_a_unique_deterministic_seed():
    seeds = [spec["seed"] for spec in bench.CASES.values()]
    assert len(set(seeds)) == len(seeds), "a data seed is shared"


def test_at_least_two_dataset_geometries_are_exercised():
    geometries = {tuple(spec["feature_shape"])
                  for spec in bench.CASES.values()}
    assert len(geometries) >= 2, geometries
    # One of them carries more than one value per sample at rank > 1, which
    # is what a convolutional model consumes.
    assert any(len(shape) > 1 for shape in geometries), geometries
    assert all(int(np.prod(shape)) > 1 for shape in geometries), geometries


def test_the_configurations_are_deterministic_positive_integers():
    for name, spec in bench.CASES.items():
        for variant in bench.CONFIGURATIONS:
            config = spec["configurations"][variant]
            assert config, (name, variant)
            for key, value in config.items():
                assert type(value) is int, (name, variant, key, type(value))
                assert value >= 1, (name, variant, key, value)


def test_smoke_is_smallest_and_profile_is_no_smaller_than_full():
    for name, spec in bench.CASES.items():
        smoke = spec["configurations"]["smoke"]
        full = spec["configurations"]["full"]
        profile = spec["configurations"]["profile"]
        assert set(smoke) == set(full) == set(profile), name
        for key in full:
            assert smoke[key] <= full[key] <= profile[key], (name, key)
        if spec["fixed_configuration"]:
            assert smoke == full == profile, name
        else:
            assert full != smoke, name


def test_multiple_dataset_sizes_and_batch_sizes_are_covered():
    samples = {spec["configurations"]["full"]["samples"]
               for spec in bench.CASES.values()}
    assert len(samples) >= 3, samples
    batch_sizes = {config["batch_size"]
                   for spec in bench.CASES.values()
                   for config in spec["configurations"].values()
                   if "batch_size" in config}
    assert len(batch_sizes) >= 2, batch_sizes


def test_exact_divisible_and_short_final_planning_are_both_covered():
    exact = bench.CASES["plan_sequential_exact"]["configurations"]
    short = bench.CASES["plan_sequential_short_final"]["configurations"]
    for variant in bench.CONFIGURATIONS:
        assert exact[variant]["samples"] % exact[variant]["batch_size"] == 0
        assert short[variant]["samples"] % short[variant]["batch_size"] != 0


def test_a_genuine_mid_epoch_planning_position_is_measured():
    spec = bench.CASES["next_batch_indices_mid_epoch"]
    for variant in bench.CONFIGURATIONS:
        config = spec["configurations"][variant]
        cursor = config["cursor"]
        batches = -(-config["samples"] // config["batch_size"])
        assert 0 < cursor < batches, (variant, cursor, batches)
    assert "cursor" not in bench.CASES["next_batch_indices_fresh"][
        "configurations"]["full"]


@needs_native
def test_every_declared_case_is_executable_and_no_other_is():
    payload = bench.run_benchmark(**SMOKE)
    produced = []
    for record in payload["cases"]:
        if record["case"] not in produced:
            produced.append(record["case"])
    assert tuple(produced) == EXPECTED_CASES
    for bad in ("no_such_case", "", "plan"):
        with pytest.raises(ValueError):
            bench.run_benchmark(cases=[bad], **SMOKE)
    with pytest.raises(ValueError):
        bench.run_benchmark(workloads=["no_such_workload"], **SMOKE)
    with pytest.raises(ValueError):
        bench.run_benchmark(cases=[], **SMOKE)


# ===========================================================================
# 4. Reference honesty
# ===========================================================================


def test_the_reference_registry_is_exactly_two_honest_labels():
    assert bench.REFERENCE_TYPES == ("numpy", "native_only")
    assert bench.REFERENCE_NUMPY == "numpy"
    assert bench.NATIVE_ONLY == "native_only"


def test_the_native_only_decisions_are_the_ones_j8_recorded():
    """Planning, permutation construction, materialization, and delivery
    have no honest equivalent; the two host-indexing families do."""
    for name, spec in bench.CASES.items():
        if spec["workload"] == "dataset_indexing":
            assert spec["reference_type"] == bench.REFERENCE_NUMPY, name
            assert spec["ratio_meaning"], name
        else:
            assert spec["reference_type"] == bench.NATIVE_ONLY, name
            assert spec["ratio_meaning"] is None, name
            assert "none" in spec["reference_detail"].lower(), name


def _ratio_is_honest(record):
    """Whether one payload record's ratio is defensible.

    A ``native_only`` record must publish no ratio and no reference
    timing; a record that does publish one must declare a reference type
    from the registry, carry reference statistics, and state what the
    ratio means. The dtype is a property of the whole record, so a ratio
    can only ever be same-dtype — and this checker refuses a record that
    tries to name a second one.
    """
    if record["native_only"] or record["reference_type"] == "native_only":
        return (record["ratio_to_reference"] is None
                and record["reference"] is None
                and record["ratio_meaning"] is None)
    if record["reference_type"] not in bench.REFERENCE_TYPES:
        return False
    if record["reference"] is None or record["ratio_to_reference"] is None:
        return False
    if not record["ratio_meaning"]:
        return False
    if record["dtype"] not in bench.DTYPES:
        return False
    # Nothing in the record may name the *other* width: a ratio that did
    # would be the cross-dtype comparison this project refuses to publish.
    other = [name for name in bench.DTYPES if name != record["dtype"]]
    blob = json.dumps({key: value for key, value in record.items()
                       if key in ("reference_detail", "ratio_meaning",
                                  "reference_type")})
    return not any(name in blob for name in other)


@needs_native
def test_every_published_ratio_is_honest_and_every_native_only_case_has_none():
    payload = bench.run_benchmark(**SMOKE)
    for record in payload["cases"]:
        assert _ratio_is_honest(record), record["case"]
        if record["native_only"]:
            assert record["ratio_to_reference"] is None, record["case"]
            assert record["reference"] is None, record["case"]
        else:
            assert record["ratio_to_reference"] > 0.0, record["case"]
            assert set(record["reference"]) == STATISTICS_KEYS


def test_the_ratio_honesty_checker_can_actually_fail():
    """Negative control: each dishonest shape must be rejected."""
    honest = {
        "native_only": False, "reference_type": "numpy", "dtype": "float64",
        "reference": {"median_ns": 1.0}, "ratio_to_reference": 1.5,
        "ratio_meaning": "measured median / reference median",
        "reference_detail": "an independently written numpy expression",
    }
    assert _ratio_is_honest(honest)
    # A native-only case that published a ratio anyway.
    assert not _ratio_is_honest({**honest, "native_only": True,
                                 "reference_type": "native_only"})
    # A ratio with no reference timing behind it.
    assert not _ratio_is_honest({**honest, "reference": None})
    # A ratio whose meaning is not stated.
    assert not _ratio_is_honest({**honest, "ratio_meaning": ""})
    # A ratio that names the other width — the cross-dtype comparison.
    assert not _ratio_is_honest({
        **honest, "ratio_meaning": "float64 median / float32 median"})
    # An invented reference label.
    assert not _ratio_is_honest({**honest, "reference_type": "pytorch"})


@needs_native
def test_a_numpy_reference_measures_the_same_inputs_at_the_same_dtype():
    """The reference is not merely declared — it is exercised on the same
    snapshot, the same indices, and the same dtype, and the gate proves
    the two agree before either is timed."""
    for name in ("host_feature_gather_sequential",
                 "dataset_target_batch_sequential"):
        spec = bench.CASES[name]
        for dtype in bench.DTYPES:
            case = spec["build"](dtype, spec["configurations"]["smoke"], spec)
            try:
                metrics = case.check()
                assert metrics["reference_agrees"] is True
                assert case.has_reference
                measured = case.run(case.prepare())
                reference = case.reference_run(case.reference_prepare())
                assert np.asarray(measured).dtype == np.asarray(
                    reference).dtype
                assert np.asarray(measured).shape == np.asarray(
                    reference).shape
                assert np.array_equal(measured, reference)
            finally:
                case.teardown()


def test_no_case_invents_a_reference_for_a_semantically_different_operation():
    """The materialization family allocates and transfers; a NumPy gather
    does not. Every one of those cases must therefore be native_only, and
    the source must say why rather than leaving it to be inferred."""
    for name, spec in bench.CASES.items():
        if spec["workload"] in ("host_to_native_materialization",
                                "loader_delivery"):
            assert spec["native_only"] is True, name
            assert "allocat" in spec["reference_detail"].lower() or (
                "transactional" in spec["reference_detail"].lower()), name
        if spec["workload"] == "permutation_construction":
            assert spec["native_only"] is True, name


# ===========================================================================
# 5. Correctness before timing
# ===========================================================================


@needs_native
@pytest.mark.parametrize("name", EXPECTED_CASES)
def test_every_case_gates_before_the_timer(name, monkeypatch):
    """If ``measure`` is reached at all, ``check()` has already returned.
    Proved by making the timer explode: the exception must come from the
    timer, which means the gate ran and passed first."""
    spy = SpyTimer()
    monkeypatch.setattr(bench, "measure", spy)
    with pytest.raises(RuntimeError, match="timer reached"):
        bench.run_case(name, "float64", 0, 1, "smoke")
    assert spy.calls == [(0, 1)]


@needs_native
def test_every_case_reports_a_passed_gate_at_each_dtype():
    payload = bench.run_benchmark(**SMOKE)
    assert payload["cases"], "the harness produced no rows"
    for record in payload["cases"]:
        assert record["correctness"]["passed"] is True, record["case"]
        assert record["correctness"]["gate"] in bench.GATES, record["case"]
        assert record["dtype"] in bench.DTYPES


@needs_native
@pytest.mark.parametrize("case_name, dtype", [
    ("next_batch_indices_fresh", "float64"),
    ("next_batch_indices_mid_epoch", "float32"),
])
def test_wrong_indices_fail_before_any_timing(case_name, dtype, monkeypatch,
                                              live_storages):
    """Injected wrong indices: the gate must catch a rotated plan slice,
    and no timing may exist for the case."""
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    original = NativeBatchSampler.next_batch_indices

    def rotated(self):
        indices = original(self)
        return indices[1:] + indices[:1]

    monkeypatch.setattr(NativeBatchSampler, "next_batch_indices", rotated)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError):
        bench.run_case(case_name, dtype, 0, 1, "smoke")
    assert spy.calls == [], "timing was reached despite a failed gate"
    assert settled(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("case_name", ["permutation_cold_reference",
                                       "permutation_cold_large"])
def test_a_wrong_permutation_fails_before_any_timing(case_name, monkeypatch,
                                                     live_storages):
    """Injected identity order: the committed reference vector rejects it
    for the small case, and the non-vacuity check rejects it for the
    large one."""
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)

    def identity(self, epoch=None):
        return tuple(range(self.dataset.samples))

    monkeypatch.setattr(NativeBatchSampler, "epoch_permutation", identity)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError):
        bench.run_case(case_name, "float64", 0, 1, "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_wrong_feature_bits_fail_before_any_timing(dtype, monkeypatch,
                                                   live_storages):
    """The negative control that makes the materialization gate
    load-bearing: perturb the native result and the case must fail in the
    gate, before ``measure`` is entered."""
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    original = cpp.NativeTensorCore.to_numpy

    def perturbed(self):
        return original(self) + np.asarray(
            1.0, dtype=bench.NUMPY_DTYPES[dtype])

    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", perturbed)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError):
        bench.run_case("feature_batch_small", dtype, 0, 1, "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


@needs_native
def test_wrong_targets_fail_before_any_timing(monkeypatch, live_storages):
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    original = NativeTensorDataset.target_batch

    def shifted(self, indices):
        return np.asarray(original(self, indices)) + 1

    monkeypatch.setattr(NativeTensorDataset, "target_batch", shifted)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError):
        bench.run_case("dataset_target_batch_sequential", "float64", 0, 1,
                       "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


@needs_native
def test_a_writeable_target_batch_fails_the_gate(monkeypatch):
    """The read-only half of the target contract is genuinely checked: a
    batch a caller could edit in place must be rejected."""
    original = NativeTensorDataset.target_batch

    def writeable(self, indices):
        return np.array(original(self, indices))

    monkeypatch.setattr(NativeTensorDataset, "target_batch", writeable)
    with pytest.raises(AssertionError, match="writeable"):
        bench.run_case("dataset_target_batch_sequential", "float64", 0, 1,
                       "smoke")


@needs_native
def test_a_wrong_dtype_result_is_caught_by_the_gate():
    """A gate that silently accepted the other width would let a case run
    at float64 while reporting float32 — the one mistake that would make
    the whole harness dishonest."""
    with pytest.raises(AssertionError, match="expected a float32 array"):
        bench.bits_of(np.ones(4, dtype=np.float64), "float32")
    with pytest.raises(AssertionError, match="expected a float64 array"):
        bench.bits_of(np.ones(4, dtype=np.float32), "float64")
    # ...and it accepts the matching width without converting anything.
    assert bench.bits_of(np.zeros(2, dtype=np.float32), "float32").dtype == (
        np.uint32)


@needs_native
def test_wrong_ownership_fails_before_any_timing(monkeypatch, live_storages):
    """A borrowed batch is not a materialization result: the caller could
    not close it, so the gate must refuse it."""
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    monkeypatch.setattr(NativeTensor, "owns_core",
                        property(lambda self: False))
    baseline = settled(live_storages)
    with pytest.raises(AssertionError, match="own its core"):
        bench.run_case("feature_batch_large", "float64", 0, 1, "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


@needs_native
def test_a_short_plan_fails_before_any_timing(monkeypatch, live_storages):
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    original = NativeBatchSampler.plan

    def truncated(self, epoch=None):
        return original(self, epoch)[:-1]

    monkeypatch.setattr(NativeBatchSampler, "plan", truncated)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError):
        bench.run_case("plan_sequential_exact", "float64", 0, 1, "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


@needs_native
def test_a_failed_delivery_gate_still_cleans_up(monkeypatch, live_storages):
    """A gate failure inside the delivery transaction must still release
    everything: the undelivered tensor, the iterator, the loader, and the
    dataset."""
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    original = NativeTensorDataset.target_batch

    def shifted(self, indices):
        return np.asarray(original(self, indices)) + 1

    # Deliberately *not* an index injection: the iterator plans through
    # the same call the gate records its expectation from, so a rotated
    # plan would be invisible to both alike. Perturbing the delivered
    # labels is a fault the transaction really can be caught committing.
    monkeypatch.setattr(NativeTensorDataset, "target_batch", shifted)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError, match="delivered targets"):
        bench.run_case("loader_next_batch", "float64", 0, 1, "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


# ===========================================================================
# 6. What each family actually proves
# ===========================================================================


@needs_native
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_dataset_indexing_is_host_only_and_allocates_nothing_native(
        dtype, live_storages):
    """The indexing family measures the host layer. Its timed call must
    allocate no native storage at all — that is the whole reason the
    NumPy reference is honest there and dishonest in the materialization
    family."""
    for name in ("host_feature_gather_sequential",
                 "host_feature_gather_shuffled",
                 "host_feature_gather_duplicates",
                 "dataset_target_batch_sequential",
                 "dataset_target_batch_shuffled"):
        spec = bench.CASES[name]
        case = spec["build"](dtype, spec["configurations"]["smoke"], spec)
        try:
            case.check()
            baseline = settled(live_storages)
            for _ in range(3):
                state = case.prepare()
                result = case.run(state)
                assert settled(live_storages) == baseline, name
                case.cleanup(state, result)
            assert settled(live_storages) == baseline, name
        finally:
            case.teardown()


@needs_native
def test_dataset_indexing_preserves_order_and_duplicates_exactly():
    spec = bench.CASES["host_feature_gather_duplicates"]
    config = spec["configurations"]["smoke"]
    dataset, features, _ = bench.build_dataset("float64", config, spec)
    try:
        wanted = list(bench.index_set("duplicates", dataset, config["batch"],
                                      spec))
        assert len(set(wanted)) < len(wanted), "no duplicate was produced"
        snapshot = np.array(features, dtype=np.float64, order="C", copy=True)
        gathered = snapshot[wanted]
        for position, index in enumerate(wanted):
            assert np.array_equal(bench.bits_of(gathered[position],
                                                "float64"),
                                  bench.bits_of(snapshot[index], "float64"))
        # A deduplicated or sorted gather is a different answer and the
        # gate must be able to tell.
        assert len(gathered) == len(wanted)
        assert not np.array_equal(gathered, snapshot[sorted(set(wanted))])
    finally:
        dataset.close()


@needs_native
def test_the_dataset_is_not_mutated_by_indexing():
    spec = bench.CASES["dataset_target_batch_shuffled"]
    config = spec["configurations"]["smoke"]
    dataset, _, _ = bench.build_dataset("float32", config, spec)
    try:
        before = dataset.identity()
        wanted = list(bench.index_set("shuffled", dataset, config["batch"],
                                      spec))
        for _ in range(3):
            dataset.target_batch(wanted)
        assert dataset.identity() == before
    finally:
        dataset.close()


@needs_native
@pytest.mark.parametrize("name", ["plan_sequential_exact",
                                  "plan_sequential_short_final",
                                  "plan_shuffled_reference",
                                  "plan_shuffled_large",
                                  "next_batch_indices_fresh",
                                  "next_batch_indices_mid_epoch"])
def test_planning_moves_no_position_and_allocates_nothing(name,
                                                          live_storages):
    spec = bench.CASES[name]
    case = spec["build"]("float64", spec["configurations"]["smoke"], spec)
    try:
        case.check()
        baseline = settled(live_storages)
        sampler = case.prepare()
        first = case.run(sampler)
        second = case.run(case.prepare())
        assert first == second, name
        assert settled(live_storages) == baseline, name
    finally:
        case.teardown()


@needs_native
def test_the_committed_reference_plan_is_what_the_sampler_plans():
    """The known answer, from design §8.9, checked against the live
    planner rather than against the harness's own arithmetic."""
    spec = bench.CASES["plan_shuffled_reference"]
    config = spec["configurations"]["full"]
    dataset, _, _ = bench.build_dataset("float64", config, spec)
    try:
        sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True,
                                     seed=7)
        assert sampler.plan() == bench.REFERENCE_PLAN
        assert sampler.plan() == ((7, 5, 4), (0, 1, 3), (6, 2))
        assert sampler.epoch_permutation() == (7, 5, 4, 0, 1, 3, 6, 2)
    finally:
        dataset.close()


@needs_native
def test_the_committed_reference_permutations_match_the_live_sampler():
    for (length, seed, epoch), expected in bench.REFERENCE_PERMUTATIONS.items():
        features = np.zeros((length, 2), dtype=np.float64)
        targets = np.zeros(length, dtype=np.int64)
        dataset = NativeTensorDataset(features, targets, dtype="float64")
        try:
            sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True,
                                         seed=seed)
            assert sampler.epoch_permutation(epoch) == expected, (length,
                                                                  seed, epoch)
        finally:
            dataset.close()


@needs_native
def test_a_cold_permutation_case_is_genuinely_cold():
    """Structural: the sampler a cold repetition receives has an empty
    permutation cache, so the timed call must construct one. Read through
    the sampler's private cache field, which is the only place the state
    is observable — no cache-control API exists and J8 adds none."""
    spec = bench.CASES["permutation_cold_large"]
    case = spec["build"]("float64", spec["configurations"]["smoke"], spec)
    try:
        first = case.prepare()
        second = case.prepare()
        assert first is not second, "a cold case reused its sampler"
        assert first._cache_key is None
        assert first._cache_order is None
        order = case.run(first)
        assert first._cache_key is not None, (
            "the timed call did not populate the cache, so it did not "
            "construct the order")
        assert case.run(second) == order
    finally:
        case.teardown()


@needs_native
def test_the_warm_permutation_case_is_genuinely_a_cache_hit():
    """The warm case's prepare() populates the cache outside the timer, so
    the timed call returns the *same tuple object* — which only a hit can
    do. Cold and warm are separate cases and are never averaged."""
    spec = bench.CASES["permutation_cache_hit"]
    assert spec["cache_state"] == bench.WARM
    case = spec["build"]("float64", spec["configurations"]["smoke"], spec)
    try:
        sampler = case.prepare()
        assert sampler._cache_key is not None, "prepare() did not warm"
        first = case.run(sampler)
        second = case.run(case.prepare())
        assert first is second, "the warm call recomputed instead of hitting"
    finally:
        case.teardown()
    cold = {name for name, spec in bench.CASES.items()
            if spec["cache_state"] == bench.COLD}
    warm = {name for name, spec in bench.CASES.items()
            if spec["cache_state"] == bench.WARM}
    assert cold and warm and not (cold & warm)


def test_the_harness_adds_no_cache_control_surface():
    """Dropping the cache must never become an API, a statistic, or a
    benchmark-only production branch."""
    for banned in ("clear_cache", "reset_cache", "cache_info", "cache_stats",
                   "_cache_key", "_cache_order", "drop_cache",
                   "invalidate"):
        assert banned not in BENCHMARK_CODE, banned
    for surface in (NativeBatchSampler, NativeTensorDataset,
                    NativeDataLoader):
        for banned in ("clear_cache", "reset_cache", "cache_info",
                       "cache_stats", "benchmark", "profile"):
            assert not hasattr(surface, banned), (surface.__name__, banned)


@needs_native
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_materialization_returns_fresh_owning_storage_at_both_dtypes(
        dtype, live_storages):
    spec = bench.CASES["feature_batch_image"]
    config = spec["configurations"]["smoke"]
    dataset, features, _ = bench.build_dataset(dtype, config, spec)
    try:
        wanted = list(range(config["batch"]))
        snapshot = np.array(features, dtype=bench.NUMPY_DTYPES[dtype],
                            order="C", copy=True)
        baseline = settled(live_storages)
        first = dataset.feature_batch(wanted)
        second = dataset.feature_batch(wanted)
        try:
            assert first.dtype == dtype and second.dtype == dtype
            assert first.device == "cpu"
            assert first.owns_core and first.contiguous
            assert first.shape == (config["batch"],) + spec["feature_shape"]
            assert np.array_equal(
                bench.bits_of(first.to_numpy(), dtype),
                bench.bits_of(snapshot[wanted], dtype))
            assert first is not second
            assert settled(live_storages) == baseline + 2
        finally:
            first.close()
            second.close()
        assert settled(live_storages) == baseline
    finally:
        dataset.close()


@needs_native
def test_the_delivery_case_restores_its_position_between_repetitions():
    """A stateful call must start every repetition from the same place, or
    supposedly identical repetitions are not identical."""
    spec = bench.CASES["loader_next_batch"]
    assert spec["cache_state"] == bench.WARM
    case = spec["build"]("float64", spec["configurations"]["smoke"], spec)
    try:
        case.check()
        positions = []
        delivered = []
        for _ in range(3):
            iterator = case.prepare()
            sampler = iterator._sampler
            # The warm claim, proved rather than declared: a state load
            # invalidates the permutation cache, so setup must repopulate
            # it or the timed delivery would be measuring an epoch's
            # permutation construction under a delivery label.
            assert sampler._cache_key is not None, (
                "the delivery case entered the timer on a cold cache")
            positions.append((sampler.epoch, sampler.cursor))
            result = case.run(iterator)
            delivered.append(bench.bits_of(result[0].to_numpy(),
                                           "float64").tolist())
            case.cleanup(iterator, result)
        assert positions == [(0, 0)] * 3, positions
        assert delivered[0] == delivered[1] == delivered[2]
    finally:
        case.teardown()


# ===========================================================================
# 7. Timing methodology
# ===========================================================================


def test_one_measured_sample_is_exactly_one_operation():
    calls = {"prepare": 0, "run": 0, "cleanup": 0}

    def prepare():
        calls["prepare"] += 1
        return calls["prepare"]

    def run(state):
        calls["run"] += 1
        return state

    def cleanup(state, result):
        calls["cleanup"] += 1

    samples = bench.measure(prepare, run, cleanup, warmup=2, repetitions=5)
    assert len(samples) == 5
    assert calls == {"prepare": 7, "run": 7, "cleanup": 7}


def test_warmup_samples_are_excluded_and_every_measured_sample_is_kept(
        monkeypatch):
    """A deterministic clock proves both halves at once: exactly the
    measured intervals come back, in order, including a deliberate
    outlier that must not be trimmed."""
    ticks = iter([
        # two warm-up repetitions are timed by nothing at all
        # (the timer is only read inside the measured loop)
        0, 10,            # sample 1 -> 10
        100, 130,         # sample 2 -> 30
        1000, 1_000_000,  # sample 3 -> 999000, a deliberate outlier
        5000, 5007,       # sample 4 -> 7
    ])
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: next(ticks))
    samples = bench.measure(lambda: None, lambda state: None,
                            lambda state, result: None,
                            warmup=2, repetitions=4)
    assert samples == [10, 30, 999_000, 7]
    assert max(samples) in samples, "the outlier was removed"


def test_no_timer_overhead_is_subtracted():
    """Structural, over the harness's own **code**: the measured interval
    is a bare difference of two reads, appended raw."""
    assert "elapsed = time.perf_counter_ns() - start" in BENCHMARK_CODE
    assert "samples.append(elapsed)" in BENCHMARK_CODE
    for banned in ("elapsed -", "elapsed /", "elapsed *", "overhead",
                   "calibrat", "trim", "outlier", "winsor"):
        assert banned not in BENCHMARK_CODE, banned


@needs_native
def test_setup_and_cleanup_are_outside_the_timed_region(monkeypatch):
    """Behavioural, not structural: a clock that only advances during
    ``run`` must produce non-zero samples, and one that only advances
    during ``prepare``/``cleanup`` must produce zeros."""
    phase = {"name": "outside"}
    clock = {"value": 0}

    def tick():
        if phase["name"] == "run":
            clock["value"] += 5
        else:
            clock["value"] += 1_000_000
        return clock["value"]

    monkeypatch.setattr(bench.time, "perf_counter_ns", tick)

    def prepare():
        phase["name"] = "prepare"
        return None

    def run(state):
        phase["name"] = "run"
        return None

    def cleanup(state, result):
        phase["name"] = "cleanup"

    samples = bench.measure(prepare, run, cleanup, warmup=1, repetitions=3)
    # Two reads bracket the call and nothing else happens between them.
    assert samples == [5, 5, 5], samples


@needs_native
def test_the_effective_counts_are_reported_on_every_record():
    payload = bench.run_benchmark(cases=["plan_sequential_exact"],
                                  warmup=2, repetitions=5, **SMOKE)
    for record in payload["cases"]:
        assert record["warmup"] == 2
        assert record["statistics"]["sample_count"] == 5
        assert len(record["statistics"]["samples_ns"]) == 5
    assert payload["methodology"]["warmup"] == 2
    assert payload["methodology"]["repetitions"] == 5


@needs_native
def test_the_repetition_and_warmup_arguments_are_validated():
    for bad in (0, -1, 1.5, True, "3"):
        with pytest.raises(ValueError):
            bench.run_benchmark(repetitions=bad, **SMOKE)
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError):
            bench.run_benchmark(warmup=bad, **SMOKE)
    # Zero warm-up is legal; zero repetitions is not.
    payload = bench.run_benchmark(cases=["plan_shuffled_reference"],
                                  warmup=0, repetitions=1, **SMOKE)
    assert payload["cases"][0]["warmup"] == 0


def test_every_case_declares_its_timed_boundary_setup_and_cleanup():
    for name, spec in bench.CASES.items():
        assert "timer" in spec["setup"] or "outside" in spec["setup"], name
        assert spec["timed"].startswith("exactly one"), name
        assert spec["cleanup"].strip(), name
        if spec["workload"] in ("host_to_native_materialization",
                                "loader_delivery"):
            assert "clos" in spec["cleanup"], name
            assert "outside the timer" in spec["cleanup"], name


# ===========================================================================
# 8. Statistics
# ===========================================================================


def test_the_percentile_rule_has_known_answers():
    assert bench.percentile([10], 0.25) == 10.0
    assert bench.percentile([10], 0.75) == 10.0
    # Even count: q * (n - 1) lands between two order statistics.
    assert bench.percentile([1, 2, 3, 4], 0.25) == pytest.approx(1.75)
    assert bench.percentile([1, 2, 3, 4], 0.75) == pytest.approx(3.25)
    # Odd count.
    assert bench.percentile([1, 2, 3], 0.25) == pytest.approx(1.5)
    assert bench.percentile([1, 2, 3], 0.75) == pytest.approx(2.5)
    # Exactly on an order statistic.
    assert bench.percentile([0, 4, 8, 12, 16], 0.25) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        bench.percentile([], 0.5)


def test_the_summary_has_known_answers_at_odd_and_even_counts():
    odd = bench.summarize([30, 10, 20])
    assert odd["sample_count"] == 3
    assert odd["median_ns"] == 20.0
    assert odd["min_ns"] == 10.0 and odd["max_ns"] == 30.0
    assert odd["p25_ns"] == pytest.approx(15.0)
    assert odd["p75_ns"] == pytest.approx(25.0)
    assert odd["iqr_ns"] == pytest.approx(10.0)
    assert odd["mean_ns"] == pytest.approx(20.0)
    assert odd["relative_iqr"] == pytest.approx(0.5)
    even = bench.summarize([4, 1, 3, 2])
    assert even["sample_count"] == 4
    assert even["median_ns"] == pytest.approx(2.5)
    assert even["p25_ns"] == pytest.approx(1.75)
    assert even["p75_ns"] == pytest.approx(3.25)
    assert even["iqr_ns"] == pytest.approx(1.5)
    single = bench.summarize([7])
    assert single["iqr_ns"] == 0.0
    assert single["median_ns"] == 7.0


def test_the_summary_retains_every_raw_sample_in_collection_order():
    samples = [50, 10, 999999, 20, 30]
    summary = bench.summarize(samples)
    assert summary["samples_ns"] == samples, "samples were reordered or lost"
    assert summary["sample_count"] == len(samples)
    assert 999999 in summary["samples_ns"], "an outlier was removed"
    assert summary["max_ns"] == 999999.0


def test_the_headline_and_spread_statistics_are_named_not_implied():
    summary = bench.summarize([1, 2, 3, 4, 5])
    assert summary["headline_statistic"] == "median"
    assert summary["spread_statistic"] == "interquartile range (p75 - p25)"
    assert summary["units"] == "nanoseconds_per_call"
    # A mean is present as secondary information, and is not the headline.
    assert "mean_ns" in summary


@needs_native
def test_no_measured_value_carries_a_pass_fail_meaning():
    """Nothing in the payload classifies a duration or a ratio. A field
    that did would be a threshold in prose."""
    payload = bench.run_benchmark(cases=["feature_batch_small"], **SMOKE)
    blob = json.dumps(payload).lower()
    for banned in ("regression", "acceptable", "budget", "threshold exceeded",
                   "too slow", "passed timing", "failed timing", "target_ns",
                   "limit_ns", "max_ns_allowed"):
        assert banned not in blob, banned
    for record in payload["cases"]:
        stats = record["statistics"]
        assert stats["min_ns"] <= stats["median_ns"] <= stats["max_ns"]
        assert stats["iqr_ns"] >= 0.0


# ===========================================================================
# 9. Dtype separation
# ===========================================================================


@needs_native
def test_both_dtypes_are_measured_and_every_case_appears_at_each():
    payload = bench.run_benchmark(**SMOKE)
    assert payload["dtypes"] == ["float64", "float32"]
    by_dtype = {}
    for record in payload["cases"]:
        by_dtype.setdefault(record["dtype"], set()).add(record["case"])
    assert set(by_dtype) == {"float64", "float32"}
    assert by_dtype["float64"] == by_dtype["float32"] == set(EXPECTED_CASES)


@needs_native
def test_no_cross_dtype_ratio_appears_anywhere_in_the_payload():
    payload = bench.run_benchmark(**SMOKE)
    banned_keys = _cross_dtype_key_offenders(payload)
    assert banned_keys == [], banned_keys
    blob = json.dumps(payload).lower()
    for banned in ("speedup", "vs_float", "faster than", "x faster",
                   "float32/float64 ratio of", "improvement over",
                   "float64_over_float32", "float32_over_float64"):
        assert banned not in blob, banned


def _cross_dtype_key_offenders(payload):
    """Every payload key that names a dtype comparison.

    Keys are compared **token by token** rather than as substrings:
    ``configuration`` innocently contains ``ratio``'s letters in other
    words, and a checker that tripped on prose would be noise rather than
    a guardrail. A key naming *both* widths, or naming a speed verdict, is
    the real offence.
    """
    offenders = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                tokens = set(key.lower().replace("-", "_").split("_"))
                if {"float32", "float64"} <= tokens:
                    offenders.append(key)
                if tokens & {"speedup", "faster", "slower", "wins",
                             "improvement", "regression"}:
                    offenders.append(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return offenders


def test_the_cross_dtype_key_scanner_can_actually_fail():
    """Negative control: a fabricated payload naming both widths, or
    naming a verdict, must be caught — and the honest keys the payload
    really uses must not be."""
    assert _cross_dtype_key_offenders(
        {"float32_over_float64": 1.2}) == ["float32_over_float64"]
    assert _cross_dtype_key_offenders(
        {"cases": [{"speedup": 2.0}]}) == ["speedup"]
    assert _cross_dtype_key_offenders(
        {"deep": {"nested": {"regression_ns": 5}}}) == ["regression_ns"]
    assert _cross_dtype_key_offenders({
        "ratio_to_reference": 1.0, "ratio_meaning": "x", "dtype": "float64",
        "configuration": "smoke", "reference_type": "numpy",
    }) == []


@needs_native
def test_the_report_publishes_no_dtype_speed_claim():
    payload = bench.run_benchmark(cases=["feature_batch_small"], **SMOKE)
    report = bench.format_report(payload).lower()
    assert "float32" in report and "float64" in report
    for banned in ("faster", "slower", "speedup", "outperform", "beats",
                   "wins", "loses", "acceptable", "budget", "regression"):
        assert banned not in report, banned
    assert "no" in report and "float32/float64 ratio" in report


@needs_native
def test_the_one_allowed_cross_dtype_claim_is_the_index_sequence():
    """Index and permutation sequences carry no dtype, so two datasets of
    the same length may legitimately be proved to plan identically. That
    is the *only* cross-dtype claim this harness makes, and it is a
    correctness claim rather than a timing one."""
    spec = bench.CASES["plan_shuffled_large"]
    config = spec["configurations"]["smoke"]
    plans = {}
    orders = {}
    for dtype in bench.DTYPES:
        dataset, _, _ = bench.build_dataset(dtype, config, spec)
        try:
            sampler = NativeBatchSampler(dataset,
                                         batch_size=config["batch_size"],
                                         shuffle=True,
                                         seed=spec["sampler_seed"])
            plans[dtype] = sampler.plan()
            orders[dtype] = sampler.epoch_permutation()
        finally:
            dataset.close()
    assert plans["float64"] == plans["float32"]
    assert orders["float64"] == orders["float32"]


@needs_native
def test_each_dtype_is_gated_independently():
    """The gate compares a result only against a reference **at its own
    width**. A float32 case must never be validated against float64
    values."""
    payload = bench.run_benchmark(cases=["feature_batch_small"], **SMOKE)
    for record in payload["cases"]:
        assert record["dtype"] in bench.DTYPES
        assert record["correctness"]["passed"] is True
    # No widening conversion and no tolerance is used to make a comparison
    # succeed. Scanned over the code, not the prose that explains the rule.
    for banned in ("astype", "allclose", "isclose", "approx", "atol",
                   "rtol", "round(", "np.float64(", "np.float32("):
        assert banned not in BENCHMARK_CODE, banned
    # The one dtype conversion the harness performs is at input
    # construction, before any dataset exists, and it is explicit.
    assert "ascontiguousarray(values, dtype=NUMPY_DTYPES[dtype])" in (
        BENCHMARK_CODE)


# ===========================================================================
# 10. The CLI
# ===========================================================================


def run_cli(*arguments, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, str(BENCHMARK_FILE), *arguments],
        capture_output=True, text=True, timeout=1800,
        cwd=str(cwd or REPO_ROOT), env=env)


@needs_native
def test_cli_smoke_json_parses_and_keeps_stdout_clean():
    result = run_cli("--smoke", "--json")
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout)
    assert set(payload) == ROOT_KEYS
    assert payload["benchmark"] == bench.BENCHMARK_NAME
    assert payload["mode"] == "smoke"
    assert len(payload["cases"]) == 2 * len(EXPECTED_CASES)
    # One JSON object and nothing else: no prose, no banner, no progress.
    assert result.stdout.strip().startswith("{")
    assert result.stdout.strip().endswith("}")
    assert result.stdout.count("\n") == 1


@needs_native
def test_cli_smoke_human_output_is_readable_and_carries_the_disclaimer():
    result = run_cli("--smoke", "--case", "feature_batch_small")
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert "Local characterization only" in out
    assert "no result file is written" in out
    assert "measured separately" in out
    assert "float64" in out and "float32" in out
    assert "native_only" in out
    assert str(REPO_ROOT) not in out


@needs_native
def test_cli_default_mode_runs_the_full_configuration():
    result = run_cli("--case", "plan_shuffled_reference", "--json")
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout)
    assert payload["mode"] == "full"
    assert payload["methodology"]["repetitions"] == bench.DEFAULTS[
        "repetitions"]


@needs_native
def test_cli_selects_exactly_one_case_and_exactly_one_workload():
    single = run_cli("--smoke", "--json", "--case", "permutation_cache_hit")
    assert single.returncode == 0, single.stderr[-2000:]
    payload = json.loads(single.stdout)
    assert payload["selected_cases"] == ["permutation_cache_hit"]
    assert {row["case"] for row in payload["cases"]} == {
        "permutation_cache_hit"}
    family = run_cli("--smoke", "--json", "--workload", "batch_planning")
    assert family.returncode == 0, family.stderr[-2000:]
    payload = json.loads(family.stdout)
    assert set(payload["selected_cases"]) == REQUIRED_COVERAGE[
        "batch_planning"]
    assert payload["selected_workloads"] == ["batch_planning"]


@needs_native
def test_cli_selects_a_single_dtype():
    result = run_cli("--smoke", "--json", "--case", "feature_batch_small",
                     "--dtype", "float32")
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout)
    assert payload["dtypes"] == ["float32"]
    assert {row["dtype"] for row in payload["cases"]} == {"float32"}


@needs_native
def test_cli_profile_mode_runs_one_case_at_no_smaller_a_shape():
    result = run_cli("--json", "--profile", "plan_shuffled_reference")
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout)
    assert payload["mode"] == "profile"
    assert payload["selected_cases"] == ["plan_shuffled_reference"]
    assert payload["methodology"]["repetitions"] == bench.PROFILE_DEFAULTS[
        "repetitions"]


@pytest.mark.parametrize("arguments", [
    ("--case", "no_such_case"),
    ("--workload", "no_such_workload"),
    ("--dtype", "float16"),
    ("--case", "plan_sequential_exact", "--workload", "batch_planning"),
    ("--profile", "plan_sequential_exact", "--case", "plan_sequential_exact"),
    ("--profile", "plan_sequential_exact", "--smoke"),
])
def test_cli_rejects_bad_selections_without_polluting_stdout(arguments):
    result = run_cli(*arguments)
    assert result.returncode != 0
    assert result.stdout.strip() == ""
    assert result.stderr.strip() != ""


def test_cli_help_exits_zero_and_offers_no_output_path():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--smoke" in result.stdout and "--json" in result.stdout
    for banned in ("--save", "--output", "--out", "--baseline", "--compare",
                   "--report", "--csv", "--file", "--results", "--check"):
        assert banned not in result.stdout, banned


@needs_native
def test_cli_reports_a_correctness_failure_with_a_nonzero_exit(tmp_path):
    """A failed gate exits nonzero with clean stdout — arranged from
    outside, through a sitecustomize shim, so **no repository source is
    modified** to build the control."""
    shim = tmp_path / "sitecustomize.py"
    shim.write_text(
        "import numpy as np\n"
        "from tensorforge.experimental import NativeTensorDataset\n"
        "_original = NativeTensorDataset.target_batch\n"
        "def _writeable(self, indices):\n"
        "    return np.array(_original(self, indices))\n"
        "NativeTensorDataset.target_batch = _writeable\n",
        encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(REPO_ROOT)
    result = run_cli("--smoke", "--case", "dataset_target_batch_sequential",
                     env=environment)
    assert result.returncode != 0
    assert result.stdout.strip() == ""
    assert "correctness gate failed" in result.stderr


@needs_native
def test_the_unbuilt_backend_follows_the_benchmark_convention(monkeypatch,
                                                              capsys):
    monkeypatch.setattr(cpp, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not built"):
        bench.run_benchmark(**SMOKE)
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--smoke"])
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not built" in captured.err


# ===========================================================================
# 11. No result file, ever
# ===========================================================================


_WATCHED = ("", "benchmarks", "tests", "docs", "examples")


def _tree_fingerprint():
    return {name: sorted(entry.name
                         for entry in (REPO_ROOT / name).iterdir()
                         if entry.name != "__pycache__")
            for name in _WATCHED}


@needs_native
def test_running_the_benchmark_writes_no_file():
    before = _tree_fingerprint()
    bench.run_benchmark(cases=["feature_batch_small"], **SMOKE)
    assert _tree_fingerprint() == before


@needs_native
def test_the_cli_writes_no_file_in_any_mode(tmp_path):
    """Run from an empty working directory and prove nothing appeared —
    there and in the repository."""
    before = _tree_fingerprint()
    for arguments in (("--smoke", "--case", "plan_shuffled_reference"),
                      ("--smoke", "--json", "--case",
                       "permutation_cold_reference")):
        result = run_cli(*arguments, cwd=tmp_path)
        assert result.returncode == 0, result.stderr[-2000:]
    assert list(tmp_path.iterdir()) == [], "the harness wrote a file"
    assert _tree_fingerprint() == before


def test_no_result_artifact_of_any_kind_is_tracked():
    for pattern in ("*.json", "*.csv", "*.txt", "*.md", "*.npz", "*.png",
                    "*.svg", "*.pkl", "*.db"):
        assert not list((REPO_ROOT / "benchmarks").glob(pattern)), pattern
    assert not list(REPO_ROOT.glob("benchmark*.json"))
    assert not (REPO_ROOT / "benchmark_results").exists()
    assert not (REPO_ROOT / "benchmarks" / "results").exists()
    assert not list((REPO_ROOT / "docs").glob("*data_pipeline*.json"))


def test_the_benchmark_opens_no_file_and_imports_no_writer():
    for banned in ("write_text", "write_bytes", "read_bytes", "os.makedirs",
                   "mkdir", "savez", "np.save", "to_csv", "csv.",
                   "pickle", "shelve", "sqlite", "tempfile", "shutil",
                   "NamedTemporary", "mkdtemp", "json.dump(",
                   "urllib", "requests", "socket", "download"):
        assert banned not in BENCHMARK_CODE, banned
    # File opening, precisely: bare ``open(``, ``io.open(``, and
    # ``something.open(`` all match, while an identifier that merely ends
    # in "open" does not.
    assert not re.search(r"(?<![_\w])open\(", BENCHMARK_CODE), "open("
    # json.dumps (a string) is fine; json.dump (a file) is not.
    assert "json.dumps(payload)" in BENCHMARK_CODE
    # ``os`` is read for environment variables and the CPU count only.
    assert set(re.findall(r"\bos\.\w+", BENCHMARK_CODE)) == {
        "os.environ", "os.cpu_count"}


def test_the_file_writer_scanner_can_actually_fail():
    """Negative control: the scans above must catch a planted writer."""
    planted = "with open('out.json', 'w') as handle:\n    handle.write('x')\n"
    assert re.search(r"(?<![_\w])open\(", planted)
    assert "write" in planted
    # ...and an identifier merely ending in "open" is not a false positive.
    assert not re.search(r"(?<![_\w])open\(", "self._require_open()")


def test_no_cli_option_could_ask_for_a_file():
    parser = bench.build_parser()
    options = {action.dest for action in parser._actions}
    for banned in ("output", "out", "outfile", "results", "save", "report",
                   "csv", "path", "file", "baseline", "compare", "check",
                   "threshold", "budget", "fail_under"):
        assert banned not in options, banned
    assert options == {"help", "case", "workload", "dtype", "warmup",
                       "repetitions", "json", "smoke", "profile"}


# ===========================================================================
# 12. Determinism, environment metadata, and cleanup
# ===========================================================================


def test_the_benchmark_uses_only_local_seeded_generators():
    assert "np.random.default_rng(seed)" in BENCHMARK_CODE
    for banned in ("np.random.seed", "np.random.rand", "np.random.randn",
                   "np.random.normal", "np.random.permutation",
                   "np.random.shuffle", "random.seed", "random.random",
                   "import random", "secrets", "uuid", "time.time",
                   "datetime", "getpid", "os.urandom", "hash("):
        assert banned not in BENCHMARK_CODE, banned


def test_the_global_rng_scanner_can_actually_fail():
    planted = "values = np.random.rand(4)\n"
    assert "np.random.rand" in planted


@needs_native
def test_running_the_benchmark_moves_no_global_rng_state():
    numpy_before = np.random.get_state()[1][:8].copy()
    import random as _random
    python_before = _random.getstate()
    bench.run_benchmark(cases=["host_feature_gather_shuffled"], **SMOKE)
    assert np.array_equal(np.random.get_state()[1][:8], numpy_before)
    assert _random.getstate() == python_before


@needs_native
def test_repeated_setup_builds_equal_but_independent_objects():
    spec = bench.CASES["feature_batch_small"]
    config = spec["configurations"]["smoke"]
    first, features_a, targets_a = bench.build_dataset("float64", config,
                                                       spec)
    second, features_b, targets_b = bench.build_dataset("float64", config,
                                                        spec)
    try:
        assert first is not second
        assert first.identity() == second.identity()
        assert np.array_equal(features_a, features_b)
        assert np.array_equal(targets_a, targets_b)
        assert not np.shares_memory(features_a, features_b)
        first.close()
        assert second.closed is False
    finally:
        if not first.closed:
            first.close()
        second.close()


@needs_native
def test_repeated_runs_produce_identical_correctness_metrics():
    """Deterministic inputs mean deterministic gates; only the timings may
    move between runs."""
    def fingerprint(payload):
        return [(row["case"], row["dtype"], row["correctness"],
                 row["config"], row["seed"])
                for row in payload["cases"]]

    first = bench.run_benchmark(workloads=["batch_planning"], **SMOKE)
    second = bench.run_benchmark(workloads=["batch_planning"], **SMOKE)
    assert fingerprint(first) == fingerprint(second)


@needs_native
def test_the_environment_is_real_introspection_and_carries_no_identity():
    info = bench.environment()
    expected = {"platform", "machine", "processor", "architecture_bits",
                "python_version", "python_implementation", "numpy_version",
                "tensorforge_version", "cpu_count_logical",
                "backend_available", "thread_environment", "native_backend"}
    assert set(info) == expected
    assert info["python_version"] == ".".join(
        str(part) for part in sys.version_info[:3])
    assert info["numpy_version"] == np.__version__
    backend = info["native_backend"]
    assert backend["supported_dtypes"] == list(cpp.SUPPORTED_DTYPES)
    assert backend["dtype"] == cpp.backend_info()["dtype"]
    assert backend["raw_kernel_dtypes"] == list(cpp.RAW_KERNEL_DTYPES)
    assert backend["stable_framework_integration"] is False
    # Threading variables appear only when they are actually set.
    for name, value in info["thread_environment"].items():
        assert name in bench.THREAD_ENVIRONMENT_VARIABLES
        assert os.environ[name] == value
    for name in bench.THREAD_ENVIRONMENT_VARIABLES:
        if name not in os.environ:
            assert name not in info["thread_environment"]


@needs_native
def test_the_payload_leaks_no_path_user_or_secret():
    payload = bench.run_benchmark(cases=["plan_shuffled_reference"], **SMOKE)
    blob = json.dumps(payload)
    assert str(REPO_ROOT) not in blob
    assert str(Path.home()) not in blob
    assert os.getcwd() not in blob
    user = getpass.getuser()
    if user and len(user) >= 4:
        assert user not in blob, "the payload names the current user"
    for banned in ("PASSWORD", "SECRET", "TOKEN", "API_KEY", "PYTHONPATH",
                   "HOME=", "USERPROFILE"):
        assert banned not in blob.upper() or banned == "TOKEN", banned


@needs_native
def test_a_full_smoke_run_returns_live_storage_exactly_to_baseline(
        live_storages):
    baseline = settled(live_storages)
    bench.run_benchmark(**SMOKE)
    assert settled(live_storages) == baseline


@needs_native
def test_no_live_object_is_returned_from_a_completed_run():
    """The payload is plain Python records. A dataset, loader, sampler,
    iterator, or tensor surviving in it would be a leak in the shape of a
    convenience."""
    payload = bench.run_benchmark(cases=["loader_next_batch"], **SMOKE)
    _require_plain(payload)


def test_the_harness_never_arms_fault_injection_or_relies_on_collection():
    for banned in ("_arm_alloc_failure", "tf_test_arm_alloc_failure",
                   "fault_injection", "gc.collect", "import gc",
                   "weakref", "__del__"):
        assert banned not in BENCHMARK_CODE, banned


# ===========================================================================
# 13. Scope — J8 shipped measurement and nothing else
# ===========================================================================


def test_the_benchmark_inventory_moved_from_eight_to_nine():
    benchmarks = sorted(path.name
                        for path in (REPO_ROOT / "benchmarks").glob("*.py")
                        if path.name != "__init__.py")
    assert len(benchmarks) == 9, benchmarks
    assert benchmarks == sorted([
        "benchmark_native_autograd.py",
        "benchmark_native_classification.py",
        "benchmark_native_cnn.py",
        "benchmark_native_cpu_performance.py",
        "benchmark_native_data_pipeline.py",
        "benchmark_native_dropout.py",
        "benchmark_native_dtype.py",
        "benchmark_native_normalization.py",
        "cpp_backend.py",
    ])
    # Examples did not move: J8 shipped no example.
    examples = [path.name for path in (REPO_ROOT / "examples").glob("*.py")
                if path.name != "__init__.py"]
    assert len(examples) == 16, sorted(examples)


def test_j8_moved_no_capability_registry_or_version():
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    info = cpp.backend_info()
    assert info["dtype"] == "float64"
    assert info["device"] == "cpu"
    assert info["stable_framework_integration"] is False
    from tensorforge.experimental import (
        native_checkpoint, native_data_loader, native_optimizer_state,
        native_sampler,
    )
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert native_optimizer_state.FORMAT_VERSION == 1
    assert native_data_loader._FORMAT == "tensorforge.native_data_loader"
    assert native_data_loader._FORMAT_VERSION == 1
    assert native_sampler._FORMAT == "tensorforge.native_sampler"
    assert native_sampler._FORMAT_VERSION == 1


def test_j8_touched_no_cpp_cmake_abi_or_ci_surface():
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                source.read_text(encoding="utf-8"), re.S))
    assert len(names) == 55, sorted(names)
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    # Phase K, milestone K1 took the native CTest inventory from 24 to 25 (cpp/tests/test_dtype_int64_storage.cpp), which is the first movement since Phase I. The number is updated rather than the assertion relaxed: this test still pins an exact inventory, and still fails on an unrecorded addition.
    assert len(re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)) == 26
    for relative in ("cpp/CMakeLists.txt", "cpp/build.py", "pyproject.toml",
                     ".github/workflows/tests.yml"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "data_pipeline" not in text, relative
        assert "benchmark_native_data_pipeline" not in text, relative
    for path in sorted((REPO_ROOT / "cpp").rglob("*.cpp")):
        assert "data_pipeline" not in path.read_text(encoding="utf-8"), (
            path.name)


def test_no_ci_job_runs_or_gates_on_this_benchmark():
    workflow = (REPO_ROOT / ".github" / "workflows"
                / "tests.yml").read_text(encoding="utf-8")
    assert "benchmark_native_data_pipeline" not in workflow
    for banned in ("--profile", "median_ns", "iqr_ns", "ratio_to_reference"):
        assert banned not in workflow, banned


def test_the_benchmark_asserts_no_duration_and_declares_no_threshold():
    """Structural, over the harness's own source: a benchmark that grew a
    threshold would become a CI job that fails on a number."""
    for banned in ("assert median", "assert samples", "assert elapsed",
                   "THRESHOLD", "BUDGET", "MAX_DURATION", "MIN_SPEEDUP",
                   "max_duration", "min_speedup", "fail_if", "regression",
                   "expected_ns", "limit_ns", "target_ns"):
        assert banned not in BENCHMARK_CODE, banned
    # The only ``assert`` statements allowed here are none at all: the gates
    # raise AssertionError through ``require`` so a -O run cannot skip them.
    tree = ast.parse(BENCHMARK_SOURCE)
    assert [node for node in ast.walk(tree)
            if isinstance(node, ast.Assert)] == []


def test_the_threshold_scanner_can_actually_fail():
    planted = "assert median_ns < 1000  # THRESHOLD\n"
    assert "assert median" in planted and "THRESHOLD" in planted
    assert [node for node in ast.walk(ast.parse("assert 1 < 2"))
            if isinstance(node, ast.Assert)]


def test_j8_shipped_no_optimization_and_no_production_change():
    """The harness measures the shipped pipeline; it does not reach into
    it. No production module imports it, and it patches nothing."""
    package = REPO_ROOT / "src" / "tensorforge"
    for path in package.rglob("*.py"):
        modules = _imported_modules(path.read_text(encoding="utf-8"))
        assert "benchmarks" not in modules, path.name
        assert "benchmark_native_data_pipeline" not in modules, path.name
    # The harness reaches for no private construction seam and rewrites no
    # attribute of the code it measures.
    for banned in ("monkeypatch", "setattr(", "_uninitialized",
                   "_typed_from_array", "_typed_zeros", "_from_core",
                   "_trusted_dtype", "_deliver_batch", "_claim_batch",
                   "_native_permutation", "_validate_state",
                   "_assign_state", "_begin_iteration"):
        assert banned not in BENCHMARK_CODE, banned


def test_the_harness_reads_only_one_private_attribute_and_only_in_tests():
    """The sampler's permutation cache is observable nowhere public. The
    *benchmark* never reads it — it arranges cold and warm state through
    ordinary construction — and only these tests inspect it structurally."""
    assert "_cache_key" not in BENCHMARK_SOURCE
    assert "_cache_order" not in BENCHMARK_SOURCE
    own_source = Path(__file__).read_text(encoding="utf-8")
    assert "_cache_key" in own_source


def test_the_later_phase_j_milestone_lives_in_its_own_module():
    """J9 owns the closure module and the sanitizer matrix; J8 shipped
    neither, and that split survives J9 landing.

    Through J8 this read "the closure module is absent", which expired the
    moment J9 shipped. What replaces it is the durable half: the closure
    guardrails are a **separate** module, and J8's benchmark contract
    module still owns no phase-wide claim."""
    assert (REPO_ROOT / "tests" / "test_native_phase_j_closure.py").exists()
    assert "test_native_phase_j_closure" not in BENCHMARK_SOURCE
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for absent in ("native_data_benchmark.py", "native_benchmark.py",
                   "native_data_workers.py", "native_data_prefetch.py"):
        assert not (package / absent).exists(), absent


def test_no_worker_thread_queue_or_async_surface_was_added():
    """§16: concurrency stays a documented boundary. No test here starts a
    thread and the harness contains none."""
    for banned in ("threading", "Thread(", "multiprocessing", "concurrent",
                   "asyncio", "async def", "await ", "queue.Queue",
                   "ProcessPool", "ThreadPool", "prefetch", "pin_memory",
                   "num_workers", "collate", "Lock("):
        assert banned not in BENCHMARK_CODE, banned
    own_code = code_only(Path(__file__).read_text(encoding="utf-8"))
    for banned in ("threading", "Thread(", "asyncio", "multiprocessing",
                   "concurrent"):
        assert banned not in own_code, banned


def test_no_external_dependency_is_imported():
    modules = _imported_modules(BENCHMARK_SOURCE)
    assert modules == {"argparse", "json", "math", "os", "platform",
                       "statistics", "sys", "time", "pathlib", "numpy",
                       "tensorforge"}, sorted(modules)
    for banned in ("torch", "tensorflow", "jax", "sklearn", "pandas",
                   "matplotlib", "scipy", "pytest"):
        assert banned not in modules, banned
