"""Contract tests for the native integer characterization harness (K8).

These prove the harness's **structure and discipline**, never a speed.
There is deliberately no assertion anywhere that a case is fast, that a
median falls in a range, that one width beats another, that a ratio meets
a bound, or that the cases rank in any order — a benchmark that asserted a
duration would become a CI job that fails on a number, which §9 of
``CLAUDE.md`` forbids outright and which design §31 and the Phase-K exit
gate name as the thing that must not exist anywhere in the repository.

What is asserted instead:

- the correctness gate runs **before** the timer, structurally and
  behaviourally, and every kind of wrong result aborts before any timing
  exists;
- the four questions stay four workload families, and no composed case
  substitutes for any of them;
- ``argmax`` is gated against an independent transcription of design
  §17.5 — not against ``numpy.argmax``, whose tie and NaN rules differ —
  and ``index_select`` against a per-position slice concatenation written
  without ``numpy.take``;
- ``float64``, ``float32``, and ``int64`` are characterized **separately**,
  and no ratio between any two of them appears in the payload, the report,
  or the source; ``int64`` is presented as an index/result dtype and never
  as a compute one;
- every case is ``native_only`` and publishes **no ratio at all**, and no
  code path can assign one;
- setup and every ``close()`` happen outside the timer, the timed region
  holds exactly one operation call, a non-contiguous operand's internal
  Policy-B work stays **inside** it, and every measured sample is retained;
- no result file of any kind is written, and no CLI option could ask for
  one;
- the case inventory, workloads, gates, seeds, layouts, and configurations
  are exact and deterministic;
- importing the module runs nothing and allocates no native storage;
- K8 shipped **zero executable production code**, moved the benchmark
  inventory 9 -> 10 and nothing else, and started no part of K9.

**Every parser and every scanner here has a negative control**, driven
against text, payloads, or planted sources it must reject, so "nothing
found" is evidence rather than a dead regex.

Selector: python -m pytest -q tests/test_native_integer_benchmark.py
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

from benchmarks import benchmark_native_integer as bench       # noqa: E402

from tensorforge.backends import cpp                           # noqa: E402
from tensorforge.experimental import NativeTensor              # noqa: E402

BENCHMARK_FILE = REPO_ROOT / "benchmarks" / "benchmark_native_integer.py"
BENCHMARK_SOURCE = BENCHMARK_FILE.read_text(encoding="utf-8")

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

# ---------------------------------------------------------------------------
# The exact, ordered inventories K8 shipped. Written here independently of
# the harness, so a case appearing, disappearing, or moving family fails an
# equality rather than being absorbed silently.
# ---------------------------------------------------------------------------
EXPECTED_WORKLOADS = (
    "integer_construction",
    "host_materialization",
    "argmax",
    "index_select",
)

EXPECTED_CASES = (
    "int64_construct_small_contiguous",
    "int64_construct_large_contiguous",
    "int64_construct_noncontiguous",
    "int64_construct_matrix",
    "int64_to_numpy_small_contiguous",
    "int64_to_numpy_large_contiguous",
    "int64_to_numpy_noncontiguous",
    "argmax_axis_contiguous",
    "argmax_axis_noncontiguous",
    "argmax_full",
    "argmax_axis_long",
    "index_select_contiguous",
    "index_select_noncontiguous_source",
    "index_select_offset_index",
    "index_select_duplicates",
    "index_select_large_selection",
)

# The four questions K8 undertook to keep apart, named so a family quietly
# collapsing into another shows up here rather than as a shorter table.
REQUIRED_COVERAGE = {
    "integer_construction": {
        "int64_construct_small_contiguous",
        "int64_construct_large_contiguous",
        "int64_construct_noncontiguous",
        "int64_construct_matrix",
    },
    "host_materialization": {
        "int64_to_numpy_small_contiguous",
        "int64_to_numpy_large_contiguous",
        "int64_to_numpy_noncontiguous",
    },
    "argmax": {
        "argmax_axis_contiguous", "argmax_axis_noncontiguous",
        "argmax_full", "argmax_axis_long",
    },
    "index_select": {
        "index_select_contiguous", "index_select_noncontiguous_source",
        "index_select_offset_index", "index_select_duplicates",
        "index_select_large_selection",
    },
}

ROOT_KEYS = {
    "benchmark", "benchmark_version", "schema_version", "mode",
    "selected_cases", "selected_workloads", "dtypes", "environment",
    "methodology", "cases", "disclaimer",
}

CASE_KEYS = {
    "case", "workload", "operation", "dtype", "dtype_role", "configuration",
    "config", "shape", "summary", "layout", "index_layout", "index_pattern",
    "axis", "keepdims", "seed", "reference_type", "native_only",
    "reference_detail", "correctness", "timed_layer", "setup", "cleanup",
    "warmup", "statistics", "reference", "ratio_to_reference",
    "ratio_meaning", "notes",
}

STATISTICS_KEYS = {
    "sample_count", "samples_ns", "median_ns", "min_ns", "max_ns",
    "p25_ns", "p75_ns", "iqr_ns", "mean_ns", "relative_iqr",
    "headline_statistic", "spread_statistic", "units",
}

# The exact key set each gate publishes. A gate that started reporting less
# would make its own record unauditable.
GATE_KEYS = {
    "int64_construction_exact": {
        "passed", "gate", "elements", "shape", "contiguous_host_input",
        "int64_boundaries_checked", "independent_of_host_memory",
        "owning_contiguous_graph_free"},
    "int64_materialization_exact": {
        "passed", "gate", "elements", "shape", "contiguous_source",
        "int64_boundaries_checked", "independent_host_memory",
        "source_unchanged"},
    "argmax_oracle_exact": {
        "passed", "gate", "elements", "shape", "axis", "keepdims",
        "reduced_run_length", "output_elements", "contiguous_source",
        "reference_rows_checked", "checked_against_reference_vector",
        "owning_contiguous_graph_free"},
    "index_select_oracle_bits": {
        "passed", "gate", "elements", "shape", "axis", "index_count",
        "output_elements", "duplicate_indices", "contiguous_source",
        "contiguous_index", "compared_as_raw_bits",
        "owning_contiguous_graph_free"},
}

# The exact expression the timed region of each family contains. Extracted
# from the AST below, so "the timer holds only the operation" is a
# mechanical fact rather than a claim in a docstring.
EXPECTED_TIMED_CALLS = {
    "build_construction": "NativeTensor.from_int64_array(host)",
    "build_materialization": "tensor.to_numpy()",
    "build_argmax": "source.argmax(axis=axis, keepdims=keepdims)",
    "build_index_select": "source.index_select(axis, index_tensor)",
}

# The inventories K8 must leave exactly where K7 left them, and the one it
# moves. Written as literals here because this module's whole job at the
# scope level is to notice a movement.
K7_EXPORT_COUNT = 56
K7_CTEST_COUNT = 27
K7_EXAMPLE_COUNT = 17
K7_EXPERIMENTAL_EXPORTS = 25
K7_CHECKED_KERNELS = 38
K7_BENCHMARK_COUNT = 9
K8_BENCHMARK = "benchmark_native_integer.py"
K8_BENCHMARK_COUNT = K7_BENCHMARK_COUNT + 1                     # 10

# K9's own module, which must not exist while K8 is the newest milestone.
K9_ARTIFACT = "tests/test_native_phase_k_closure.py"

SMOKE = {"smoke": True}


def code_only(source):
    """The harness's **code**, with every docstring, comment, and string
    literal removed.

    Every substring ban below runs over this rather than over the raw
    file, on the recorded lesson every scanner in this phase inherits: a
    naive scan fails on exactly the prose that documents the prohibition.
    The harness says in words that it subtracts no timer overhead,
    discards no outlier, publishes no ratio, and uses no tolerance — and a
    checker that tripped on those sentences would be noise rather than a
    guardrail, and would push the next author to delete the explanation
    instead of the behaviour.
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


def first_dtype(name):
    """The first dtype a case declares — its own width, never another
    family's."""
    return bench.CASES[name]["dtypes"][0]


# ===========================================================================
# Shared instrumentation
# ===========================================================================


@pytest.fixture
def live_storages():
    """The ids of every open ``NativeStorage`` — the project's
    deterministic instrumentation for native-allocation lifetime, used
    unchanged since Phase C. **There is no public counter and K8 adds
    none.**

    Installed with an explicit save/restore rather than through
    ``monkeypatch``, on the recorded reason: a test that calls
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
    tensor = NativeTensor.from_int64_array(np.arange(4, dtype=np.int64))
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


def test_the_timer_spy_can_actually_fail():
    """Negative control for the spy every gate-ordering test relies on: it
    must record a call when one happens, and must be silent when none
    does."""
    silent = SpyTimer()
    assert silent.calls == []
    with pytest.raises(RuntimeError, match="timer reached"):
        silent(None, None, None, 0, 1)
    assert silent.calls == [(0, 1)]
    delegating = SpyTimer(bench.measure)
    samples = delegating(lambda: None, lambda state: None,
                         lambda state, result: None, 0, 2)
    assert delegating.calls == [(0, 2)] and len(samples) == 2


# ===========================================================================
# 1. Import safety
# ===========================================================================


def test_importing_the_module_runs_nothing(capsys, live_storages):
    """Importing a benchmark must never execute one: no output, no native
    allocation, no tensor, no CLI parsing, and no file."""
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
    or builds a tensor."""
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
                   "NativeTensor.from_int64_array", "NativeTensor.from_array",
                   "host_int64", "host_floating", "np.random.default_rng"):
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
    assert bench.BENCHMARK_NAME == "tensorforge.native_integer"
    assert bench.BENCHMARK_VERSION == "1.0"
    assert bench.SCHEMA_VERSION == 1
    assert isinstance(bench.SCHEMA_VERSION, int)
    assert not isinstance(bench.SCHEMA_VERSION, bool)


def test_the_identity_is_not_a_package_export():
    """A benchmark is a measurement tool, never a capability. Nothing here
    may reach the public surface of either line, and no production module
    may name it."""
    import tensorforge
    import tensorforge.experimental as experimental

    for name in ("BENCHMARK_NAME", "BENCHMARK_VERSION", "SCHEMA_VERSION",
                 "CASES", "WORKLOADS", "run_benchmark", "measure",
                 "benchmark_native_integer"):
        assert name not in tensorforge.__all__, name
        assert name not in experimental.__all__, name
        assert not hasattr(experimental, name), name
    assert len(experimental.__all__) == K7_EXPERIMENTAL_EXPORTS
    # Scanned over **code**, not raw text: the package's status docstring
    # legitimately names the harness it is recording, and a substring ban
    # that tripped on that sentence would force the status surface to stop
    # saying what shipped. What must not exist is an *executable*
    # reference, and ``test_only_the_package_status_docstring_may_name_k8``
    # pins the docstring as the single place the name may appear at all.
    for path in (REPO_ROOT / "src").rglob("*.py"):
        code = code_only(path.read_text(encoding="utf-8"))
        assert "benchmark_native_integer" not in code, path.name
        assert "BENCHMARK_NAME" not in code, path.name
    # ...and it is registered in no runtime inventory either.
    info = cpp.backend_info()
    assert "benchmark" not in json.dumps(info).lower()


@needs_native
def test_the_payload_root_and_case_schema_are_exact():
    payload = bench.run_benchmark(cases=["argmax_full"], **SMOKE)
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
def test_every_gate_publishes_its_exact_key_set():
    """One row per family, so all four gate schemas are checked rather than
    whichever one a single case happens to use."""
    payload = bench.run_benchmark(**SMOKE)
    seen = set()
    for record in payload["cases"]:
        gate = record["correctness"]["gate"]
        seen.add(gate)
        assert set(record["correctness"]) == GATE_KEYS[gate], record["case"]
        assert record["correctness"]["passed"] is True, record["case"]
    assert seen == set(bench.GATES) == set(GATE_KEYS)


@needs_native
def test_the_payload_round_trips_through_json_as_plain_python():
    payload = bench.run_benchmark(cases=["int64_construct_matrix"], **SMOKE)
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
        _require_plain({"value": np.int64(1)})
    with pytest.raises(AssertionError):
        _require_plain({"value": (1, 2)})
    with pytest.raises(AssertionError):
        _require_plain({1: "int key"})
    _require_plain({"ok": [1, 2.0, "three", True, None]})


# ===========================================================================
# 3. Case and workload inventory
# ===========================================================================


def test_the_case_registry_is_exact_and_ordered():
    assert tuple(bench.CASES) == EXPECTED_CASES
    assert len(bench.CASES) == 16


def test_the_workload_registry_is_exact_and_every_family_is_populated():
    assert bench.WORKLOADS == EXPECTED_WORKLOADS
    populated = {spec["workload"] for spec in bench.CASES.values()}
    assert populated == set(EXPECTED_WORKLOADS), (
        f"declared but empty: {set(EXPECTED_WORKLOADS) - populated}")
    for workload, names in REQUIRED_COVERAGE.items():
        present = {name for name, spec in bench.CASES.items()
                   if spec["workload"] == workload}
        assert present == names, (workload, present ^ names)


def test_the_four_required_questions_are_four_separate_families():
    """K8's four questions must stay four measurements, and there must be
    no fifth family standing in for any of them."""
    assert len(bench.WORKLOADS) == 4
    for required in EXPECTED_WORKLOADS:
        assert required in bench.WORKLOADS
        assert bench.cases_for_workloads([required]), required
    # No composition ships, and no case claims to be one.
    for name, spec in bench.CASES.items():
        blob = f"{name} {spec['operation']} {spec['notes']}".lower()
        assert "composition" not in blob or "no composed" in blob, name
    assert "composition" not in {spec["workload"]
                                 for spec in bench.CASES.values()}


def test_every_case_declares_a_complete_auditable_specification():
    required_fields = (
        "workload", "label", "operation", "build", "gate", "reference_type",
        "native_only", "reference_detail", "ratio_meaning",
        "correctness_reference", "dtypes", "dtype_role", "geometry",
        "layout", "index_layout", "index_pattern", "index_offset", "axis",
        "keepdims", "seed", "index_seed", "configurations", "setup",
        "timed", "cleanup", "notes",
    )
    for name, spec in bench.CASES.items():
        for field in required_fields:
            assert field in spec, (name, field)
        assert spec["label"] == name, name
        assert callable(spec["build"]), name
        assert spec["workload"] in bench.WORKLOADS, name
        assert spec["gate"] in bench.GATES, name
        assert spec["reference_type"] in bench.REFERENCE_TYPES, name
        assert spec["native_only"] is True, name
        assert spec["ratio_meaning"] is None, name
        assert spec["dtype_role"] in bench.DTYPE_ROLES, name
        assert spec["geometry"] in bench.GEOMETRIES, name
        assert spec["layout"] in bench.LAYOUTS, name
        assert spec["index_layout"] in (None,) + bench.LAYOUTS, name
        assert spec["index_pattern"] in (None,) + bench.INDEX_PATTERNS, name
        assert type(spec["index_offset"]) is int, name
        assert spec["index_offset"] >= 0, name
        assert isinstance(spec["seed"], int) and not isinstance(
            spec["seed"], bool), name
        for text_field in ("operation", "reference_detail",
                           "correctness_reference", "setup", "timed",
                           "cleanup", "notes"):
            assert spec[text_field].strip(), (name, text_field)
        assert set(spec["configurations"]) == set(bench.CONFIGURATIONS), name


def test_every_case_carries_a_unique_deterministic_seed():
    seeds = [spec["seed"] for spec in bench.CASES.values()]
    assert len(set(seeds)) == len(seeds), "a data seed is shared"
    index_seeds = [spec["index_seed"] for spec in bench.CASES.values()
                   if spec["index_seed"] is not None]
    assert len(set(index_seeds)) == len(index_seeds), "an index seed is shared"


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
        assert full != smoke, name


def test_both_layouts_and_several_geometries_are_covered_per_family():
    """A family that only ever measured one layout would answer half its
    question."""
    layouts = {}
    for spec in bench.CASES.values():
        layouts.setdefault(spec["workload"], set()).add(spec["layout"])
    assert bench.LAYOUT_STRIDED_HOST in layouts["integer_construction"]
    assert bench.LAYOUT_TRANSPOSED in layouts["host_materialization"]
    assert bench.LAYOUT_TRANSPOSED in layouts["argmax"]
    assert bench.LAYOUT_TRANSPOSED in layouts["index_select"]
    for workload, present in layouts.items():
        assert bench.LAYOUT_CONTIGUOUS in present, workload
    geometries = {spec["geometry"] for spec in bench.CASES.values()}
    assert geometries == set(bench.GEOMETRIES)
    # The index operand has its own layout axis, and it is exercised too.
    index_layouts = {spec["index_layout"] for spec in bench.CASES.values()
                     if spec["workload"] == "index_select"}
    assert index_layouts == {bench.LAYOUT_CONTIGUOUS, bench.LAYOUT_OFFSET}


def test_the_argmax_family_covers_axis_full_and_keepdims():
    axes = {bench.CASES[name]["axis"] for name in REQUIRED_COVERAGE["argmax"]}
    assert None in axes and 1 in axes, axes
    keepdims = {bench.CASES[name]["keepdims"]
                for name in REQUIRED_COVERAGE["argmax"]}
    assert keepdims == {True, False}, keepdims
    # A genuinely longer reduction axis, not merely a bigger tensor.
    long_case = bench.CASES["argmax_axis_long"]["configurations"]["full"]
    plain_case = bench.CASES["argmax_axis_contiguous"][
        "configurations"]["full"]
    assert long_case["columns"] > plain_case["columns"]


def test_the_index_select_family_covers_duplicates_order_and_both_axes():
    patterns = {bench.CASES[name]["index_pattern"]
                for name in REQUIRED_COVERAGE["index_select"]}
    assert patterns == set(bench.INDEX_PATTERNS), patterns
    axes = {bench.CASES[name]["axis"]
            for name in REQUIRED_COVERAGE["index_select"]}
    assert axes == {0, 1}, axes
    # A selection larger than the axis it selects from is legal and is
    # measured; it can only be built with duplicates.
    large = bench.CASES["index_select_large_selection"][
        "configurations"]["full"]
    assert large["indices"] > large["rows"]


@needs_native
def test_every_declared_case_is_executable_and_no_other_is():
    payload = bench.run_benchmark(**SMOKE)
    produced = []
    for record in payload["cases"]:
        if record["case"] not in produced:
            produced.append(record["case"])
    assert tuple(produced) == EXPECTED_CASES
    for bad in ("no_such_case", "", "argmax"):
        with pytest.raises(ValueError):
            bench.run_benchmark(cases=[bad], **SMOKE)
    with pytest.raises(ValueError):
        bench.run_benchmark(workloads=["no_such_workload"], **SMOKE)
    with pytest.raises(ValueError):
        bench.run_benchmark(cases=[], **SMOKE)
    with pytest.raises(ValueError):
        bench.run_benchmark(dtypes=["float16"], **SMOKE)


@needs_native
def test_a_selection_that_matches_nothing_is_an_error_not_an_empty_run():
    """An empty run is a mistake, never a silent success: a payload with no
    rows would report "nothing failed" for work that never happened."""
    with pytest.raises(ValueError, match="no case matches"):
        bench.run_benchmark(workloads=["argmax"], dtypes=["int64"], **SMOKE)
    with pytest.raises(ValueError, match="no case matches"):
        bench.run_benchmark(cases=["int64_construct_matrix"],
                            dtypes=["float64"], **SMOKE)


# ===========================================================================
# 4. Reference honesty — every case is native-only and publishes no ratio
# ===========================================================================


def test_the_reference_registry_is_one_honest_label():
    assert bench.REFERENCE_TYPES == ("native_only",)
    assert bench.NATIVE_ONLY == "native_only"


def test_every_case_is_native_only_and_says_why():
    for name, spec in bench.CASES.items():
        assert spec["reference_type"] == bench.NATIVE_ONLY, name
        assert spec["native_only"] is True, name
        assert spec["ratio_meaning"] is None, name
        detail = spec["reference_detail"].lower()
        assert detail.startswith("none"), name
        assert ("allocat" in detail or "transfer" in detail
                or "cross" in detail), name


def test_the_argmax_family_publishes_no_numpy_argmax_ratio():
    """Design §31 names this by name as the live fairness risk: the native
    call allocates an int64 output tensor and ``numpy.argmax`` over an
    existing host array does not."""
    for name in REQUIRED_COVERAGE["argmax"]:
        spec = bench.CASES[name]
        assert spec["native_only"] is True, name
        assert spec["ratio_meaning"] is None, name
    # The reason is recorded where a reader will find it, on the case that
    # states it in full.
    detail = bench.CASES["argmax_axis_contiguous"]["reference_detail"].lower()
    assert "numpy.argmax" in detail and "allocat" in detail
    # ...and NumPy's argmax is never called at all, so there is nothing to
    # divide by even by accident. Read off the AST: every ``argmax`` call
    # the harness makes is either the operation under measurement or this
    # harness's own oracle, and no third spelling exists.
    assert "np.argmax" not in BENCHMARK_CODE
    assert "numpy.argmax" not in BENCHMARK_CODE
    callers = {ast.unparse(node.func)
               for node in ast.walk(ast.parse(BENCHMARK_SOURCE))
               if isinstance(node, ast.Call)
               and ast.unparse(node.func).split(".")[-1].startswith("argmax")}
    assert callers == {"source.argmax", "argmax_run", "argmax_oracle"}, (
        callers)


def test_no_case_uses_numpy_take_as_a_reference_or_an_oracle():
    assert "np.take" not in BENCHMARK_CODE
    assert "numpy.take" not in BENCHMARK_CODE
    assert ".take(" not in BENCHMARK_CODE


@needs_native
def test_no_record_publishes_a_ratio_or_a_reference_timing():
    payload = bench.run_benchmark(**SMOKE)
    for record in payload["cases"]:
        assert record["native_only"] is True, record["case"]
        assert record["reference_type"] == "native_only", record["case"]
        assert record["ratio_to_reference"] is None, record["case"]
        assert record["ratio_meaning"] is None, record["case"]
        assert record["reference"] is None, record["case"]


def test_no_code_path_can_assign_a_ratio():
    """Structural, over the AST: the three ratio keys are literal ``None``
    in the one place a record is built, so no arithmetic could ever produce
    one."""
    tree = ast.parse(BENCHMARK_SOURCE)
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "run_case"]
    assert len(functions) == 1
    literals = {}
    for node in ast.walk(functions[0]):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in (
                        "ratio_to_reference", "ratio_meaning", "reference"):
                    literals[key.value] = ast.unparse(value)
    assert literals == {"ratio_to_reference": "None", "ratio_meaning": "None",
                        "reference": "None"}, literals
    # ...and the harness carries no reference-timing machinery at all.
    for banned in ("reference_run", "reference_prepare", "reference_cleanup",
                   "has_reference", "ratio =", "ratio="):
        assert banned not in BENCHMARK_CODE, banned


def test_the_ratio_literal_scanner_can_actually_fail():
    """Negative control: a computed ratio in the same position must be
    visible to the scan above."""
    planted = ast.parse(
        "def run_case():\n"
        "    return {'ratio_to_reference': a / b, 'reference': stats}\n")
    seen = {}
    for node in ast.walk(planted):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                seen[key.value] = ast.unparse(value)
    assert seen["ratio_to_reference"] == "a / b"
    assert seen["reference"] == "stats"


# ===========================================================================
# 5. Correctness before timing
# ===========================================================================


@needs_native
@pytest.mark.parametrize("name", EXPECTED_CASES)
def test_every_case_gates_before_the_timer(name, monkeypatch):
    """If ``measure`` is reached at all, ``check()`` has already returned.
    Proved by making the timer explode: the exception must come from the
    timer, which means the gate ran and passed first."""
    spy = SpyTimer()
    monkeypatch.setattr(bench, "measure", spy)
    with pytest.raises(RuntimeError, match="timer reached"):
        bench.run_case(name, first_dtype(name), 0, 1, "smoke")
    assert spy.calls == [(0, 1)]


@needs_native
def test_the_gate_ordering_is_structural_too():
    """``run_case`` calls ``check()`` before ``measure`` and assigns the
    timing only afterwards — read off the AST rather than inferred."""
    tree = ast.parse(BENCHMARK_SOURCE)
    body = [node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_case"][0]
    order = [ast.unparse(node.func) for node in ast.walk(body)
             if isinstance(node, ast.Call)
             and ast.unparse(node.func) in ("case.check", "measure")]
    assert order[:2] == ["case.check", "measure"], order


@needs_native
@pytest.mark.parametrize("name, dtype", [
    ("int64_construct_small_contiguous", "int64"),
    ("int64_construct_noncontiguous", "int64"),
    ("int64_to_numpy_noncontiguous", "int64"),
    ("argmax_axis_contiguous", "float64"),
    ("argmax_axis_noncontiguous", "float32"),
    ("index_select_contiguous", "float64"),
    ("index_select_duplicates", "float32"),
])
def test_a_perturbed_result_fails_before_any_timing(name, dtype, monkeypatch,
                                                    live_storages):
    """The negative control that makes every gate load-bearing: perturb
    what the runtime returns and the case must fail in its gate, before
    ``measure`` is entered, with live storage back at baseline.

    One injection covers all four families because all four gates end in
    an exact comparison of materialized values — which is exactly the
    property that makes them gates rather than shape checks.
    """
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    original = cpp.NativeTensorCore.to_numpy

    def perturbed(self):
        return original(self) + 1

    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", perturbed)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError):
        bench.run_case(name, dtype, 0, 1, "smoke")
    assert spy.calls == [], "timing was reached despite a failed gate"
    assert settled(live_storages) == baseline


@needs_native
def test_a_wrong_shape_fails_before_any_timing(monkeypatch, live_storages):
    """A truncated construction is a different answer, and the gate must
    say so before any timing exists."""
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    original = NativeTensor.from_int64_array

    def truncated(values, *, requires_grad=False):
        return original(np.ascontiguousarray(np.asarray(values).reshape(-1)),
                        requires_grad=requires_grad)

    monkeypatch.setattr(NativeTensor, "from_int64_array", truncated)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError, match="shape"):
        bench.run_case("int64_construct_matrix", "int64", 0, 1, "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


@needs_native
def test_a_deduplicated_selection_fails_before_any_timing(monkeypatch,
                                                          live_storages):
    """Order and duplicate preservation are load-bearing: a selection that
    sorted and deduplicated its indices must be rejected, even though it
    would satisfy a naive set comparison."""
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    original = NativeTensor.index_select

    def sorted_unique(self, axis, indices):
        values = np.unique(indices.to_numpy())
        replacement = NativeTensor.from_int64_array(
            np.ascontiguousarray(values, dtype=np.int64))
        try:
            return original(self, axis, replacement)
        finally:
            replacement.close()

    monkeypatch.setattr(NativeTensor, "index_select", sorted_unique)
    baseline = settled(live_storages)
    with pytest.raises(AssertionError):
        bench.run_case("index_select_contiguous", "float64", 0, 1, "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


@needs_native
def test_a_gradient_tracking_result_fails_before_any_timing(monkeypatch,
                                                            live_storages):
    """``argmax`` and ``index_select`` results are plain leaves. A result
    that claimed a gradient must be rejected by the gate."""
    spy = SpyTimer(bench.measure)
    monkeypatch.setattr(bench, "measure", spy)
    monkeypatch.setattr(NativeTensor, "requires_grad",
                        property(lambda self: True))
    baseline = settled(live_storages)
    with pytest.raises(AssertionError):
        bench.run_case("argmax_full", "float64", 0, 1, "smoke")
    assert spy.calls == []
    assert settled(live_storages) == baseline


@needs_native
def test_a_wrong_dtype_result_is_caught_by_the_gate():
    """A gate that silently accepted another width would let a case run at
    float64 while reporting float32 — the one mistake that would make the
    whole harness dishonest. The integer boundary is checked the same way,
    including byte order."""
    with pytest.raises(AssertionError, match="expected a float32 array"):
        bench.bits_of(np.ones(4, dtype=np.float64), "float32")
    with pytest.raises(AssertionError, match="expected a float64 array"):
        bench.bits_of(np.ones(4, dtype=np.float32), "float64")
    assert bench.bits_of(np.zeros(2, dtype=np.float32), "float32").dtype == (
        np.uint32)
    with pytest.raises(AssertionError, match="expected int64"):
        bench.exact_int64(np.ones(4, dtype=np.int32), "probe")
    with pytest.raises(AssertionError, match="not native"):
        bench.exact_int64(np.ones(4, dtype=">i8"), "probe")
    assert bench.exact_int64(np.ones(4, dtype=np.int64), "probe").dtype == (
        np.int64)


# ===========================================================================
# 6. What each family actually proves
# ===========================================================================


def test_the_argmax_oracle_is_tensorforges_own_rule():
    """The oracle must reproduce design §17.5's committed answers, and must
    be **structurally** independent of NumPy: ``argmax_run`` is a direct
    transcription of the design's algorithm and touches no array library
    at all.

    It is *not* asserted that the answers differ from ``numpy.argmax``.
    They happen to coincide on these rows, and that is a coincidence the
    design explicitly declines to promise ("K0 makes no compatibility
    claim with NumPy or another framework") — which is precisely why the
    authority here is the transcription and the committed table, not the
    other library. The two tests below prove the table can tell the
    difference between this rule and its plausible alternatives.
    """
    for values, expected, description in bench.ARGMAX_REFERENCE_RUNS:
        host = np.ascontiguousarray(values, dtype=np.float64)
        assert int(bench.argmax_run(list(values))) == expected, description
        assert int(bench.argmax_oracle(host, None, False)) == expected, (
            description)
    # Structural independence: no array library inside the rule itself.
    rule = [node for node in ast.walk(ast.parse(BENCHMARK_SOURCE))
            if isinstance(node, ast.FunctionDef) and node.name == "argmax_run"]
    assert len(rule) == 1
    body = ast.unparse(rule[0])
    for banned in ("np.", "numpy", "max(", "sort", "argsort"):
        assert banned not in body, banned
    assert "math.isnan" in body


def test_the_committed_table_discriminates_the_nan_and_tie_rules():
    """Negative control for the table itself: two plausible *alternative*
    rules must fail it, or "the oracle matches the table" would be an
    empty statement.

    A skip-NaN rule and a last-maximum rule are exactly the two mistakes
    §17.5 was written to exclude, and each must disagree with the
    committed answers on at least one row.
    """
    def skips_nan(run):
        best_index, best = None, None
        for position, value in enumerate(run):
            if value != value:                        # a NaN, skipped
                continue
            if best is None or value > best:
                best_index, best = position, value
        return 0 if best_index is None else best_index

    def last_maximum(run):
        best_index, best = 0, run[0]
        for position in range(1, len(run)):
            value = run[position]
            if best != best:                          # incumbent NaN
                continue
            if value != value or value >= best:       # >= rather than >
                best_index, best = position, value
        return best_index

    skip_failures = sum(
        1 for values, expected, _ in bench.ARGMAX_REFERENCE_RUNS
        if skips_nan(list(values)) != expected)
    tie_failures = sum(
        1 for values, expected, _ in bench.ARGMAX_REFERENCE_RUNS
        if last_maximum(list(values)) != expected)
    assert skip_failures >= 1, "the table does not pin the NaN rule"
    assert tie_failures >= 1, "the table does not pin the tie rule"


def test_the_argmax_oracle_covers_every_committed_row_of_section_17_5():
    """The table is the specification, so its coverage is asserted rather
    than assumed: ties, both signed-zero orders, all -inf, +inf, one NaN,
    several NaNs, a NaN against either infinity, a NaN at index 0, and a
    length-1 run."""
    descriptions = " | ".join(row[2]
                              for row in bench.ARGMAX_REFERENCE_RUNS).lower()
    for required in ("unique maximum", "equal maxima", "signed zeros",
                     "-inf", "+inf", "exactly one nan", "several nans",
                     "index 0", "length 1"):
        assert required in descriptions, required
    assert len(bench.ARGMAX_REFERENCE_RUNS) == 12
    # Every row is exactly representable at float32 too, so the same table
    # is a known answer at both widths.
    for values, _, description in bench.ARGMAX_REFERENCE_RUNS:
        narrow = np.ascontiguousarray(values, dtype=np.float32)
        wide = np.ascontiguousarray(values, dtype=np.float64)
        assert np.array_equal(narrow, wide, equal_nan=True), description


def test_the_argmax_oracle_reduces_per_run_and_follows_reduce_shape():
    values = np.array([[1.0, 5.0, 5.0], [7.0, 2.0, 3.0]], dtype=np.float64)
    assert bench.argmax_oracle(values, 1, False).tolist() == [1, 0]
    assert bench.argmax_oracle(values, 0, False).tolist() == [1, 0, 0]
    assert bench.argmax_oracle(values, 1, True).shape == (2, 1)
    assert bench.argmax_oracle(values, 0, True).shape == (1, 3)
    # axis=None gives the flat row-major index, and keepdims gives ones.
    assert int(bench.argmax_oracle(values, None, False)) == 3
    assert bench.argmax_oracle(values, None, True).shape == (1, 1)
    # A NaN in one run never reaches another.
    mixed = np.array([[float("nan"), 1.0], [3.0, 4.0]], dtype=np.float64)
    assert bench.argmax_oracle(mixed, 1, False).tolist() == [0, 1]
    with pytest.raises(ValueError):
        bench.argmax_run([])


@needs_native
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_native_argmax_matches_the_oracle_at_each_width_separately(dtype):
    """Each width is proved only against itself. No value, index, or
    duration crosses between the two."""
    generator = np.random.default_rng(20260899)
    host = np.ascontiguousarray(generator.uniform(-3.0, 3.0, size=(7, 11)),
                                dtype=bench.NUMPY_DTYPES[dtype])
    source = NativeTensor.from_array(host, dtype=dtype)
    try:
        for axis, keepdims in ((None, False), (None, True), (0, False),
                               (1, False), (1, True), (-1, False)):
            result = source.argmax(axis=axis, keepdims=keepdims)
            try:
                expected = bench.argmax_oracle(host, axis, keepdims)
                produced = bench.exact_int64(result.to_numpy(), "probe")
                assert produced.shape == expected.shape, (axis, keepdims)
                assert np.array_equal(produced, expected), (axis, keepdims)
            finally:
                result.close()
    finally:
        source.close()


def test_the_index_select_oracle_preserves_duplicates_and_order():
    values = np.arange(12.0).reshape(4, 3)
    wanted = [3, 0, 3, 1]
    produced = bench.index_select_oracle(values, 0, wanted)
    assert produced.shape == (4, 3)
    for position, index in enumerate(wanted):
        assert np.array_equal(produced[position], values[index])
    # A sorted, deduplicated answer is a different answer.
    assert not np.array_equal(produced,
                              bench.index_select_oracle(values, 0,
                                                        sorted(set(wanted))))
    # ...and the inner axis works the same way.
    inner = bench.index_select_oracle(values, 1, [2, 2, 0])
    assert inner.shape == (4, 3)
    assert np.array_equal(inner[:, 0], values[:, 2])
    assert np.array_equal(inner[:, 1], values[:, 2])
    assert np.array_equal(inner[:, 2], values[:, 0])
    # A negative axis normalizes exactly as the runtime's does.
    assert np.array_equal(bench.index_select_oracle(values, -1, [1, 0]),
                          bench.index_select_oracle(values, 1, [1, 0]))


@needs_native
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_native_index_select_matches_the_oracle_by_raw_bits(dtype):
    """Floating payloads are compared as raw IEEE-754 bit patterns inside
    one width — ``uint64`` for float64 and ``uint32`` for float32 — and
    the two widths are never compared with one another."""
    generator = np.random.default_rng(20260898)
    host = np.ascontiguousarray(generator.uniform(-1.0, 1.0, size=(9, 5)),
                                dtype=bench.NUMPY_DTYPES[dtype])
    wanted = np.ascontiguousarray([8, 0, 8, 3, 3], dtype=np.int64)
    source = NativeTensor.from_array(host, dtype=dtype)
    indices = NativeTensor.from_int64_array(wanted)
    try:
        result = source.index_select(0, indices)
        try:
            expected = bench.index_select_oracle(host, 0, wanted)
            assert result.dtype == dtype
            assert result.shape == expected.shape
            assert np.array_equal(bench.bits_of(result.to_numpy(), dtype),
                                  bench.bits_of(expected, dtype))
        finally:
            result.close()
    finally:
        indices.close()
        source.close()
    assert bench.BIT_DTYPES == {"float64": np.uint64, "float32": np.uint32}


@needs_native
def test_the_integer_payloads_carry_both_sixty_four_bit_boundaries():
    """The values that make this a genuine integer boundary rather than a
    float64 detour: a float64 round trip would round both of them."""
    values = bench.host_int64((6, 4), 12345)
    flat = values.reshape(-1)
    assert int(flat[0]) == bench.INT64_MIN == -(2 ** 63)
    assert int(flat[-1]) == bench.INT64_MAX == 2 ** 63 - 1
    assert values.dtype == np.int64
    assert int(np.float64(bench.INT64_MAX)) != bench.INT64_MAX, (
        "the boundary would survive a float64 detour, so it proves nothing")
    tensor = NativeTensor.from_int64_array(values)
    try:
        assert np.array_equal(tensor.to_numpy(), values)
        assert tensor.tolist()[0][0] == bench.INT64_MIN
    finally:
        tensor.close()


def test_the_strided_host_builder_produces_a_real_non_contiguous_view():
    logical = bench.host_int64((8,), 999)
    base, view = bench.strided_host_int64(logical)
    assert not view.flags["C_CONTIGUOUS"]
    assert view.dtype == np.int64
    assert np.array_equal(view, logical)
    # Every skipped position holds the filler, so a constructor that read
    # the base rather than the view would be visible.
    assert np.array_equal(base[1::2],
                          np.full((8,), bench.STRIDE_FILLER, dtype=np.int64))


def test_the_index_vector_builder_is_deterministic_and_in_bounds():
    for pattern in bench.INDEX_PATTERNS:
        first = bench.index_values(pattern, 16, 8, 77)
        second = bench.index_values(pattern, 16, 8, 77)
        assert np.array_equal(first, second), pattern
        assert first.dtype == np.int64, pattern
        assert first.shape == (8,), pattern
        assert int(first.min()) >= 0 and int(first.max()) < 16, pattern
    assert len(set(bench.index_values("distinct", 16, 8, 1).tolist())) == 8
    duplicated = bench.index_values("duplicates", 16, 8, 2).tolist()
    assert len(set(duplicated)) == 4, duplicated
    with pytest.raises(ValueError):
        bench.index_values("distinct", 4, 8, 3)
    with pytest.raises(ValueError):
        bench.index_values("no_such_pattern", 4, 2, 3)


@needs_native
@pytest.mark.parametrize("name", ["int64_to_numpy_noncontiguous",
                                  "argmax_axis_noncontiguous",
                                  "index_select_noncontiguous_source"])
def test_a_non_contiguous_case_really_measures_a_non_contiguous_operand(name):
    """Structural precondition: if these views were contiguous, three
    cases would silently be duplicates of their contiguous siblings."""
    spec = bench.CASES[name]
    dtype = first_dtype(name)
    case = spec["build"](dtype, spec["configurations"]["smoke"], spec)
    try:
        metrics = case.check()
        assert metrics["contiguous_source"] is False, name
    finally:
        case.teardown()


@needs_native
def test_the_offset_index_case_really_measures_an_offset_view():
    spec = bench.CASES["index_select_offset_index"]
    assert spec["index_offset"] > 0
    case = spec["build"]("float64", spec["configurations"]["smoke"], spec)
    try:
        metrics = case.check()
        # An offset view is still contiguous; what is different is where in
        # its storage it starts.
        assert metrics["contiguous_index"] is True
    finally:
        case.teardown()


@needs_native
def test_the_duplicate_case_really_contains_duplicates():
    spec = bench.CASES["index_select_duplicates"]
    case = spec["build"]("float64", spec["configurations"]["smoke"], spec)
    try:
        metrics = case.check()
        assert metrics["duplicate_indices"] is True
    finally:
        case.teardown()
    distinct = bench.CASES["index_select_contiguous"]
    case = distinct["build"]("float64", distinct["configurations"]["smoke"],
                             distinct)
    try:
        assert case.check()["duplicate_indices"] is False
    finally:
        case.teardown()


# ===========================================================================
# 7. Timing methodology
# ===========================================================================


def _timed_expression(builder):
    """The single expression the named builder's ``run`` closure returns.

    Read off the AST, so "the timed region contains exactly the operation"
    is a mechanical fact. A ``run`` with two statements, or one that did
    anything other than return a call, fails here.
    """
    tree = ast.parse(BENCHMARK_SOURCE)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == builder):
            continue
        for inner in node.body:
            if isinstance(inner, ast.FunctionDef) and inner.name == "run":
                assert len(inner.body) == 1, (builder, "run is not one line")
                statement = inner.body[0]
                assert isinstance(statement, ast.Return), builder
                assert isinstance(statement.value, ast.Call), builder
                return ast.unparse(statement.value)
    raise AssertionError(f"{builder} has no run closure")


@pytest.mark.parametrize("builder, expression",
                         sorted(EXPECTED_TIMED_CALLS.items()))
def test_the_timed_region_holds_exactly_one_operation(builder, expression):
    assert _timed_expression(builder) == expression


def test_the_timed_expression_extractor_can_actually_fail():
    """Negative control: a ``run`` doing more than the operation must be
    rejected rather than silently accepted."""
    planted = ast.parse(
        "def build_probe():\n"
        "    def run(state):\n"
        "        warm()\n"
        "        return source.argmax()\n")
    run = planted.body[0].body[0]
    assert len(run.body) == 2, "the control is not the shape it claims"
    with pytest.raises(AssertionError):
        assert len(run.body) == 1


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
    measured intervals come back, in order, including a deliberate outlier
    that must not be trimmed."""
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


def test_no_timer_overhead_is_subtracted_and_no_sample_is_trimmed():
    """Structural, over the harness's own **code**: the measured interval
    is a bare difference of two reads, appended raw."""
    assert "elapsed = time.perf_counter_ns() - start" in BENCHMARK_CODE
    assert "samples.append(elapsed)" in BENCHMARK_CODE
    for banned in ("elapsed -", "elapsed /", "elapsed *", "overhead",
                   "calibrat", "trim", "outlier", "winsor", "discard_",
                   "clip("):
        assert banned not in BENCHMARK_CODE, banned


def test_the_overhead_scanner_can_actually_fail():
    planted = code_only("elapsed = raw - overhead_ns\nsamples.append(elapsed)")
    assert "overhead" in planted
    assert "elapsed -" in code_only("x = elapsed - 5")


def test_setup_and_cleanup_are_outside_the_timed_region(monkeypatch):
    """Behavioural, not structural: a clock that only advances during
    ``run`` must produce exactly the run's own intervals, and one that
    advances during ``prepare``/``cleanup`` must contribute nothing."""
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


def test_internal_policy_b_work_is_never_moved_out_of_the_operation():
    """A harness that materialized a non-contiguous operand outside the
    timer would report a cost no caller pays. The three non-contiguous
    cases say the materialization is inside the call, and the code calls
    no copy helper at all."""
    for banned in ("contiguous_copy", "_materialize", "_contiguous",
                   "ascontiguousarray(source", "detach("):
        assert banned not in BENCHMARK_CODE, banned
    for name in ("argmax_axis_noncontiguous",
                 "index_select_noncontiguous_source"):
        timed = bench.CASES[name]["timed"].lower()
        assert "policy-b" in timed and "including" in timed, name


def test_every_case_declares_its_timed_boundary_setup_and_cleanup():
    for name, spec in bench.CASES.items():
        assert "timer" in spec["setup"], name
        assert spec["timed"].startswith("exactly one"), name
        assert spec["cleanup"].strip(), name
        if spec["workload"] in ("integer_construction", "argmax",
                                "index_select"):
            assert "clos" in spec["cleanup"], name
            assert "outside the timer" in spec["cleanup"], name


@needs_native
def test_the_effective_counts_are_reported_on_every_record():
    payload = bench.run_benchmark(cases=["argmax_full"], warmup=2,
                                  repetitions=5, **SMOKE)
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
    payload = bench.run_benchmark(cases=["argmax_full"], warmup=0,
                                  repetitions=1, **SMOKE)
    assert payload["cases"][0]["warmup"] == 0


def test_the_warmup_and_repetition_policy_is_one_shared_rule():
    """No case picks its own count: one policy per mode, reported on every
    record, and none of the three numbers is a performance promise."""
    assert bench.SMOKE_DEFAULTS == {"warmup": 1, "repetitions": 3}
    assert bench.DEFAULTS == {"warmup": 3, "repetitions": 11}
    assert bench.PROFILE_DEFAULTS == {"warmup": 5, "repetitions": 25}
    for policy in (bench.SMOKE_DEFAULTS, bench.DEFAULTS,
                   bench.PROFILE_DEFAULTS):
        assert policy["repetitions"] >= 1 and policy["warmup"] >= 1
    assert (bench.SMOKE_DEFAULTS["repetitions"]
            < bench.DEFAULTS["repetitions"]
            < bench.PROFILE_DEFAULTS["repetitions"])
    for spec in bench.CASES.values():
        assert "warmup" not in spec and "repetitions" not in spec


# ===========================================================================
# 8. Statistics
# ===========================================================================


def test_the_percentile_rule_has_known_answers():
    assert bench.percentile([10], 0.25) == 10.0
    assert bench.percentile([10], 0.75) == 10.0
    assert bench.percentile([1, 2, 3, 4], 0.25) == pytest.approx(1.75)
    assert bench.percentile([1, 2, 3, 4], 0.75) == pytest.approx(3.25)
    assert bench.percentile([1, 2, 3], 0.25) == pytest.approx(1.5)
    assert bench.percentile([1, 2, 3], 0.75) == pytest.approx(2.5)
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
    assert "mean_ns" in summary


@needs_native
def test_no_measured_value_carries_a_pass_fail_meaning():
    """Nothing in the payload classifies a duration. A field that did would
    be a threshold in prose."""
    payload = bench.run_benchmark(cases=["index_select_duplicates"], **SMOKE)
    blob = json.dumps(payload).lower()
    for banned in ("regression", "acceptable", "budget", "threshold exceeded",
                   "too slow", "passed timing", "failed timing", "target_ns",
                   "limit_ns", "max_ns_allowed", "speedup"):
        assert banned not in blob, banned
    for record in payload["cases"]:
        stats = record["statistics"]
        assert stats["min_ns"] <= stats["median_ns"] <= stats["max_ns"]
        assert stats["iqr_ns"] >= 0.0
        assert len(stats["samples_ns"]) == stats["sample_count"]


def test_no_timing_threshold_exists_anywhere_in_the_harness_or_this_module():
    """Structural, over both files' **code**: no comparison of a duration
    against a constant, and no constant that could be one."""
    for label, code in (("benchmark", BENCHMARK_CODE),
                        ("owner test", code_only(
                            Path(__file__).read_text(encoding="utf-8")))):
        for banned in ("THRESHOLD", "MAX_NS", "MIN_NS", "BUDGET_NS",
                       "SPEEDUP", "assert elapsed", "assert median",
                       "assert duration", "fail_under"):
            assert banned not in code, (label, banned)
        assert not re.search(r"median_ns\s*[<>]", code), label
        assert not re.search(r"elapsed\s*[<>]", code), label


def test_the_threshold_scanner_can_actually_fail():
    """Negative control: a planted executable threshold must be caught."""
    planted = code_only("median_ns = 5\nassert median_ns < 1000\n")
    assert re.search(r"median_ns\s*[<>]", planted)
    assert "assert median" in planted
    # ...and a *sentence* that merely explains the rule must not be.
    prose = code_only('"""No median_ns < budget assertion exists."""\n'
                      'value = 1\n')
    assert not re.search(r"median_ns\s*[<>]", prose)


# ===========================================================================
# 9. Dtype separation and role honesty
# ===========================================================================


def test_the_dtype_rows_are_the_registrys_and_are_three_questions():
    assert bench.FLOATING_DTYPES == ("float64", "float32")
    assert bench.FLOATING_DTYPES == cpp.SUPPORTED_DTYPES
    assert bench.INDEX_DTYPES == ("int64",) == cpp.INDEX_DTYPES
    assert bench.INDEX_DTYPE == "int64"
    assert bench.MEASURED_DTYPES == ("float64", "float32", "int64")
    assert set(bench.NUMPY_DTYPES) == set(bench.MEASURED_DTYPES)
    assert set(bench.BIT_DTYPES) == set(bench.FLOATING_DTYPES)
    # int64 is an index/result dtype and not a compute one — the whole
    # point of keeping the two rows apart.
    assert bench.INDEX_DTYPE not in cpp.SUPPORTED_DTYPES
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")
    assert cpp.normalize_dtype(None) == "float64"


def test_every_case_declares_the_role_its_dtype_plays():
    assert bench.DTYPE_ROLES == ("floating_source", "index_or_result")
    for name, spec in bench.CASES.items():
        if spec["workload"] in ("integer_construction",
                                "host_materialization"):
            assert tuple(spec["dtypes"]) == bench.INDEX_DTYPES, name
            assert spec["dtype_role"] == bench.ROLE_INDEX, name
        else:
            assert tuple(spec["dtypes"]) == bench.FLOATING_DTYPES, name
            assert spec["dtype_role"] == bench.ROLE_FLOATING_SOURCE, name
        # No case straddles the two registries.
        assert not (set(spec["dtypes"]) & set(bench.FLOATING_DTYPES)
                    and set(spec["dtypes"]) & set(bench.INDEX_DTYPES)), name


@needs_native
def test_dtype_filtering_selects_the_families_it_names():
    integer = bench.run_benchmark(dtypes=["int64"], **SMOKE)
    assert {row["workload"] for row in integer["cases"]} == {
        "integer_construction", "host_materialization"}
    assert {row["dtype"] for row in integer["cases"]} == {"int64"}
    assert {row["dtype_role"] for row in integer["cases"]} == {
        "index_or_result"}
    narrow = bench.run_benchmark(dtypes=["float32"], **SMOKE)
    assert {row["workload"] for row in narrow["cases"]} == {
        "argmax", "index_select"}
    assert {row["dtype"] for row in narrow["cases"]} == {"float32"}
    assert {row["dtype_role"] for row in narrow["cases"]} == {
        "floating_source"}


@needs_native
def test_both_floating_widths_are_measured_and_every_case_appears_at_each():
    payload = bench.run_benchmark(**SMOKE)
    by_dtype = {}
    for record in payload["cases"]:
        by_dtype.setdefault(record["dtype"], set()).add(record["case"])
    assert set(by_dtype) == {"float64", "float32", "int64"}
    floating = REQUIRED_COVERAGE["argmax"] | REQUIRED_COVERAGE["index_select"]
    assert by_dtype["float64"] == by_dtype["float32"] == floating
    integer = (REQUIRED_COVERAGE["integer_construction"]
               | REQUIRED_COVERAGE["host_materialization"])
    assert by_dtype["int64"] == integer


def _cross_dtype_key_offenders(payload):
    """Every payload key that names a dtype comparison.

    Keys are compared **token by token** rather than as substrings:
    ``configuration`` innocently contains other words' letters, and a
    checker that tripped on prose would be noise rather than a guardrail.
    A key naming *two* of the three measured dtypes, or naming a speed
    verdict, is the real offence.
    """
    offenders = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                tokens = set(key.lower().replace("-", "_").split("_"))
                named = tokens & {"float32", "float64", "int64"}
                if len(named) >= 2:
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
    """Negative control: a fabricated payload naming two widths, or naming
    a verdict, must be caught — and the honest keys the payload really uses
    must not be."""
    assert _cross_dtype_key_offenders(
        {"float32_over_float64": 1.2}) == ["float32_over_float64"]
    assert _cross_dtype_key_offenders(
        {"int64_vs_float64_ratio": 2.0}) == ["int64_vs_float64_ratio"]
    assert _cross_dtype_key_offenders(
        {"cases": [{"speedup": 2.0}]}) == ["speedup"]
    assert _cross_dtype_key_offenders(
        {"deep": {"nested": {"regression_ns": 5}}}) == ["regression_ns"]
    assert _cross_dtype_key_offenders({
        "ratio_to_reference": None, "ratio_meaning": None, "dtype": "float64",
        "dtype_role": "floating_source", "configuration": "smoke",
        "reference_type": "native_only",
    }) == []


@needs_native
def test_no_cross_dtype_ratio_appears_anywhere_in_the_payload():
    payload = bench.run_benchmark(**SMOKE)
    assert _cross_dtype_key_offenders(payload) == []
    blob = json.dumps(payload).lower()
    for banned in ("speedup", "vs_float", "faster than", "x faster",
                   "float32/float64 ratio of", "improvement over",
                   "float64_over_float32", "float32_over_float64",
                   "int64_over_float", "float_over_int64"):
        assert banned not in blob, banned


@needs_native
def test_the_report_publishes_no_dtype_speed_claim():
    payload = bench.run_benchmark(cases=["argmax_full"], **SMOKE)
    report = bench.format_report(payload).lower()
    for banned in ("faster", "slower", "speedup", "outperform", "beats",
                   "wins", "loses", "acceptable", "budget", "regression"):
        assert banned not in report, banned
    assert "no" in report and "float32/float64 ratio" in report
    assert "int64/floating ratio" in report


@needs_native
def test_each_dtype_is_gated_independently_and_nothing_is_widened():
    payload = bench.run_benchmark(cases=["argmax_axis_contiguous"], **SMOKE)
    for record in payload["cases"]:
        assert record["dtype"] in bench.MEASURED_DTYPES
        assert record["correctness"]["passed"] is True
    # No widening conversion and no tolerance is used to make a comparison
    # succeed. Scanned over the code, not the prose that explains the rule.
    for banned in ("astype", "allclose", "isclose", "approx", "atol",
                   "rtol", "round(", "np.float64(", "np.float32(",
                   "equal_nan"):
        assert banned not in BENCHMARK_CODE, banned
    # The one dtype conversion the harness performs is at input
    # construction, before any tensor exists, and it is explicit.
    assert "ascontiguousarray(values, dtype=NUMPY_DTYPES[dtype])" in (
        BENCHMARK_CODE)


def test_no_wording_implies_that_int64_is_a_supported_compute_dtype():
    """The one claim a dtype-filtering CLI could accidentally make."""
    flat = " ".join(BENCHMARK_SOURCE.split()).lower()
    for banned in (
        "supported_dtypes includes int64",
        "int64 is a supported compute dtype",
        "int64 is supported",
        "int64 joins supported_dtypes",
        "supported dtypes are float64, float32, and int64",
    ):
        assert banned not in flat, banned
    # ...and the honest statement is actually made, more than once.
    assert flat.count("index/result dtype") >= 3
    assert "not a supported compute dtype" in flat
    # Negative control for the scan itself.
    assert "int64 is supported" in " int64 is supported ".lower()


def test_the_dtype_option_help_distinguishes_the_two_registries():
    parser = bench.build_parser()
    help_text = " ".join(parser.format_help().split()).lower()
    assert "int64 selects the index/result families" in help_text
    assert "float64/float32 select the floating source width" in help_text
    assert "is not in supported_dtypes" in help_text


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
    integer_rows = len(REQUIRED_COVERAGE["integer_construction"]) + len(
        REQUIRED_COVERAGE["host_materialization"])
    floating_rows = len(REQUIRED_COVERAGE["argmax"]) + len(
        REQUIRED_COVERAGE["index_select"])
    assert len(payload["cases"]) == integer_rows + 2 * floating_rows == 25
    # One JSON object and nothing else: no prose, no banner, no progress.
    assert result.stdout.strip().startswith("{")
    assert result.stdout.strip().endswith("}")
    assert result.stdout.count("\n") == 1
    # Strictly parsed: NaN and Infinity are Python extensions, not JSON,
    # and a timing that produced one would make the payload unreadable by
    # any other parser. The hook fires on the *tokens*, so prose that
    # merely spells "NaN" — the argmax rule's own explanation does — is
    # correctly ignored.
    strict = json.loads(result.stdout, parse_constant=_reject_constant)
    assert strict == payload
    _require_finite_numbers(strict)


def _reject_constant(token):                          # pragma: no cover
    raise AssertionError(f"the payload carries a non-JSON constant {token!r}")


def _require_finite_numbers(node, path="payload"):
    """Every number in the payload is an ordinary finite value."""
    if isinstance(node, dict):
        for key, value in node.items():
            _require_finite_numbers(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _require_finite_numbers(value, f"{path}[{index}]")
    elif isinstance(node, float):
        assert node == node and node not in (float("inf"), float("-inf")), (
            f"{path}: {node!r}")


def test_the_strict_json_helpers_can_actually_fail():
    """Negative controls: the constant hook must fire on a real NaN token,
    and the finiteness walk must catch a non-finite number."""
    with pytest.raises(AssertionError, match="non-JSON constant"):
        json.loads('{"value": NaN}', parse_constant=_reject_constant)
    with pytest.raises(AssertionError):
        _require_finite_numbers({"deep": [float("inf")]})
    with pytest.raises(AssertionError):
        _require_finite_numbers({"deep": {"value": float("nan")}})
    _require_finite_numbers({"ok": [1, 2.5, "three", None, True]})


@needs_native
def test_cli_smoke_human_output_is_readable_and_carries_the_disclaimer():
    result = run_cli("--smoke", "--workload", "argmax")
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert "Local characterization only" in out
    assert "no result file is written" in out
    assert "measured separately" in out
    assert "float64" in out and "float32" in out
    assert "native_only" in out
    # The four questions are separated by name in the default output.
    for workload in EXPECTED_WORKLOADS:
        assert workload in bench.format_report(
            bench.run_benchmark(**SMOKE))
    assert str(REPO_ROOT) not in out


@needs_native
def test_cli_default_mode_runs_the_full_configuration():
    result = run_cli("--case", "argmax_full", "--json")
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout)
    assert payload["mode"] == "full"
    assert payload["methodology"]["repetitions"] == bench.DEFAULTS[
        "repetitions"]


@needs_native
def test_cli_selects_exactly_one_case_and_exactly_one_workload():
    single = run_cli("--smoke", "--json", "--case", "int64_construct_matrix")
    assert single.returncode == 0, single.stderr[-2000:]
    payload = json.loads(single.stdout)
    assert payload["selected_cases"] == ["int64_construct_matrix"]
    assert {row["case"] for row in payload["cases"]} == {
        "int64_construct_matrix"}
    family = run_cli("--smoke", "--json", "--workload", "index_select")
    assert family.returncode == 0, family.stderr[-2000:]
    payload = json.loads(family.stdout)
    assert set(payload["selected_cases"]) == REQUIRED_COVERAGE["index_select"]
    assert payload["selected_workloads"] == ["index_select"]


@needs_native
def test_cli_selects_a_single_dtype_in_either_registry():
    narrow = run_cli("--smoke", "--json", "--dtype", "float32")
    assert narrow.returncode == 0, narrow.stderr[-2000:]
    payload = json.loads(narrow.stdout)
    assert payload["dtypes"] == ["float32"]
    assert {row["dtype"] for row in payload["cases"]} == {"float32"}
    integer = run_cli("--smoke", "--json", "--dtype", "int64")
    assert integer.returncode == 0, integer.stderr[-2000:]
    payload = json.loads(integer.stdout)
    assert payload["dtypes"] == ["int64"]
    assert {row["dtype"] for row in payload["cases"]} == {"int64"}


@needs_native
def test_cli_profile_mode_runs_one_case_at_no_smaller_a_shape():
    result = run_cli("--json", "--profile", "int64_construct_small_contiguous")
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout)
    assert payload["mode"] == "profile"
    assert payload["selected_cases"] == ["int64_construct_small_contiguous"]
    assert payload["methodology"]["repetitions"] == bench.PROFILE_DEFAULTS[
        "repetitions"]


@pytest.mark.parametrize("arguments", [
    ("--case", "no_such_case"),
    ("--workload", "no_such_workload"),
    ("--dtype", "float16"),
    ("--dtype", "int32"),
    ("--case", "argmax_full", "--workload", "argmax"),
    ("--profile", "argmax_full", "--case", "argmax_full"),
    ("--profile", "argmax_full", "--smoke"),
    ("--smoke", "--workload", "argmax", "--dtype", "int64"),
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
                   "--report", "--csv", "--file", "--results", "--check",
                   "--threshold", "--budget"):
        assert banned not in result.stdout, banned


@needs_native
def test_cli_reports_a_correctness_failure_with_a_nonzero_exit(tmp_path):
    """A failed gate exits nonzero with clean stdout — arranged from
    outside, through a sitecustomize shim, so **no repository source is
    modified** to build the control."""
    shim = tmp_path / "sitecustomize.py"
    shim.write_text(
        "from tensorforge.backends import cpp\n"
        "_original = cpp.NativeTensorCore.to_numpy\n"
        "def _perturbed(self):\n"
        "    return _original(self) + 1\n"
        "cpp.NativeTensorCore.to_numpy = _perturbed\n",
        encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(REPO_ROOT)
    result = run_cli("--smoke", "--case", "int64_construct_small_contiguous",
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


def test_the_tree_fingerprint_can_actually_fail(tmp_path):
    """Negative control for every no-file assertion below."""
    before = _tree_fingerprint()
    planted = REPO_ROOT / "docs" / "_k8_fingerprint_probe.tmp"
    planted.write_text("probe", encoding="utf-8")
    try:
        assert _tree_fingerprint() != before
    finally:
        planted.unlink()
    assert _tree_fingerprint() == before


@needs_native
def test_running_the_benchmark_writes_no_file():
    before = _tree_fingerprint()
    bench.run_benchmark(cases=["int64_to_numpy_small_contiguous"], **SMOKE)
    assert _tree_fingerprint() == before


@needs_native
def test_the_cli_writes_no_file_in_any_mode(tmp_path):
    """Run from an empty working directory and prove nothing appeared —
    there and in the repository."""
    before = _tree_fingerprint()
    for arguments in (("--smoke", "--case", "argmax_full"),
                      ("--smoke", "--json", "--case",
                       "index_select_offset_index"),
                      ("--smoke", "--workload", "host_materialization"),
                      ("--smoke", "--dtype", "int64")):
        result = run_cli(*arguments, cwd=tmp_path)
        assert result.returncode == 0, result.stderr[-2000:]
    assert list(tmp_path.iterdir()) == [], "the harness wrote a file"
    assert _tree_fingerprint() == before


@needs_native
def test_a_failed_gate_writes_no_file_either(tmp_path, monkeypatch):
    before = _tree_fingerprint()
    original = cpp.NativeTensorCore.to_numpy
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy",
                        lambda self: original(self) + 1)
    with pytest.raises(AssertionError):
        bench.run_case("argmax_full", "float64", 0, 1, "smoke")
    assert _tree_fingerprint() == before


def test_no_result_artifact_of_any_kind_exists():
    for pattern in ("*.json", "*.csv", "*.txt", "*.md", "*.npz", "*.png",
                    "*.svg", "*.pkl", "*.db"):
        assert not list((REPO_ROOT / "benchmarks").glob(pattern)), pattern
    assert not list(REPO_ROOT.glob("benchmark*.json"))
    assert not (REPO_ROOT / "benchmark_results").exists()
    assert not (REPO_ROOT / "benchmarks" / "results").exists()
    assert not list((REPO_ROOT / "docs").glob("*native_integer*.json"))


def test_the_benchmark_opens_no_file_and_imports_no_writer():
    for banned in ("write_text", "write_bytes", "read_bytes", "read_text",
                   "os.makedirs", "mkdir", "savez", "np.save", "to_csv",
                   "csv.", "pickle", "shelve", "sqlite", "tempfile",
                   "shutil", "NamedTemporary", "mkdtemp", "json.dump(",
                   "urllib", "requests", "socket", "download", "http"):
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
    assert _imported_modules(BENCHMARK_SOURCE) == {
        "argparse", "json", "math", "os", "platform", "statistics", "sys",
        "time", "pathlib", "numpy", "tensorforge"}


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
    planted = code_only("values = np.random.rand(4)\n")
    assert "np.random.rand" in planted


@needs_native
def test_running_the_benchmark_moves_no_global_rng_state():
    numpy_before = np.random.get_state()[1][:8].copy()
    import random as _random
    python_before = _random.getstate()
    bench.run_benchmark(cases=["index_select_large_selection"], **SMOKE)
    assert np.array_equal(np.random.get_state()[1][:8], numpy_before)
    assert _random.getstate() == python_before


@needs_native
def test_repeated_setup_builds_equal_but_independent_inputs():
    first = bench.host_int64((4, 3), 4242)
    second = bench.host_int64((4, 3), 4242)
    assert np.array_equal(first, second)
    assert not np.shares_memory(first, second)
    left = bench.host_floating((4, 3), "float32", 99)
    right = bench.host_floating((4, 3), "float32", 99)
    assert np.array_equal(bench.bits_of(left, "float32"),
                          bench.bits_of(right, "float32"))
    assert not np.shares_memory(left, right)


@needs_native
def test_repeated_runs_produce_identical_correctness_metrics():
    """Deterministic inputs mean deterministic gates; only the timings may
    move between runs."""
    def fingerprint(payload):
        return [(row["case"], row["dtype"], row["correctness"], row["config"],
                 row["shape"], row["seed"], row["summary"])
                for row in payload["cases"]]

    first = bench.run_benchmark(workloads=["host_materialization"], **SMOKE)
    second = bench.run_benchmark(workloads=["host_materialization"], **SMOKE)
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
    assert backend["index_dtypes"] == list(cpp.INDEX_DTYPES)
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


def test_no_environment_value_changes_what_the_harness_measures():
    """Metadata is reported, never consulted: nothing branches on it."""
    tree = ast.parse(BENCHMARK_SOURCE)
    readers = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)
               and node.name in ("thread_environment", "environment")]
    assert len(readers) == 2
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)) and node not in readers:
            test_source = ast.unparse(node.test)
            assert "os.environ" not in test_source, test_source
            assert "platform." not in test_source, test_source


@needs_native
def test_the_payload_leaks_no_path_user_or_secret():
    payload = bench.run_benchmark(cases=["argmax_axis_long"], **SMOKE)
    blob = json.dumps(payload)
    assert str(REPO_ROOT) not in blob
    assert str(Path.home()) not in blob
    assert os.getcwd() not in blob
    user = getpass.getuser()
    if user and len(user) >= 4:
        assert user not in blob, "the payload names the current user"
    for banned in ("PASSWORD", "SECRET", "API_KEY", "PYTHONPATH",
                   "HOME=", "USERPROFILE"):
        assert banned not in blob.upper(), banned


@needs_native
def test_a_full_smoke_run_returns_live_storage_exactly_to_baseline(
        live_storages):
    baseline = settled(live_storages)
    bench.run_benchmark(**SMOKE)
    assert settled(live_storages) == baseline


@needs_native
def test_repeated_runs_leak_nothing(live_storages):
    baseline = settled(live_storages)
    for _ in range(3):
        bench.run_benchmark(workloads=["index_select"], **SMOKE)
        assert settled(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("name", EXPECTED_CASES)
def test_every_case_returns_live_storage_to_baseline(name, live_storages):
    baseline = settled(live_storages)
    bench.run_case(name, first_dtype(name), 0, 1, "smoke")
    assert settled(live_storages) == baseline


@needs_native
def test_a_selected_full_case_returns_live_storage_to_baseline(live_storages):
    baseline = settled(live_storages)
    bench.run_benchmark(cases=["int64_construct_small_contiguous"])
    assert settled(live_storages) == baseline


@needs_native
def test_no_live_object_is_returned_from_a_completed_run():
    """The payload is plain Python records. A tensor surviving in one would
    be a leak in the shape of a convenience."""
    payload = bench.run_benchmark(cases=["index_select_contiguous"], **SMOKE)
    _require_plain(payload)


def test_the_harness_closes_explicitly_and_relies_on_no_collection():
    for banned in ("gc.collect", "import gc", "weakref", "__del__",
                   "_arm_alloc_failure", "tf_test_arm_alloc_failure",
                   "fault_injection"):
        assert banned not in BENCHMARK_CODE, banned
    assert ".close()" in BENCHMARK_CODE
    # Cleanup for a call that returns native storage is a real close, not a
    # drop.
    assert "result.close()" in BENCHMARK_CODE


# ---------------------------------------------------------------------------
# 12a. Ownership — every constructed owning tensor is *explicitly* closed
#
# The two proofs below are deliberately independent and neither may stand in
# for the other:
#
#   * the **runtime** proof holds a strong reference to every owning tensor
#     the harness constructs, so refcounting, ``__del__``, and ``gc.collect``
#     cannot release one on its behalf. If a builder abandoned an owner, its
#     storage would still be live at the assertion and the test would fail.
#   * the **structural** proof reads the harness's AST and rejects a
#     construction that is overwritten, discarded inline, or never closed —
#     the shapes a runtime proof only catches once someone writes a case that
#     exercises them.
#
# Every live-storage assertion here reads ``len(live_storages)`` **directly**.
# ``settled()`` is not used: it calls ``gc.collect()``, which is exactly the
# crutch this section exists to remove.
# ---------------------------------------------------------------------------

# The two owning constructors the benchmark is allowed to call. Everything
# they return is caller-owned native storage; a NumPy array is not.
OWNING_CONSTRUCTORS = ("NativeTensor.from_int64_array", "NativeTensor.from_array")


class _RetainingConstructors:
    """Instrumentation that keeps a **strong** reference to every owning
    tensor the code under test constructs.

    Installed by hand rather than through ``monkeypatch`` so an ``undo()``
    elsewhere in a test cannot silently disarm it, and restored in a
    ``finally``. It delegates to the real classmethods, so what is measured
    is production construction rather than a stand-in.
    """

    def __init__(self):
        self.tensors = []
        self._saved = {}

    def __enter__(self):
        for name in ("from_int64_array", "from_array"):
            self._saved[name] = NativeTensor.__dict__[name]
        original_int64 = NativeTensor.from_int64_array
        original_array = NativeTensor.from_array

        def tracked_int64(values, *, requires_grad=False):
            tensor = original_int64(values, requires_grad=requires_grad)
            self.tensors.append(tensor)
            return tensor

        def tracked_array(values, dtype=None, device="cpu",
                          requires_grad=False):
            tensor = original_array(values, dtype=dtype, device=device,
                                    requires_grad=requires_grad)
            self.tensors.append(tensor)
            return tensor

        NativeTensor.from_int64_array = staticmethod(tracked_int64)
        NativeTensor.from_array = staticmethod(tracked_array)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name, original in self._saved.items():
            setattr(NativeTensor, name, original)
        return False

    @property
    def owning(self):
        return [tensor for tensor in self.tensors if tensor.owns_core]

    @property
    def unclosed_owners(self):
        return [tensor for tensor in self.owning if not tensor.closed]


@needs_native
def test_the_retaining_instrument_can_actually_detect_an_abandoned_owner(
        live_storages):
    """Negative control for the runtime proof below, and the one that makes
    it a regression rather than a ritual.

    The **exact** abandoned-owner shape is planted — an owning tensor
    constructed into a name that is immediately overwritten by a second
    construction — and the instrument must report the first tensor still
    open with its storage still live, *while a strong reference to it is
    held*. An instrument that could not see this would make the proof
    below vacuous.
    """
    baseline = len(live_storages)
    wanted = np.ascontiguousarray([0, 1, 2], dtype=np.int64)
    with _RetainingConstructors() as retained:
        # The planted defect, verbatim.
        index_owner = NativeTensor.from_int64_array(wanted)
        index_owner = NativeTensor.from_int64_array(wanted)
        index_tensor = index_owner
        # Exactly what a correct teardown would close: the survivor.
        index_tensor.close()

        assert len(retained.owning) == 2, "the control built the wrong shape"
        abandoned = retained.unclosed_owners
        assert len(abandoned) == 1, (
            "the instrument did not notice the overwritten owner")
        assert abandoned[0] is not index_owner
        # ...and its storage is still live, with no collection consulted.
        assert len(live_storages) == baseline + 1, (
            "the abandoned owner's storage was released by something other "
            "than an explicit close")
        abandoned[0].close()
        assert len(live_storages) == baseline
    # The instrument really was uninstalled.
    assert NativeTensor.__dict__["from_int64_array"] is not None
    fresh = NativeTensor.from_int64_array(wanted)
    try:
        assert fresh not in retained.tensors
    finally:
        fresh.close()
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("name", ["index_select_contiguous",
                                  "index_select_duplicates",
                                  "index_select_large_selection",
                                  "index_select_offset_index",
                                  "index_select_noncontiguous_source"])
def test_index_select_setup_abandons_no_owning_tensor(name, live_storages):
    """**Retained-reference** lifecycle proof for the family that builds two
    owning tensors — a floating source and an ``int64`` index — and, in one
    case, a borrowing view of each.

    Every owning tensor the builder, the gate, and the measured repetitions
    construct is held in a strong-reference list for the whole test, so
    nothing here can be released by refcounting, by ``__del__``, or by a
    collection this test never runs. After ``teardown()`` every one of them
    must be **explicitly** closed and live storage must be back at baseline
    while those references are still alive — which is only possible if the
    harness closed each of them itself.
    """
    spec = bench.CASES[name]
    config = spec["configurations"]["smoke"]
    baseline = len(live_storages)
    with _RetainingConstructors() as retained:
        case = spec["build"]("float64", config, spec)
        try:
            case.check()
            for _ in range(2):
                state = case.prepare()
                result = case.run(state)
                case.cleanup(state, result)
        finally:
            case.teardown()

        # The builder alone constructs a floating source and an int64 index;
        # the gate constructs more. Anything less would mean this test is
        # not exercising the path it names.
        assert len(retained.owning) >= 2, retained.owning
        assert retained.unclosed_owners == [], (
            f"{name}: {len(retained.unclosed_owners)} owning tensor(s) were "
            f"abandoned without an explicit close")
        # Read directly: no gc.collect(), and every constructed tensor is
        # still strongly referenced by ``retained``.
        assert len(live_storages) == baseline, (
            f"{name}: native storage outlived teardown while its wrapper was "
            f"still referenced")
    assert len(live_storages) == baseline


@needs_native
def test_every_case_closes_every_owning_tensor_it_constructs(live_storages):
    """The same retained-reference proof, once per case, so no family can
    abandon an owner where another family's proof would not look."""
    for name, spec in bench.CASES.items():
        dtype = first_dtype(name)
        baseline = len(live_storages)
        with _RetainingConstructors() as retained:
            case = spec["build"](dtype, spec["configurations"]["smoke"], spec)
            try:
                case.check()
                state = case.prepare()
                case.cleanup(state, case.run(state))
            finally:
                case.teardown()
            assert retained.owning, name
            assert retained.unclosed_owners == [], name
            assert len(live_storages) == baseline, name


# --- structural ownership guard --------------------------------------------


def _ownership_offences(source):
    """Every owning-tensor construction in ``source`` that is abandoned.

    Deliberately **not** a whole-program ownership analyzer. It protects the
    four shapes this benchmark can actually get wrong:

    1. a construction assigned to a name that is **overwritten by another
       construction in the same block** before that name is closed — the
       abandoned-owner defect;
    2. a construction used **inline as an argument**, so nothing can close
       it;
    3. a construction evaluated as a **bare expression statement**;
    4. a construction assigned to a name that is **never closed** anywhere
       in its enclosing function or the closures nested inside it, and never
       returned.

    A ``return ctor(...)`` is *allowed*: ownership transfers to the caller,
    which is how the timed ``run`` closures hand their result to
    ``_close_result``. Two constructions assigning the same name in
    **different branches** of an ``if`` are allowed too — they are mutually
    exclusive, which is exactly the offset/ordinary index split.
    """
    tree = ast.parse(source)
    offences = []

    # Map every node to its enclosing function, so "closed somewhere in this
    # function or a closure inside it" is answerable.
    enclosing = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Module)):
            for child in ast.walk(node):
                enclosing.setdefault(child, node)

    def closed_names(scope):
        names = set()
        for node in ast.walk(scope):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "close"
                    and isinstance(node.func.value, ast.Name)):
                names.add(node.func.value.id)
        return names

    def returned_names(scope):
        names = set()
        for node in ast.walk(scope):
            if isinstance(node, ast.Return) and isinstance(node.value,
                                                           ast.Name):
                names.add(node.value.id)
        return names

    def is_construction(node):
        return (isinstance(node, ast.Call)
                and ast.unparse(node.func) in OWNING_CONSTRUCTORS)

    # (1) overwritten-before-close, within one statement list.
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            built = {}
            for statement in block:
                closed_here = closed_names(statement)
                for name in list(built):
                    if name in closed_here:
                        built.pop(name)
                if (isinstance(statement, ast.Assign)
                        and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and is_construction(statement.value)):
                    target = statement.targets[0].id
                    if target in built:
                        offences.append(
                            (built[target], "overwritten before close",
                             target))
                    built[target] = statement.lineno
                elif isinstance(statement, ast.Assign):
                    # A re-assignment from something that is *not* a
                    # construction still abandons the owner.
                    for element in statement.targets:
                        if (isinstance(element, ast.Name)
                                and element.id in built):
                            offences.append(
                                (built[element.id],
                                 "overwritten before close", element.id))
                            built.pop(element.id)

    # (2)-(4) every construction, classified by how its result is used.
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    for node in ast.walk(tree):
        if not is_construction(node):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Return):
            continue                                   # ownership transfer
        if isinstance(parent, ast.Expr):
            offences.append((node.lineno, "discarded as a statement", None))
            continue
        if not (isinstance(parent, ast.Assign)
                and len(parent.targets) == 1
                and isinstance(parent.targets[0], ast.Name)):
            offences.append((node.lineno, "discarded inline", None))
            continue
        target = parent.targets[0].id
        scope = enclosing.get(node, tree)
        if target in closed_names(scope) or target in returned_names(scope):
            continue
        # An alias assigned from this owner and then closed counts too.
        aliased = {statement.targets[0].id
                   for statement in ast.walk(scope)
                   if isinstance(statement, ast.Assign)
                   and len(statement.targets) == 1
                   and isinstance(statement.targets[0], ast.Name)
                   and isinstance(statement.value, ast.Name)
                   and statement.value.id == target}
        if aliased & closed_names(scope):
            continue
        offences.append((node.lineno, "never closed", target))
    return sorted(set(offences))


def test_the_benchmark_abandons_no_constructed_owning_tensor():
    """The structural half: no owning construction in the harness is
    overwritten, discarded, or left unclosed."""
    assert _ownership_offences(BENCHMARK_SOURCE) == []


def test_the_ownership_scanner_rejects_the_planted_abandoned_owner():
    """The negative control the guard exists for, containing the **exact**
    bad shape, plus the three other ways an owner can be abandoned — and the
    two legitimate shapes that must NOT be flagged."""
    planted = (
        "def build_probe():\n"
        "    index_owner = NativeTensor.from_int64_array(wanted)\n"
        "    index_owner = NativeTensor.from_int64_array(wanted)\n"
        "    index_tensor = index_owner\n"
        "    def teardown():\n"
        "        index_owner.close()\n"
        "    return teardown\n")
    offences = _ownership_offences(planted)
    assert offences, "the exact reported defect was not detected"
    assert offences[0][1] == "overwritten before close"
    assert offences[0][2] == "index_owner"

    inline = ("def probe():\n"
              "    use(NativeTensor.from_array(values, dtype='float64'))\n")
    assert [o[1] for o in _ownership_offences(inline)] == ["discarded inline"]

    bare = "def probe():\n    NativeTensor.from_int64_array(values)\n"
    assert [o[1] for o in _ownership_offences(bare)] == [
        "discarded as a statement"]

    never = ("def probe():\n"
             "    owner = NativeTensor.from_int64_array(values)\n"
             "    return owner.shape\n")
    assert [o[1] for o in _ownership_offences(never)] == ["never closed"]

    rebound = ("def probe():\n"
               "    owner = NativeTensor.from_int64_array(values)\n"
               "    owner = something_else()\n"
               "    owner.close()\n")
    assert [o[1] for o in _ownership_offences(rebound)] == [
        "overwritten before close"]

    # ...and the two shapes the harness legitimately uses.
    transfer = ("def probe():\n"
                "    return NativeTensor.from_int64_array(host)\n")
    assert _ownership_offences(transfer) == []

    branched = ("def probe():\n"
                "    if offset:\n"
                "        owner = NativeTensor.from_int64_array(padded)\n"
                "        view = owner.narrow(0, offset, count)\n"
                "    else:\n"
                "        owner = NativeTensor.from_int64_array(wanted)\n"
                "        view = owner\n"
                "    def teardown():\n"
                "        if view is not owner:\n"
                "            view.close()\n"
                "        owner.close()\n"
                "    return teardown\n")
    assert _ownership_offences(branched) == [], (
        "mutually exclusive branches were mistaken for an overwrite")

    reused = ("def probe():\n"
              "    owner = NativeTensor.from_int64_array(a)\n"
              "    owner.close()\n"
              "    owner = NativeTensor.from_int64_array(b)\n"
              "    owner.close()\n")
    assert _ownership_offences(reused) == [], (
        "a closed-then-rebuilt owner is not an abandonment")


def test_every_owning_construction_in_the_harness_is_enumerated():
    """The audit itself, written down: every executable call to either
    owning constructor, with the owner variable it enters and the close
    that releases it. NumPy arrays are deliberately outside this rule —
    they own no native storage."""
    constructions = [
        (node.lineno, ast.unparse(node.func))
        for node in ast.walk(ast.parse(BENCHMARK_SOURCE))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in OWNING_CONSTRUCTORS]
    assert len(constructions) == 9, constructions
    by_ctor = {}
    for _, func in constructions:
        by_ctor[func] = by_ctor.get(func, 0) + 1
    assert by_ctor == {"NativeTensor.from_int64_array": 6,
                       "NativeTensor.from_array": 3}, by_ctor
    # No third construction door is reachable from the harness at all.
    for banned in ("NativeTensor.zeros", "NativeTensor.full",
                   "NativeTensor._from_core", "NativeTensorCore.from_array",
                   "NativeStorage("):
        assert banned not in BENCHMARK_CODE, banned


def test_the_harness_reaches_for_no_private_seam():
    for banned in ("monkeypatch", "setattr(", "_uninitialized",
                   "_typed_from_array", "_typed_zeros", "_typed_full",
                   "_from_core", "_from_int64_array", "_trusted_dtype",
                   "_require_open", "NativeTensorView", "NativeTensorCore",
                   "NativeStorage", "_normalize_index_dtype"):
        assert banned not in BENCHMARK_CODE, banned


def test_the_harness_contains_no_concurrency_and_no_network():
    for banned in ("threading", "multiprocessing", "concurrent", "asyncio",
                   "Thread", "Lock", "RLock", "Semaphore", "Condition",
                   "Queue", "Future", "ThreadPoolExecutor",
                   "ProcessPoolExecutor", "start_new_thread", "socket",
                   "urlopen", "subprocess"):
        assert banned not in BENCHMARK_CODE, banned


def test_this_module_starts_no_thread():
    own = code_only(Path(__file__).read_text(encoding="utf-8"))
    for banned in ("threading", "Thread", "asyncio", "ThreadPoolExecutor",
                   "start_new_thread"):
        assert banned not in own, banned


# ===========================================================================
# 13. Scope — K8 shipped measurement and nothing else
# ===========================================================================


def test_the_benchmark_inventory_moved_from_nine_to_ten():
    benchmarks = sorted(path.name
                        for path in (REPO_ROOT / "benchmarks").glob("*.py")
                        if path.name != "__init__.py")
    assert len(benchmarks) == K8_BENCHMARK_COUNT == 10, benchmarks
    assert benchmarks == sorted([
        "benchmark_native_autograd.py",
        "benchmark_native_classification.py",
        "benchmark_native_cnn.py",
        "benchmark_native_cpu_performance.py",
        "benchmark_native_data_pipeline.py",
        "benchmark_native_dropout.py",
        "benchmark_native_dtype.py",
        "benchmark_native_integer.py",
        "benchmark_native_normalization.py",
        "cpp_backend.py",
    ])
    # K8's own artifact, named and subtracted, so the count still fails on
    # an *unrecorded* benchmark rather than absorbing one.
    assert K8_BENCHMARK in benchmarks
    assert len([name for name in benchmarks
                if name != K8_BENCHMARK]) == K7_BENCHMARK_COUNT
    # Every harness that declares an identity declares a *distinct* one,
    # and K8's is among them. Not every harness carries the constant —
    # the older ones predate the convention — so the check is over the
    # ones that do rather than over a count that would fail for a reason
    # this milestone did not cause.
    identities = []
    for name in benchmarks:
        text = (REPO_ROOT / "benchmarks" / name).read_text(encoding="utf-8")
        match = re.search(r'BENCHMARK_NAME = "([^"]+)"', text)
        if match:
            identities.append(match.group(1))
    assert bench.BENCHMARK_NAME in identities
    assert len(set(identities)) == len(identities), identities
    assert len(identities) >= 5, identities


def test_k8_added_no_example_and_the_example_inventory_is_unmoved():
    examples = sorted(path.name
                      for path in (REPO_ROOT / "examples").glob("*.py")
                      if path.name != "__init__.py")
    assert len(examples) == K7_EXAMPLE_COUNT == 17, examples
    assert "native_integer_indexing.py" in examples          # K6's
    for name in examples:
        assert "benchmark" not in name, name


def test_the_c_abi_and_ctest_inventories_did_not_move_at_k8():
    exports = set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        exports.update(re.findall(
            r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
            path.read_text(encoding="utf-8"), re.S))
    assert len(exports) == K7_EXPORT_COUNT == 56, sorted(exports)
    for landed in ("tf_core_argmax", "tf_core_index_select"):
        assert landed in exports, landed
    for absent in ("tf_core_gather", "tf_core_scatter", "tf_core_argmin",
                   "tf_core_max", "tf_core_index_select_backward",
                   "tf_core_benchmark", "tf_core_profile", "tf_timer_start"):
        assert absent not in exports, absent
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    registered = re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)
    assert len(registered) == K7_CTEST_COUNT == 27, registered
    assert len(set(registered)) == len(registered)
    for name in registered:
        assert "benchmark" not in name.lower(), name


def test_the_export_scanner_can_actually_fail():
    """Negative control, on a temporary string: an export really is found,
    so 56 is a measurement rather than an artifact."""
    source = 'TF_EXPORT void tf_core_probe(const void* a) { use(a); }'
    assert re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(", source,
                      re.S) == ["tf_core_probe"]
    assert re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                      "void tf_core_probe(void);", re.S) == []


@needs_native
def test_the_built_library_still_exports_the_same_inventory():
    """The stale-artifact guard: a library built before K4 would export 55
    and would fail here rather than quietly satisfying the tests above."""
    storage_tests = pytest.importorskip("test_native_storage_allocation")
    _, names = storage_tests.exported_names(cpp._LIBRARY_PATH)
    if names is None:                                  # pragma: no cover
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == K7_EXPORT_COUNT, exported


def test_the_registries_and_versions_are_exactly_what_k7_left():
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.INDEX_DTYPES == ("int64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert len(cpp._CHECKED_KERNELS) == K7_CHECKED_KERNELS == 38
    assert len(set(cpp._CHECKED_KERNELS)) == K7_CHECKED_KERNELS
    assert "argmax" in cpp.TENSOR_CORE_OPS
    assert "index_select" in cpp.TENSOR_CORE_OPS
    assert "argmax" not in cpp.AUTOGRAD_OPS
    assert "index_select" not in cpp.AUTOGRAD_OPS
    info = cpp.backend_info()
    assert info["supported_dtypes"] == ("float64", "float32")
    assert info["index_dtypes"] == ("int64",)
    assert info["dtype"] == "float64"
    assert info["stable_framework_integration"] is False


def test_the_checkpoint_loader_and_sampler_versions_are_unmoved():
    from tensorforge.experimental import native_checkpoint as checkpoint
    from tensorforge.experimental import native_data_loader as loader
    from tensorforge.experimental import native_sampler as sampler

    assert checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert checkpoint._FORMAT_VERSION == 3
    assert checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert loader._FORMAT == "tensorforge.native_data_loader"
    assert loader._FORMAT_VERSION == 1
    assert loader._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert sampler._FORMAT == "tensorforge.native_sampler"
    assert sampler._FORMAT_VERSION == 1
    assert sampler._SUPPORTED_FORMAT_VERSIONS == (1,)
    # No version 4 constant exists anywhere under src/.
    for path in (REPO_ROOT / "src").rglob("*.py"):
        code = code_only(path.read_text(encoding="utf-8"))
        assert "_FORMAT_VERSION = 4" not in code, path.name
        assert "(1, 2, 3, 4)" not in code, path.name


def test_no_absent_operation_appeared_beside_the_benchmark():
    """A benchmark measures what exists; it never motivates an addition."""
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ("argmin", "gather", "scatter", "scatter_add",
                       "embedding", "max", "amax", "take", "topk", "sort",
                       "argsort", "index_add", "index_put",
                       "index_select_backward", "benchmark", "profile",
                       "timeit"):
            assert not hasattr(owner, absent), (owner.__name__, absent)


def test_no_production_module_gained_a_timing_or_benchmark_hook():
    for path in (REPO_ROOT / "src").rglob("*.py"):
        code = code_only(path.read_text(encoding="utf-8"))
        for banned in ("perf_counter", "time.time", "monotonic",
                       "process_time", "default_timer", "benchmark",
                       "profile_hook", "TF_BENCHMARK", "TF_PROFILE"):
            assert banned not in code, (path.name, banned)


# ---------------------------------------------------------------------------
# Zero executable production code — the durable claim, and the pre-commit
# byte-identity proof beside it.
# ---------------------------------------------------------------------------

_PRODUCTION_DOCSTRING_EXCEPTION = "src/tensorforge/experimental/__init__.py"


def _executable_fingerprint(source):
    """A source file's **executable** shape, with every string constant
    blanked.

    Two files with the same fingerprint differ only in prose. This is what
    makes "the package docstring may change and nothing else may" a
    checkable statement rather than a promise.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.dump(tree)


def test_the_executable_fingerprint_can_actually_fail():
    """Negative control: a planted executable change must be visible, and a
    docstring-only change must not be."""
    base = '"""One docstring."""\nVALUE = 1\n'
    prose_only = '"""Another, longer docstring."""\nVALUE = 1\n'
    executable = '"""One docstring."""\nVALUE = 2\n'
    added = '"""One docstring."""\nVALUE = 1\nEXTRA = 3\n'
    assert _executable_fingerprint(base) == _executable_fingerprint(prose_only)
    assert _executable_fingerprint(base) != _executable_fingerprint(executable)
    assert _executable_fingerprint(base) != _executable_fingerprint(added)


def test_only_the_package_status_docstring_may_name_k8():
    """The durable half of the production claim: K8 vocabulary appears
    under ``src/`` only in the one status docstring the milestone is
    allowed to touch, and never in executable code anywhere."""
    naming = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        code = code_only(source)
        for banned in ("K8", "benchmark_native_integer", "BENCHMARK_NAME",
                       "SCHEMA_VERSION"):
            assert banned not in code, (relative, banned)
        if re.search(r"\bK8\b", source):
            naming.append(relative)
    assert naming == [_PRODUCTION_DOCSTRING_EXCEPTION], naming


def _git_bytes(*arguments):
    """One read-only Git query, or a documented skip where Git cannot
    answer."""
    try:
        done = subprocess.run(["git", *arguments], cwd=REPO_ROOT,
                              capture_output=True, timeout=300)
    except OSError:                                    # pragma: no cover
        pytest.skip("git is unavailable, so the tree cannot be inspected")
    if done.returncode != 0:                           # pragma: no cover
        pytest.skip("this tree has no git object to read")
    return done.stdout


def _normalized(data):
    """Bytes with line endings normalized.

    ``git show`` hands back the stored blob while the working tree carries
    whatever the checkout's ``core.autocrlf`` produced, so a raw byte
    comparison on Windows would report every file as changed. Normalizing
    the endings compares *content*, which is what "this milestone changed
    no production code" actually claims."""
    return data.replace(b"\r\n", b"\n")


def test_the_line_ending_normalizer_can_actually_fail():
    """Negative control: the normalizer must hide only line endings, and
    must still see a real content change."""
    assert _normalized(b"a\r\nb") == _normalized(b"a\nb")
    assert _normalized(b"a\r\nb") != _normalized(b"a\r\nc")


def test_no_production_file_differs_from_head_except_one_docstring():
    """The milestone half: before K8 is committed, every production source
    is byte-identical to ``HEAD`` except the package status docstring, and
    that one is **executable-code identical**.

    After the milestone is committed this comparison is trivially true,
    which is exactly right: the claim it makes is about the K8 patch, and
    ``test_only_the_package_status_docstring_may_name_k8`` above is the
    durable statement that survives it.
    """
    tracked = _git_bytes("ls-files", "src").decode("utf-8").splitlines()
    assert tracked, "no production source is tracked"
    checked = 0
    for relative in tracked:
        if not relative.endswith(".py"):
            continue
        checked += 1
        live = _normalized((REPO_ROOT / relative).read_bytes())
        head = _normalized(_git_bytes("show", f"HEAD:{relative}"))
        if relative == _PRODUCTION_DOCSTRING_EXCEPTION:
            assert (_executable_fingerprint(live.decode("utf-8"))
                    == _executable_fingerprint(head.decode("utf-8"))), relative
        else:
            assert live == head, relative
    assert checked > 10, checked


def test_no_production_or_build_surface_outside_src_changed():
    """``cpp/``, ``examples/``, ``.github/``, and the project files carry no
    K8 change at all — no rebuild was performed and none was required."""
    for directory in ("cpp", "examples", ".github"):
        for relative in _git_bytes(
                "ls-files", directory).decode("utf-8").splitlines():
            assert _normalized(
                (REPO_ROOT / relative).read_bytes()) == _normalized(
                _git_bytes("show", f"HEAD:{relative}")), relative
    for name in ("pyproject.toml", "uv.lock"):
        assert _normalized((REPO_ROOT / name).read_bytes()) == _normalized(
            _git_bytes("show", f"HEAD:{name}")), name


def test_no_ci_job_runs_or_gates_on_this_benchmark():
    workflow = (REPO_ROOT / ".github" / "workflows"
                / "tests.yml").read_text(encoding="utf-8")
    assert "benchmark_native_integer" not in workflow
    for banned in ("--profile", "threshold", "budget", "regression"):
        assert banned not in workflow, banned


def test_k9_has_not_started():
    """K8 is not K9: the closure module must still be absent, and this
    module performs no rebuild, no sanitizer run, and no cross-platform
    comparison.

    Scanned over **executable code**, not raw text — a ban list spelled in
    this module's own prose would otherwise fail on the sentence that
    documents it, which is the scanner failure mode this repository
    already has on record.
    """
    assert not (REPO_ROOT / K9_ARTIFACT).exists(), K9_ARTIFACT
    own_code = code_only(Path(__file__).read_text(encoding="utf-8"))
    for banned in ("tf_sanitize", "tf_build_tests", "asan", "ubsan",
                   "sanitize", "leaksanitizer"):
        assert banned not in own_code.lower(), banned
    # ``cmake`` and ``ctest`` are deliberately **not** in that list: this
    # module reads ``cpp/CMakeLists.txt`` to count registered CTests, which
    # is inspection rather than a build. What must not happen is *running*
    # one, and that is what the launcher scan below asserts.
    #
    # Every subprocess this module starts is either the benchmark itself
    # or a read-only Git query — never a build tool.
    launched = []
    for node in ast.walk(ast.parse(
            Path(__file__).read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call)
                and ast.unparse(node.func) == "subprocess.run"):
            first = node.args[0]
            assert isinstance(first, ast.List), ast.unparse(node)
            launched.append(ast.unparse(first.elts[0]))
    assert set(launched) == {"sys.executable", "'git'"}, launched


def test_the_subprocess_scanner_can_actually_fail():
    """Negative control: a planted build invocation must be visible."""
    planted = ast.parse("subprocess.run(['cmake', '-S', 'cpp'])")
    launched = [ast.unparse(node.args[0].elts[0])
                for node in ast.walk(planted)
                if isinstance(node, ast.Call)
                and ast.unparse(node.func) == "subprocess.run"]
    assert launched == ["'cmake'"]


def test_k8_shipped_no_optimization_and_no_production_import():
    """The harness measures the shipped runtime; it does not reach into
    it. No production module imports it, and it patches nothing."""
    package = REPO_ROOT / "src" / "tensorforge"
    for path in package.rglob("*.py"):
        modules = _imported_modules(path.read_text(encoding="utf-8"))
        assert "benchmarks" not in modules, path.name
        assert "benchmark_native_integer" not in modules, path.name
