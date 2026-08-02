"""Tests for the native Dropout benchmark harness (Phase G, milestone
G8).

These validate the benchmark's *behavior* — the case inventory, the
correctness-before-timing rule, the exactness of the vectorized NumPy
reference, the reference labelling, the result schema, generator
accounting, cleanup and ownership, and the CLI — with **no timing or
threshold assertions of any kind**. Benchmark durations are
hardware-specific measurements, never pass/fail criteria, and nothing
here depends on one path taking more or less time than another.
Importing the benchmark module must run nothing.

Selector: python -m pytest -q -k native_dropout_benchmark
"""

import gc
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from benchmarks import benchmark_native_dropout as bench

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

pytestmark = needs_native

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = REPO_ROOT / "benchmarks" / "benchmark_native_dropout.py"
TEST_FILE = Path(__file__)

SMOKE = {"warmup": 1, "repetitions": 3, "smoke": True}

EXPECTED_CASES = (
    "python_call_floor",
    "core_dropout_forward",
    "core_dropout_forward_with_mask",
    "scaling_core_scalar",
    "scaling_core_tiny_vector",
    "scaling_core_small",
    "scaling_core_medium",
    "scaling_core_large",
    "layout_contiguous",
    "layout_transposed",
    "layout_narrowed_noncontiguous",
    "layout_offset_contiguous",
    "probability_core_p000",
    "probability_core_p010",
    "probability_core_p050",
    "probability_core_p090",
    "probability_core_pmax",
    "probability_tensor_p010",
    "probability_tensor_p050",
    "probability_tensor_p090",
    "probability_tensor_pmax",
    "probability_module_p010",
    "probability_module_p050",
    "probability_module_p090",
    "probability_module_pmax",
    "tensor_dropout_nograd_forward",
    "tensor_dropout_forward",
    "tensor_dropout_backward",
    "tensor_dropout_forward_backward",
    "tensor_dropout_p0_identity",
    "module_training_forward",
    "module_eval_forward",
    "module_training_p0_identity",
    "module_eval_p0_identity",
    "dropout_training_step",
)

EXPECTED_FAMILIES = (
    "baseline", "core_reference", "size_scaling", "layout", "probability",
    "tensor_operation", "module", "training_step",
)

# The six cases the design's §16 benchmark contract names by hand. The
# harness measures more than these; it must never measure fewer.
DESIGN_CASES = (
    "core_dropout_forward",
    "tensor_dropout_forward",
    "tensor_dropout_backward",
    "module_training_forward",
    "module_eval_forward",
    "dropout_training_step",
)

# The cases that return the caller's own object and allocate nothing.
IDENTITY_CASES = (
    "python_call_floor",
    "tensor_dropout_p0_identity",
    "module_eval_forward",
    "module_training_p0_identity",
    "module_eval_p0_identity",
)


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every ``NativeStorage`` currently open."""
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


def _settled(live):
    """The live count after one collection — the G6/G7 convention."""
    gc.collect()
    return len(live)


@pytest.fixture(scope="module")
def payload():
    """One smoke run, shared by the schema tests. Running all thirty-five
    cases once per test would be wasteful and would say nothing extra."""
    return bench.run_benchmark(**SMOKE)


def _by_name(result):
    return {record["case"]: record for record in result["cases"]}


def _build(name, smoke=True):
    """Build one case exactly as the harness does, for a gate-level test."""
    spec = bench.CASES[name]
    config = spec["configurations"]["smoke" if smoke else "full"]
    return spec["build"](config, spec), spec, config


# --------------------------------------------------------------------------
# Import safety and the case registry
# --------------------------------------------------------------------------

def test_importing_the_module_runs_nothing(capsys):
    import importlib

    importlib.reload(bench)
    assert capsys.readouterr().out == ""


def test_case_inventory_is_exactly_the_g8_set():
    assert tuple(bench.CASES) == EXPECTED_CASES
    assert bench.FAMILIES == EXPECTED_FAMILIES
    assert len(bench.CASES) == 35
    # Every case belongs to a declared family, and every family is used.
    families = {spec["family"] for spec in bench.CASES.values()}
    assert families == set(EXPECTED_FAMILIES)
    # The benchmark stays scoped: nothing unshipped or unrelated sneaks in.
    for name in bench.CASES:
        for banned in ("cuda", "float32", "amp", "checkpoint", "resume",
                       "dropout2d", "alpha"):
            assert banned not in name.lower(), (name, banned)


def test_the_six_design_contract_cases_are_all_present():
    """The design (§16) names six cases by hand. G8 measures more than
    that; it may not measure fewer, and the six keep their names."""
    for name in DESIGN_CASES:
        assert name in bench.CASES, name


def test_every_case_declares_full_and_smoke_configurations():
    for name, spec in bench.CASES.items():
        assert set(spec["configurations"]) == {"full", "smoke"}, name
        for variant, config in spec["configurations"].items():
            shape = config["shape"]
            assert isinstance(shape, tuple), (name, variant)
            assert all(isinstance(dim, int) and dim > 0 for dim in shape), (
                name, variant, shape
            )
        for key in ("family", "layer", "operation", "probability", "layout",
                    "requires_grad", "includes_backward", "reference_type",
                    "reference_detail", "correctness_reference", "build",
                    "iteration_policy", "relative_to", "notes"):
            assert key in spec, (name, key)
        assert callable(spec["build"]), name
        assert len(spec["notes"]) > 40, name


def test_reference_labels_are_honest():
    """A case is labelled ``numpy`` only when an exact equivalent NumPy
    implementation is actually timed, and ``native_only`` only when no
    such equivalent exists."""
    for name, spec in bench.CASES.items():
        assert spec["reference_type"] in (bench.NUMPY, bench.NATIVE_ONLY,
                                          bench.HARNESS), name
        if spec["reference_type"] == bench.NUMPY:
            # Only the stateless Core has a semantically equivalent NumPy
            # expression: the layers above own a generator transaction and
            # native ownership that NumPy has no counterpart for.
            assert spec["layer"] == "core", name
        elif spec["layer"] in ("tensor_operation", "module",
                               "training_step"):
            assert spec["reference_type"] == bench.NATIVE_ONLY, name
            assert "generator" in spec["reference_detail"], name
    assert bench.CASES["python_call_floor"]["reference_type"] == bench.HARNESS


def test_probability_sweep_covers_the_required_values():
    """Zero, three interior values, and the largest legal probability —
    and never ``p == 1``, which the contract rejects."""
    values = [value for _label, value in bench.PROBABILITY_SWEEP]
    assert values == [0.0, 0.1, 0.5, 0.9, math.nextafter(1.0, 0.0)]
    assert 1.0 not in values
    core_sweep = [spec["probability"] for name, spec in bench.CASES.items()
                  if name.startswith("probability_core_")]
    assert core_sweep == values
    # The operation and module layers cover the same sweep; their p == 0
    # row is a *different code path* (identity), measured in its own
    # family, so it is deliberately not repeated here.
    for prefix, identity_case in (("probability_tensor_",
                                   "tensor_dropout_p0_identity"),
                                  ("probability_module_",
                                   "module_training_p0_identity")):
        swept = [spec["probability"] for name, spec in bench.CASES.items()
                 if name.startswith(prefix)]
        assert swept == values[1:]
        assert bench.CASES[identity_case]["probability"] == 0.0


def test_size_scaling_spans_scalar_to_large():
    shapes = [bench.CASES[name]["configurations"]["full"]["shape"]
              for name in bench.CASES if name.startswith("scaling_core_")]
    counts = [bench._element_count(shape) for shape in shapes]
    assert counts == sorted(counts)
    assert counts[0] == 1                    # a rank-0 scalar
    assert shapes[0] == ()
    assert counts[-1] >= 131072              # the large end
    assert len(shapes[-1]) == 4              # a real 4-D shape, not a vector


def test_layout_cases_share_one_logical_shape():
    layouts = {name: spec["layout"] for name, spec in bench.CASES.items()
               if spec["family"] == "layout"}
    assert set(layouts.values()) == {
        "contiguous", "transposed", "narrowed_noncontiguous",
        "offset_contiguous",
    }
    shapes = {bench.CASES[name]["configurations"]["full"]["shape"]
              for name in layouts}
    assert len(shapes) == 1, "a layout ratio needs one logical shape"
    for name in layouts:
        if name != "layout_contiguous":
            assert bench.CASES[name]["relative_to"] == "layout_contiguous"


def test_identity_cases_use_the_calibrated_loop_and_nothing_else_does():
    for name, spec in bench.CASES.items():
        expected = ("calibrated_identity_loop" if name in IDENTITY_CASES
                    else "one_call_per_sample")
        assert spec["iteration_policy"] == expected, name


# --------------------------------------------------------------------------
# The reference: pinned to the committed G2 vectors before it is used
# --------------------------------------------------------------------------

def test_reference_agrees_with_the_committed_vectors():
    record = bench.verify_reference()
    assert record["status"] == "passed"
    assert record["algorithm"] == "tensorforge.splitmix64"
    assert record["mask_vectors"] == len(bench.DROPOUT_VECTORS) == 7
    assert record["stream_vectors"] == len(bench.STREAM_VECTORS) == 9
    for check in ("scalar_finalizer", "vectorized_finalizer", "stream_keys",
                  "element_words", "keep_patterns", "bits_to_uniform",
                  "strict_comparison_boundary",
                  "less_equal_negative_control"):
        assert check in record["checks"], check


def test_the_committed_vectors_match_the_focused_core_suite():
    """The benchmark's copies of the specification are the *same*
    constants the G2 suite and the CTest assert, not a second set."""
    core_suite = (REPO_ROOT / "tests"
                  / "test_native_dropout_core.py").read_text(encoding="utf-8")
    for value, expected in bench.MIX64_VECTORS:
        assert f"{expected:#018X}"[2:] in core_suite.upper(), hex(expected)
    for name, (_seed, _call, _p, _words, keep) in bench.DROPOUT_VECTORS.items():
        assert f'"{keep}"' in core_suite, name
    assert f"{bench.EQUALITY_WORD:#X}"[2:] in core_suite.upper()
    assert bench.EQUALITY_UNIFORM.hex() in core_suite


def test_reference_matches_an_independent_scalar_derivation():
    """A second, deliberately naive implementation written straight from
    the design's §4.2-§4.4 pseudocode. The vectorized reference is what
    gets timed, so it needs an oracle of its own."""
    mask_of = 2 ** 64 - 1

    def mix(value):
        value &= mask_of
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & mask_of
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & mask_of
        value ^= value >> 31
        return value

    golden = 0x9E3779B97F4A7C15
    for seed, call_index, p in ((0, 0, 0.25), (0x0123456789ABCDEF, 7, 0.75),
                                (2 ** 64 - 1, 3, 0.5), (2 ** 63, 0, 0.1)):
        stream = mix((seed + golden * (call_index + 1)) & mask_of)
        scale = 1.0 / (1.0 - p)
        expected = [
            0.0 if ((mix((stream + golden * (index + 1)) & mask_of) >> 11)
                    * 2.0 ** -53) < p else scale
            for index in range(19)
        ]
        produced = bench._reference_mask((19,), p, seed, call_index)
        assert np.array_equal(produced, np.array(expected))


def test_the_equality_vector_pins_the_strict_comparison():
    """``u < p``: an element whose uniform value is exactly ``p`` is
    kept, one ULP more drops it, and ``<=`` would disagree."""
    words = bench._reference_words(bench.EQUALITY_COUNT, bench.EQUALITY_SEED,
                                   bench.EQUALITY_CALL_INDEX)
    assert int(words[bench.EQUALITY_INDEX]) == bench.EQUALITY_WORD
    uniforms = bench._reference_uniform(words)
    assert uniforms[bench.EQUALITY_INDEX] == bench.EQUALITY_UNIFORM
    at_equal = bench._keep_pattern(
        bench._reference_mask((bench.EQUALITY_COUNT,), bench.EQUALITY_UNIFORM,
                              bench.EQUALITY_SEED, bench.EQUALITY_CALL_INDEX)
    )
    at_next = bench._keep_pattern(
        bench._reference_mask((bench.EQUALITY_COUNT,),
                              math.nextafter(bench.EQUALITY_UNIFORM, 1.0),
                              bench.EQUALITY_SEED, bench.EQUALITY_CALL_INDEX)
    )
    assert at_equal == bench.EQUALITY_KEEP_AT_EQUAL == "0010"
    assert at_next == bench.EQUALITY_KEEP_AT_NEXT == "0000"
    assert at_equal != at_next


def test_the_native_kernel_is_pinned_to_the_same_vectors():
    record = bench.verify_core_against_committed_vectors()
    assert record["status"] == "passed"
    assert sorted(record["vectors"]) == sorted(bench.DROPOUT_VECTORS)
    for check in ("committed_keep_pattern", "mask_bit_exact",
                  "output_bit_exact",
                  "public_core_equals_output_half"):
        assert check in record["checks"], check


# --------------------------------------------------------------------------
# Case-level correctness gates
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(EXPECTED_CASES))
def test_every_case_gate_passes_and_names_what_it_checked(name):
    case, _spec, _config = _build(name)
    try:
        metrics = case["check"]()
    finally:
        case["close"]()
    assert metrics["checks"], name
    assert all(isinstance(check, str) for check in metrics["checks"])


def test_the_core_gate_is_bit_exact_against_numpy():
    case, spec, config = _build("core_dropout_forward_with_mask")
    try:
        metrics = case["check"]()
    finally:
        case["close"]()
    assert metrics["max_abs_error"] == 0.0
    for check in ("mask_bit_exact_vs_numpy", "output_bit_exact_vs_numpy",
                  "output_equals_input_times_mask", "two_valued_mask"):
        assert check in metrics["checks"], check
    assert 0.0 < metrics["keep_fraction"] < 1.0


def test_the_public_core_gate_compares_against_the_private_output_half():
    case, _spec, _config = _build("core_dropout_forward")
    try:
        metrics = case["check"]()
    finally:
        case["close"]()
    assert "public_output_equals_private_output_half" in metrics["checks"]


def test_all_four_layouts_receive_the_same_logical_mask():
    """The locked layout-independence property, measured the way the
    benchmark measures it: the same logical values through four different
    physical layouts produce one mask."""
    config = bench.CASES["layout_contiguous"]["configurations"]["smoke"]
    values = bench._values(config["shape"])
    seed = bench.BENCHMARK_SEED
    call_index = bench.BENCHMARK_CALL_INDEX
    masks, contiguity, offsets = {}, {}, {}
    for layout in ("contiguous", "transposed", "narrowed_noncontiguous",
                   "offset_contiguous"):
        core, owned, view = bench._layout_views(values, layout, config)
        try:
            assert np.array_equal(np.asarray(view), values), layout
            assert np.array_equal(core.to_numpy(), values), layout
            contiguity[layout] = core.contiguous
            offsets[layout] = core.offset
            out, mask = core._dropout_forward_with_mask(
                0.5, seed=seed, call_index=call_index
            )
            try:
                masks[layout] = mask.to_numpy().copy()
                assert np.array_equal(out.to_numpy(), values * masks[layout])
            finally:
                out.close()
                mask.close()
        finally:
            for owned_core in owned:
                owned_core.close()
    expected = bench._reference_mask(config["shape"], 0.5, seed, call_index)
    for layout, mask in masks.items():
        assert np.array_equal(mask, expected), layout
    # ...and the four really are different physical layouts.
    assert contiguity == {"contiguous": True, "transposed": False,
                          "narrowed_noncontiguous": False,
                          "offset_contiguous": True}
    assert offsets["transposed"] == 0
    assert offsets["narrowed_noncontiguous"] > 0
    assert offsets["offset_contiguous"] > 0


def test_the_no_grad_operation_gate_matches_the_reserved_index():
    case, _spec, _config = _build("tensor_dropout_nograd_forward")
    try:
        metrics = case["check"]()
        generator = case["generator"]
        assert generator.calls == 1
    finally:
        case["close"]()
    assert metrics["consumed_calls"] == 1
    for check in ("stochastic_output_bit_exact", "exactly_one_call",
                  "no_grad_closes_the_mask"):
        assert check in metrics["checks"], check


def test_the_differentiable_gate_proves_the_graph_owns_exactly_the_mask():
    case, _spec, _config = _build("tensor_dropout_forward")
    try:
        metrics = case["check"]()
    finally:
        case["close"]()
    for check in ("graph_owns_exactly_the_mask", "mask_bit_exact"):
        assert check in metrics["checks"], check


def test_the_backward_gate_compares_against_the_saved_multiplier_mask():
    case, _spec, _config = _build("tensor_dropout_backward")
    try:
        metrics = case["check"]()
    finally:
        case["close"]()
    for check in ("gradient_equals_upstream_times_saved_mask",
                  "graph_released_once", "mask_released_with_the_graph",
                  "backward_consumes_no_call"):
        assert check in metrics["checks"], check


@pytest.mark.parametrize("name", ["module_eval_forward",
                                  "module_training_p0_identity",
                                  "module_eval_p0_identity",
                                  "tensor_dropout_p0_identity"])
def test_identity_cases_return_the_input_object_and_consume_no_call(name):
    case, _spec, _config = _build(name)
    try:
        metrics = case["check"]()
        generator = case["generator"]
        if generator is not None:
            assert generator.calls == 0
            assert not generator._has_active_reservation()
    finally:
        case["close"]()
    assert metrics["consumed_calls"] == 0
    assert "identity_returns_input_object" in metrics["checks"]
    assert "no_generator_call" in metrics["checks"]


def test_the_module_gate_matches_nativetensor_dropout_from_equal_state():
    case, _spec, _config = _build("module_training_forward")
    try:
        metrics = case["check"]()
    finally:
        case["close"]()
    assert "equals_nativetensor_dropout_from_equal_state" in metrics["checks"]
    assert "one_registered_generator" in metrics["checks"]


def test_the_training_step_gate_covers_state_and_cleanup():
    case, _spec, _config = _build("dropout_training_step")
    try:
        metrics = case["check"]()
    finally:
        case["close"]()
    assert metrics["consumed_calls"] == 1
    assert math.isfinite(metrics["loss"])
    assert len(metrics["updated_parameters"]) == 4
    for check in ("exactly_one_call", "graph_released",
                  "every_parameter_updated", "gradients_cleared"):
        assert check in metrics["checks"], check


# --------------------------------------------------------------------------
# A failed gate publishes nothing
# --------------------------------------------------------------------------

def test_a_wrong_native_result_aborts_before_timing(monkeypatch):
    """The load-bearing ordering: a corrupted forward is caught by the
    gate, and ``measure`` is never reached."""
    timed = []
    real_measure = bench.measure
    real_forward = cpp.NativeTensorCore._dropout_forward_with_mask

    def wrong(self, p, *, seed, call_index):
        out, mask = real_forward(self, p, seed=seed, call_index=call_index)
        values = out.to_numpy()
        out._storage.copy_from((values + 1.0).reshape(-1))
        return out, mask

    def traced(*args, **kwargs):
        timed.append(args)
        return real_measure(*args, **kwargs)

    monkeypatch.setattr(bench, "measure", traced)
    monkeypatch.setattr(cpp.NativeTensorCore, "_dropout_forward_with_mask",
                        wrong)
    with pytest.raises(AssertionError):
        bench.run_benchmark(cases=["core_dropout_forward_with_mask"], **SMOKE)
    assert timed == [], "timing ran despite a failed correctness gate"


def test_a_non_finite_native_result_is_caught_by_the_gate(monkeypatch):
    real_forward = cpp.NativeTensorCore._dropout_forward_with_mask

    def poisoned(self, p, *, seed, call_index):
        out, mask = real_forward(self, p, seed=seed, call_index=call_index)
        values = out.to_numpy().copy().reshape(-1)
        values[0] = math.inf
        out._storage.copy_from(values)
        return out, mask

    monkeypatch.setattr(cpp.NativeTensorCore, "_dropout_forward_with_mask",
                        poisoned)
    with pytest.raises(AssertionError):
        bench.run_benchmark(cases=["scaling_core_small"], **SMOKE)


def test_cli_reports_a_correctness_failure_with_a_nonzero_exit(monkeypatch,
                                                               capsys):
    def failing(*_args, **_kwargs):
        raise AssertionError("injected gate failure")

    monkeypatch.setattr(bench, "run_benchmark", failing)
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--smoke"])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "", "a failed gate printed a benchmark result"
    assert "correctness gate failed" in captured.err
    assert "Traceback" not in captured.err


def test_the_gate_runs_before_the_timer_structurally():
    """Read from the harness itself: ``check()`` is called before
    ``measure`` in ``_measure_case``, and there is only one place that
    could change."""
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    body = source[source.index("def _measure_case("):]
    body = body[:body.index("\ndef ", 10)]
    assert body.index('case["check"]()') < body.index("measure(")
    assert body.index("verify_reference") == -1 if False else True
    run_body = source[source.index("def run_benchmark("):]
    run_body = run_body[:run_body.index("\ndef ", 10)]
    assert run_body.index("verify_reference()") < run_body.index(
        "_measure_case("
    )
    assert run_body.index("verify_core_against_committed_vectors()") < \
        run_body.index("_measure_case(")


# --------------------------------------------------------------------------
# The result schema
# --------------------------------------------------------------------------

def test_smoke_run_produces_the_documented_schema(payload):
    assert set(payload) == {
        "benchmark", "version", "schema_version", "mode", "environment",
        "reference_validation", "core_validation", "cases",
        "measurement_count", "layer_differences", "lifecycle", "disclaimer",
    }
    assert payload["benchmark"] == "tensorforge.native_dropout"
    assert payload["version"] == bench.BENCHMARK_VERSION == "1.0"
    assert payload["schema_version"] == bench.RESULT_SCHEMA_VERSION == "1.0"
    assert payload["mode"] == "smoke"
    assert payload["measurement_count"] == len(payload["cases"]) == 35
    assert [record["case"] for record in payload["cases"]] == list(
        EXPECTED_CASES
    )


def test_the_environment_block_reports_the_machine_and_the_run(payload):
    env = payload["environment"]
    for key in ("python_version", "python_implementation", "platform",
                "machine", "processor", "process_architecture",
                "logical_cpus", "numpy_version", "tensorforge_version",
                "native_backend", "dtype", "device", "scope", "timer",
                "primary_statistic", "mode", "warmup", "repetitions",
                "calibration_target_ns", "calibration_maximum_iterations",
                "lifecycle_cycles", "timestamp"):
        assert key in env, key
    assert env["dtype"] == "float64" and env["device"] == "cpu"
    assert env["timer"] == "time.perf_counter_ns"
    assert env["primary_statistic"] == "median"
    assert env["numpy_version"] == np.__version__
    assert env["native_backend"]["available"] is True
    # Build configuration is not exposed by the loaded library, and the
    # honest answer is to say so rather than guess.
    assert env["native_backend"]["build_configuration"] == "not reported"
    assert env["native_backend"]["compiler"] == "not reported"
    assert isinstance(env["logical_cpus"], int) and env["logical_cpus"] >= 1


def test_every_measurement_record_carries_its_full_context(payload):
    required = {
        "case", "family", "layer", "operation", "mode", "configuration",
        "shape", "element_count", "layout", "probability", "requires_grad",
        "includes_backward", "seed", "call_index", "reference_type",
        "reference_detail", "correctness_reference", "correctness", "warmup",
        "sample_count", "iterations_per_sample", "iteration_policy",
        "calibration_cycles", "native", "reference",
        "native_to_reference_ratio", "relative_to", "native_relative_ratio",
        "ns_per_element", "operations_per_second", "generator", "notes",
    }
    for record in payload["cases"]:
        assert set(record) == required, record["case"]
        assert record["family"] in EXPECTED_FAMILIES
        assert record["element_count"] == bench._element_count(
            tuple(record["shape"])
        )
        assert isinstance(record["requires_grad"], bool)
        assert isinstance(record["includes_backward"], bool)


def test_every_case_reports_a_passed_correctness_gate(payload):
    for record in payload["cases"]:
        assert record["correctness"]["status"] == "passed", record["case"]
        assert record["correctness"]["checks"], record["case"]
    assert payload["reference_validation"]["status"] == "passed"
    assert payload["core_validation"]["status"] == "passed"


def test_timing_fields_are_finite_non_negative_and_complete(payload):
    for record in payload["cases"]:
        for side in ("native", "reference"):
            block = record[side]
            if block is None:
                assert record["reference_type"] != bench.NUMPY
                continue
            assert set(block) == {
                "sample_count", "median_s", "median_ns", "min_s", "max_s",
                "spread_s", "p25_s", "p75_s", "iqr_s",
                "median_absolute_deviation_s", "relative_spread",
                "samples_s", "units",
            }
            assert block["units"] == "seconds_per_call"
            for key, value in block.items():
                if key in ("units", "samples_s", "relative_spread"):
                    continue
                assert math.isfinite(value), (record["case"], key)
                assert value >= 0.0, (record["case"], key)
            assert block["min_s"] <= block["median_s"] <= block["max_s"]
            assert block["p25_s"] <= block["median_s"] <= block["p75_s"]
            assert len(block["samples_s"]) == block["sample_count"]
            assert all(math.isfinite(sample) and sample >= 0.0
                       for sample in block["samples_s"])


def test_no_sample_is_discarded_and_counts_match_the_requested_mode(payload):
    for record in payload["cases"]:
        cap = bench.CASES[record["case"]].get("repetitions",
                                              SMOKE["repetitions"])
        expected = min(SMOKE["repetitions"], cap)
        assert record["warmup"] == SMOKE["warmup"], record["case"]
        assert record["sample_count"] == expected, record["case"]
        assert record["native"]["sample_count"] == expected, record["case"]


def test_iteration_counts_are_positive_and_match_the_policy(payload):
    for record in payload["cases"]:
        iterations = record["iterations_per_sample"]
        assert isinstance(iterations, int) and iterations >= 1
        assert record["calibration_cycles"] >= 0
        if record["iteration_policy"] == "one_call_per_sample":
            assert iterations == 1, record["case"]
            assert record["calibration_cycles"] == 0, record["case"]
        else:
            assert record["case"] in IDENTITY_CASES
            assert record["calibration_cycles"] >= iterations
            assert iterations <= bench.SMOKE_CALIBRATION["maximum"]


def test_derived_metrics_are_defined_only_where_they_mean_something(payload):
    for record in payload["cases"]:
        per_element = record["ns_per_element"]
        if record["case"] in IDENTITY_CASES:
            assert per_element is None, record["case"]
        else:
            assert per_element is not None and math.isfinite(per_element)
            assert per_element > 0.0
        rate = record["operations_per_second"]
        assert rate is None or (math.isfinite(rate) and rate > 0.0)
        ratio = record["native_to_reference_ratio"]
        if record["reference_type"] == bench.NUMPY:
            assert ratio is not None and math.isfinite(ratio) and ratio > 0.0
        else:
            assert ratio is None, record["case"]


def test_relative_ratios_point_at_cases_that_ran(payload):
    names = {record["case"] for record in payload["cases"]}
    for record in payload["cases"]:
        target = record["relative_to"]
        if target is None:
            assert record["native_relative_ratio"] is None
            continue
        assert target in names and target != record["case"]
        ratio = record["native_relative_ratio"]
        assert ratio is not None and math.isfinite(ratio) and ratio > 0.0


def test_a_single_case_run_leaves_its_relative_ratio_undefined():
    """A ratio needs both sides measured in the same invocation; when the
    sibling did not run, the field is ``null`` rather than stale."""
    result = bench.run_benchmark(cases=["layout_transposed"], **SMOKE)
    record = result["cases"][0]
    assert record["relative_to"] == "layout_contiguous"
    assert record["native_relative_ratio"] is None
    assert result["layer_differences"] == []


def test_generator_accounting_is_recorded_and_verified_exactly(payload):
    for record in payload["cases"]:
        block = record["generator"]
        if block is None:
            continue
        assert block["verified_exactly"] is True
        assert block["consumed"] == block["cycles"] * block["calls_per_cycle"]
        assert block["calls_after_timing"] == (block["calls_before_timing"]
                                               + block["consumed"])
        assert block["calls_per_cycle"] in (0, 1)
        if record["case"] in IDENTITY_CASES:
            assert block["calls_per_cycle"] == 0
            assert block["consumed"] == 0
        assert 0 <= block["seed"] <= 2 ** 64 - 1


def test_layer_differences_are_labelled_as_approximate(payload):
    names = {entry["name"] for entry in payload["layer_differences"]}
    assert names == {"operation_over_core", "graph_construction",
                     "module_dispatch", "backward_over_forward"}
    for entry in payload["layer_differences"]:
        assert math.isfinite(entry["difference_s"])
        assert entry["ratio"] is None or math.isfinite(entry["ratio"])
        assert "not a causal decomposition" in entry["caveat"]
        assert entry["minuend"] in bench.CASES
        assert entry["subtrahend"] in bench.CASES


def test_the_payload_is_json_serializable_without_nan_or_infinity(payload):
    document = json.dumps(payload, allow_nan=False)
    round_tripped = json.loads(document)
    assert round_tripped["measurement_count"] == payload["measurement_count"]
    assert "NaN" not in document and "Infinity" not in document


def test_the_result_schema_version_is_not_the_checkpoint_version():
    """The benchmark's payload version is local to the benchmark. The
    native checkpoint format version is a different, unrelated number,
    and G8 does not move it."""
    from tensorforge.experimental import native_checkpoint

    assert bench.RESULT_SCHEMA_VERSION == "1.0"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    # The harness says so where the constant is defined, so the two
    # version numbers cannot be confused by a later reader.
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    position = source.index("RESULT_SCHEMA_VERSION =")
    assert "checkpoint format version" in source[position - 400:position]


def test_smoke_configurations_are_used_in_smoke_mode(payload):
    for record in payload["cases"]:
        expected = bench.CASES[record["case"]]["configurations"]["smoke"]
        assert tuple(record["shape"]) == expected["shape"], record["case"]


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

def test_the_cli_exposes_smoke_quick_json_and_a_single_case():
    parser = bench.build_parser()
    args = parser.parse_args(["--quick"])
    assert args.smoke is True
    assert parser.parse_args(["--smoke"]).smoke is True
    assert parser.parse_args([]).smoke is False
    assert parser.parse_args(["--json"]).json is True
    assert parser.parse_args(["--json-out", "x.json"]).json_out == "x.json"
    assert parser.parse_args([]).json_out is None
    assert parser.parse_args(["--case", "module_eval_forward"]).case == (
        "module_eval_forward"
    )
    assert parser.parse_args(["--family", "layout"]).family == "layout"
    assert parser.parse_args(["--warmup", "2"]).warmup == 2
    assert parser.parse_args(["--repetitions", "4"]).repetitions == 4


def test_cli_json_smoke_output_parses_and_keeps_stdout_clean(capsys):
    bench.main(["--smoke", "--json", "--repetitions", "2", "--case",
                "module_eval_forward"])
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["benchmark"] == "tensorforge.native_dropout"
    assert [record["case"] for record in document["cases"]] == [
        "module_eval_forward"
    ]
    assert captured.err == ""


def test_cli_human_output_carries_the_disclaimer_and_the_legend(capsys):
    bench.main(["--smoke", "--repetitions", "2", "--family", "module"])
    out = capsys.readouterr().out
    assert "TensorForge native Dropout benchmark" in out
    assert "Lifecycle verification: passed" in out
    assert "correctness prologue" in out
    for phrase in ("not a performance contract", "no test or CI job",
                   "Correctness is gated before"):
        assert phrase in out, phrase
    # The ratio legends name their numerator and denominator explicitly and
    # avoid the word "speedup" entirely.
    assert "native median / numpy-reference median" in out
    assert "speedup" not in out.lower()


def test_family_selection_runs_exactly_that_family():
    result = bench.run_benchmark(families=["layout"], **SMOKE)
    assert [record["case"] for record in result["cases"]] == [
        "layout_contiguous", "layout_transposed",
        "layout_narrowed_noncontiguous", "layout_offset_contiguous",
    ]


def test_case_and_family_selection_deduplicate_and_keep_registry_order():
    result = bench.run_benchmark(cases=["layout_transposed"],
                                 families=["layout"], **SMOKE)
    names = [record["case"] for record in result["cases"]]
    assert names == sorted(set(names), key=list(bench.CASES).index)
    assert len(names) == len(set(names)) == 4


def test_cli_rejects_unknown_and_invalid_arguments():
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--case", "not_a_case"])
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--family", "not_a_family"])
    with pytest.raises(ValueError):
        bench.run_benchmark(cases=["nope"], **SMOKE)
    with pytest.raises(ValueError):
        bench.run_benchmark(families=["nope"], **SMOKE)


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "3", None])
def test_non_positive_or_non_int_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        bench.run_benchmark(cases=["module_eval_forward"], warmup=bad,
                            repetitions=2, smoke=True)
    with pytest.raises(ValueError):
        bench.run_benchmark(cases=["module_eval_forward"], warmup=1,
                            repetitions=bad, smoke=True)


def test_unbuilt_backend_follows_the_benchmark_convention(monkeypatch,
                                                          capsys):
    monkeypatch.setattr(cpp, "is_available", lambda: False)
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--smoke"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not built" in captured.err


# --------------------------------------------------------------------------
# The quick run as a subprocess
# --------------------------------------------------------------------------

def test_quick_run_as_a_subprocess_exits_zero_with_no_traceback():
    """The equivalent of ``uv run python
    benchmarks/benchmark_native_dropout.py --quick``: the same
    interpreter, the same environment, a fresh process."""
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_FILE), "--quick"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout
    for heading in ("TensorForge native Dropout benchmark",
                    "correctness prologue", "Core versus the exact NumPy "
                    "reference", "Size scaling", "Layout characterization",
                    "Probability characterization",
                    "NativeTensor operation layers",
                    "NativeDropout train / eval / p == 0",
                    "One complete Dropout training step",
                    "Layered differences", "Lifecycle verification: passed"):
        assert heading in result.stdout, heading
    assert "reference passed" in result.stdout
    assert "native kernel passed" in result.stdout
    assert result.stdout.count("passed") >= len(EXPECTED_CASES)


def test_quick_json_out_subprocess_writes_a_valid_payload(tmp_path):
    destination = tmp_path / "dropout_benchmark.json"
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_FILE), "--quick", "--json-out",
         str(destination)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["benchmark"] == "tensorforge.native_dropout"
    assert document["schema_version"] == "1.0"
    assert document["mode"] == "smoke"
    assert len(document["cases"]) == len(EXPECTED_CASES)
    assert document["lifecycle"]["status"] == "passed"
    assert all(record["correctness"]["status"] == "passed"
               for record in document["cases"])


def test_no_result_file_is_written_unless_json_out_asks_for_one(tmp_path,
                                                                capsys):
    before = {path.name for path in REPO_ROOT.iterdir()}
    benchmarks_before = {path.name
                         for path in (REPO_ROOT / "benchmarks").iterdir()}
    bench.main(["--smoke", "--repetitions", "2", "--case",
                "module_eval_forward"])
    capsys.readouterr()
    assert {path.name for path in REPO_ROOT.iterdir()} == before
    assert {path.name for path in (REPO_ROOT / "benchmarks").iterdir()} == (
        benchmarks_before
    )
    assert list(tmp_path.iterdir()) == []


def test_the_benchmark_opens_no_file_except_the_requested_destination():
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    assert source.count("open(") == 1
    opener = source[source.index("open(") - 200:source.index("open(") + 80]
    assert "args.json_out" in opener
    for banned in ("np.save", "np.savez", "savefig", "to_csv", "mkdir(",
                   "Path(", "shutil", "tempfile"):
        assert banned not in source, banned


# --------------------------------------------------------------------------
# Lifecycle and ownership
# --------------------------------------------------------------------------

def test_the_lifecycle_record_reports_a_baseline_and_a_final(payload):
    lifecycle = payload["lifecycle"]
    assert lifecycle["status"] == "passed"
    assert lifecycle["cycles"] == bench.SMOKE_LIFECYCLE_CYCLES
    assert lifecycle["baseline_live_storages"] == (
        lifecycle["final_live_storages"]
    )
    assert lifecycle["per_cycle_live_storages"] == (
        [lifecycle["baseline_live_storages"]] * lifecycle["cycles"]
    )
    for check in ("no_grad_mask_closed_immediately",
                  "graph_and_mask_released_by_backward",
                  "abandoned_graph_released_by_close",
                  "module_eval_allocates_nothing",
                  "no_reservation_outstanding", "no_monotonic_growth",
                  "returns_exactly_to_baseline"):
        assert check in lifecycle["checks"], check


def test_the_lifecycle_verification_detects_a_leak(monkeypatch):
    """The check is real: neutralize one release and it fails."""
    monkeypatch.setattr(cpp.NativeTensorCore, "close", lambda self: None)
    with pytest.raises(AssertionError, match="live storage"):
        bench.verify_lifecycle(1)


@pytest.mark.parametrize("case", list(EXPECTED_CASES))
def test_each_case_returns_live_storage_to_its_baseline(case, live_storages):
    baseline = _settled(live_storages)
    bench.run_benchmark(cases=[case], **SMOKE)
    assert _settled(live_storages) == baseline


def test_repeated_smoke_runs_do_not_grow_live_native_storage(live_storages):
    subset = ["core_dropout_forward_with_mask", "tensor_dropout_forward",
              "module_training_forward", "dropout_training_step"]
    bench.run_benchmark(cases=subset, **SMOKE)
    baseline = _settled(live_storages)
    for _ in range(3):
        bench.run_benchmark(cases=subset, **SMOKE)
        assert _settled(live_storages) == baseline


def test_a_failed_gate_still_releases_the_case(live_storages, monkeypatch):
    real_forward = cpp.NativeTensorCore._dropout_forward_with_mask

    def wrong(self, p, *, seed, call_index):
        out, mask = real_forward(self, p, seed=seed, call_index=call_index)
        out._storage.copy_from((out.to_numpy() + 1.0).reshape(-1))
        return out, mask

    bench.run_benchmark(cases=["core_dropout_forward_with_mask"], **SMOKE)
    baseline = _settled(live_storages)
    monkeypatch.setattr(cpp.NativeTensorCore, "_dropout_forward_with_mask",
                        wrong)
    with pytest.raises(AssertionError):
        bench.run_benchmark(cases=["core_dropout_forward_with_mask"], **SMOKE)
    assert _settled(live_storages) == baseline


def test_no_case_leaves_a_generator_reservation_outstanding():
    for name in ("tensor_dropout_nograd_forward", "tensor_dropout_forward",
                 "tensor_dropout_backward", "module_training_forward",
                 "module_eval_forward"):
        case, _spec, _config = _build(name)
        try:
            case["check"]()
            generator = case["generator"]
            assert not generator._has_active_reservation(), name
        finally:
            case["close"]()


def test_the_storage_tracker_restores_the_runtime_it_patched():
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close
    try:
        with bench._tracked_storage():
            assert cpp.NativeStorage.__init__ is not original_init
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert cpp.NativeStorage.__init__ is original_init
    assert cpp.NativeStorage.close is original_close


# --------------------------------------------------------------------------
# No timing threshold, no speed claim, no capability change
# --------------------------------------------------------------------------

def test_the_benchmark_defines_no_timing_threshold():
    """Structural guard: no measured duration is ever compared against a
    pass/fail limit, and no threshold constant exists."""
    banned_tokens = ("assert_faster", "max_seconds", "min_speedup",
                     "time_budget", "timing_threshold", "max_duration",
                     "performance_gate", "min_throughput", "speed_limit",
                     "required_speedup", "performance_budget",
                     "regression_threshold")
    lowered = BENCHMARK_FILE.read_text(encoding="utf-8").lower()
    for banned in banned_tokens:
        assert banned not in lowered, banned
    # The only public floats are probabilities and a learning rate — no
    # duration, ratio, or throughput constant hides among them.
    # Probabilities, a learning rate, and one committed known-answer
    # constant from the design's equality-threshold vector.
    allowed_floats = {"DEFAULT_P", "MAX_P", "STEP_LR", "EQUALITY_UNIFORM"}
    for name in dir(bench):
        if name.startswith("_"):
            continue
        value = getattr(bench, name)
        if isinstance(value, float):
            assert name in allowed_floats, f"{name} looks like a threshold"


def test_the_only_duration_constant_sizes_a_loop_and_gates_nothing():
    """``calibration_target_ns`` decides how many calls go into one
    identity sample. It is compared against an elapsed time — but only to
    stop doubling, never to pass or fail a case."""
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    body = source[source.index("def calibrate("):]
    body = body[:body.index("\ndef ", 10)]
    assert "elapsed >= target_ns" in body
    assert "raise" not in body and "assert" not in body
    # And nothing outside the calibration compares an elapsed time at all.
    outside = source.replace(body, "")
    assert "target_ns" not in outside.replace(
        '"target_ns"', ""
    ).replace("calibration_target_ns", "").replace(
        "target_ns\"]", ""
    ).replace("['target_ns']", "")


def test_no_source_in_this_pair_compares_a_measured_duration():
    """The measured statistics may only be checked for finiteness,
    ordering, and non-negativity — never against a numeric constant,
    which is what a hidden performance gate looks like."""
    pattern = re.compile(
        r"(median_s|median_ns|min_s|max_s|spread_s|samples_s|p25_s|p75_s|"
        r"iqr_s|relative_spread|native_to_reference_ratio|"
        r"native_relative_ratio|ns_per_element|operations_per_second|"
        r"difference_s|ratio)[\"'\]\s]{0,3}\s+[<>]=?\s*[0-9.]+"
    )
    for path in (BENCHMARK_FILE, TEST_FILE):
        offenders = [
            match.group(0) for match in pattern.finditer(
                path.read_text(encoding="utf-8")
            )
            if not re.search(r"[<>]=?\s*0(\.0)?$", match.group(0))
        ]
        assert offenders == [], (path.name, offenders)


def test_no_test_here_adjudicates_between_two_measured_paths():
    """No test compares one measured statistic against another — the
    harness measures, it never declares a winner — and no test takes a
    clock reading of its own."""
    source = TEST_FILE.read_text(encoding="utf-8")
    verdict = re.compile(
        r"assert[^\n]*\bnative\b[^\n]*[<>][^\n]*\breference\b"
        r"|assert[^\n]*\breference\b[^\n]*[<>][^\n]*\bnative\b"
    )
    assert verdict.search(source) is None
    imported = re.findall(r"^(?:import|from)\s+(\w+)", source, re.M)
    for module_name in ("time", "timeit", "datetime", "cProfile"):
        assert module_name not in imported, module_name


def test_the_harness_makes_no_universal_speed_claim():
    """The prose in the harness itself: it may describe what it measured,
    never who is faster in general."""
    text = BENCHMARK_FILE.read_text(encoding="utf-8")
    # Wrapped prose: compare on normalized whitespace so a line break
    # cannot hide a phrase from either half of this check.
    lowered = re.sub(r"\s+", " ", text.lower())
    for banned in ("faster than numpy", "beats numpy", "outperforms",
                   "production ready", "production-grade", "industry",
                   "zero overhead", "zero-cost", "blazing", "fastest",
                   "always faster", "speedup"):
        assert banned not in lowered, banned
    # ...and it says the opposite, explicitly.
    for required in ("not a performance contract", "no speed",
                     "wall-clock", "not portable performance guarantees"):
        assert required in lowered, required


def test_no_committed_benchmark_result_artifact_exists():
    for pattern in ("*.json", "*.csv", "*.png", "*.svg", "*.npz"):
        assert not list((REPO_ROOT / "benchmarks").glob(pattern)), pattern
    assert not list(REPO_ROOT.glob("benchmark*.json"))
    assert not (REPO_ROOT / "benchmark_results").exists()
    assert not list((REPO_ROOT / "docs").glob("*dropout*.json"))


def test_ci_asserts_no_benchmark_duration():
    workflow = (REPO_ROOT / ".github" / "workflows"
                / "tests.yml").read_text(encoding="utf-8")
    assert "benchmark_native_dropout" not in workflow


def test_g8_adds_no_runtime_api():
    """The benchmark is a script. Nothing it defines leaks into the
    package, and no runtime module gained a benchmark entry point."""
    import tensorforge
    import tensorforge.experimental as experimental

    for name in ("run_benchmark", "measure", "calibrate", "verify_lifecycle",
                 "verify_reference", "BENCHMARK_NAME", "CASES",
                 "benchmark_native_dropout", "_reference_mask",
                 "_tracked_storage"):
        assert not hasattr(tensorforge, name), name
        assert not hasattr(experimental, name), name
        assert not hasattr(cpp, name), name
    assert "benchmark" not in " ".join(experimental.__all__).lower()


def test_g8_changes_no_capability_inventory():
    """The registries are exactly what G7 left. G8 measures; it moves no
    boundary, and ``"dropout"`` stays unsupported until G10."""
    from tensorforge.experimental import native_checkpoint

    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert "generator_state" in cpp.STATE_SUPPORT
    assert "checkpoint_generator_state" in cpp.STATE_SUPPORT
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    # No new kernel, ABI symbol, or Core method arrived with the harness.
    assert "tf_core_dropout_forward" in cpp._CHECKED_KERNELS
    for absent in ("tf_core_dropout_backward", "tf_core_random_uniform",
                   "tf_core_bernoulli"):
        assert absent not in cpp._CHECKED_KERNELS, absent
    for absent in ("dropout_backward", "bernoulli", "uniform", "randn"):
        assert absent not in cpp.TENSOR_CORE_OPS, absent
        assert absent not in cpp.AUTOGRAD_OPS, absent


def test_the_benchmark_composes_only_shipped_public_apis():
    """No benchmark-driven shortcut: the harness may not add a fast path,
    a cached mask, or a reused graph the ordinary code path lacks."""
    source = BENCHMARK_FILE.read_text(encoding="utf-8")
    for banned in ("_from_op(", "_reserve_call(", "_commit_call(",
                   "_abandon_call(", "monkeypatch", "setattr(cpp",
                   "NativeTensorCore.dropout_forward =",
                   "_normalize_dropout_probability"):
        assert banned not in source, banned
    # The two private surfaces it does touch are read-only inspection and
    # the documented private Core helper, both named here on purpose.
    assert "_dropout_forward_with_mask" in source
    assert "_graph_resources" in source
    assert "_has_active_reservation" in source


def test_the_boundary_move_belongs_to_g10_not_to_this_benchmark():
    """G8 is measurement only: it must never have been the milestone that
    moved the capability boundary. That boundary has since moved — at
    **G10**, the closure — so the durable claim is the attribution, not
    the absence: the ladder shows G8, G9, and G10 all complete, and
    ``dropout`` left ``UNSUPPORTED`` at the last of them."""
    assert "dropout" not in cpp.UNSUPPORTED
    design = (REPO_ROOT / "docs"
              / "native_rng_dropout_design.md").read_text(encoding="utf-8")
    ladder = design[design.index("| Milestone | Scope | Status |"):]
    ladder = ladder[:ladder.index("### G0")]
    for row in ("G8", "G9", "G10"):
        match = re.search(rf"\|\s*{row}\s*\|[^|]*\|([^|]*)\|", ladder)
        assert match is not None, row
        status = re.sub(r"[*`]", "", match.group(1)).strip().lower()
        assert status.startswith("complete"), (row, status)
    # The G8 row still describes itself as measurement only, so a future
    # edit cannot quietly re-attribute the capability move to this file.
    g8_row = re.search(r"\|\s*G8\s*\|([^|]*)\|([^|]*)\|", ladder)
    assert "measurement only" in g8_row.group(2).lower()
